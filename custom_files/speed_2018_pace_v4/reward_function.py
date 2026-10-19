"""Pace v4: unlock the speed the action space already has.

v3 capped the target speed three times over: by alignment, by a heuristic
curvature target, and by hard steering-angle thresholds. On a 17.7m track
that produced 83% of the lap at 1.0-1.55 m/s and 3% at 4.3 m/s, while the
same reward still scored 4.3 m/s seven times higher than 1.55 on a straight.

Changes against v3:
  * lookahead grows with speed (0.30 + 0.30*v) instead of hugging 0.6*width,
    so pure-pursuit geometry stops asking for 25-30 degrees at 4 m/s;
  * the alignment term no longer scales the speed limit, and the
    "steering >= 12/25 degrees -> cap 2.55/1.55" rules are gone;
  * the speed-match kernel is wider (1.2 m/s) and the overspeed falloff is
    less of a cliff (0.9 m/s), keeping a real penalty without teaching fear;
  * lane keeping is relaxed in tight bends so an apex line is affordable;
  * a mild under-speed factor replaces the old implicit reward for crawling.

Pure geometry, no I/O, no cross-episode state. Not a guaranteed 8-second
policy: it moves the ceiling, it does not prove the ceiling.
"""
import math

MAX_SPEED = 4.3
MIN_SPEED = 1.25
MAX_REWARD = 40.0
MIN_REWARD = 0.001
WHEELBASE = 0.164
MAX_STEERING = 30.0


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _angle_difference(target, heading):
    return (target - heading + 180.0) % 360.0 - 180.0


def _bearing(start, end):
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _point_ahead(points, origin, next_index, direction, distance):
    current = origin
    for offset in range(len(points) + 1):
        following = points[(next_index + direction * offset) % len(points)]
        segment = math.hypot(following[0] - current[0], following[1] - current[1])
        if segment > 1e-9:
            if distance <= segment:
                ratio = distance / segment
                return (current[0] + ratio * (following[0] - current[0]),
                        current[1] + ratio * (following[1] - current[1]))
            distance -= segment
        current = following
    return current


def _target_speed(points, projected, following, direction, speed):
    """Curvature profile over the distance needed to brake from the current speed.

    A fixed 2.2m preview only covers half a second at 4.3 m/s, which is why the
    old target looked safe until the corner arrived. The horizon here follows
    v^2/(2a), so the car is told to slow down before it has to.
    """
    spacing = 0.25
    horizon = _clamp(speed * speed / (2.0 * 2.5) + 1.0, 1.2, 4.0)
    steps = int(horizon / spacing) + 1
    samples = [_point_ahead(points, projected, following, direction, spacing * i)
               for i in range(steps + 2)]
    target = MAX_SPEED
    for i in range(steps):
        heading_in = _bearing(samples[i], samples[i + 1])
        heading_out = _bearing(samples[i + 1], samples[i + 2])
        curvature = abs(math.radians(_angle_difference(heading_out, heading_in))) / spacing
        corner_speed = _clamp(math.sqrt(3.6 / max(curvature, 1e-6)), MIN_SPEED, MAX_SPEED)
        # Distance from the car to this sample, not from the preview start.
        permitted = math.sqrt(corner_speed ** 2 + 2.0 * 2.5 * spacing * i)
        target = min(target, permitted)
    return target


def reward_function(params):
    if params.get("is_offtrack", False) or params.get("is_crashed", False):
        return MIN_REWARD
    try:
        x, y = float(params["x"]), float(params["y"])
        heading, steering = float(params["heading"]), float(params["steering_angle"])
        width = float(params["track_width"])
        distance = float(params["distance_from_center"])
        speed = float(params["speed"])
        progress = float(params.get("progress", 0.0))
        steps = float(params.get("steps", 0.0))
        if not all(math.isfinite(v) for v in (x, y, heading, steering, width, distance, speed, progress, steps)):
            return MIN_REWARD
        if width <= 0 or distance < 0 or speed <= 0 or steps < 0 or not 0 <= progress <= 100:
            return MIN_REWARD
        points = [(float(p[0]), float(p[1])) for p in params["waypoints"]]
        if len(points) < 3 or not all(math.isfinite(v) for p in points for v in p):
            return MIN_REWARD
        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) < 1e-9:
            points = points[:-1]
        if len(points) < 3:
            return MIN_REWARD
        previous, following = [int(i) % len(points) for i in params["closest_waypoints"]]
    except (KeyError, ValueError, TypeError, OverflowError, IndexError):
        return MIN_REWARD
    delta = (following - previous) % len(points)
    if delta == 0:
        return MIN_REWARD
    direction = 1 if delta <= len(points) / 2 else -1
    behind, ahead = points[previous], points[following]
    dx, dy = ahead[0] - behind[0], ahead[1] - behind[1]
    squared = dx * dx + dy * dy
    if squared < 1e-12:
        return MIN_REWARD
    fraction = _clamp(((x - behind[0]) * dx + (y - behind[1]) * dy) / squared, 0, 1)
    projected = (behind[0] + fraction * dx, behind[1] + fraction * dy)
    preview = _point_ahead(points, projected, following, direction, 0.8 * width)
    bend = abs(_angle_difference(_bearing(projected, preview), _bearing(behind, ahead)))
    # Pure pursuit needs a lookahead that grows with speed: 0.77m at 4.3 m/s
    # forces 25 degrees of steering, which v3 then punished as "too fast".
    lookahead = _clamp((0.30 + 0.30 * speed) / (1.0 + bend / 75.0), 0.45, 1.70)
    target = _point_ahead(points, projected, following, direction, lookahead)
    target_distance = math.hypot(target[0] - x, target[1] - y)
    if target_distance < 1e-9:
        return MIN_REWARD
    error = _angle_difference(_bearing((x, y), target), heading)
    if abs(error) > 100.0:
        return MIN_REWARD
    desired = _clamp(math.degrees(math.atan2(
        2.0 * WHEELBASE * math.sin(math.radians(error)), target_distance)),
        -MAX_STEERING, MAX_STEERING)
    direction_score = math.exp(-((error / 25.0) ** 2))
    steering_score = math.exp(-(((steering - desired) / 15.0) ** 2))
    edge = _clamp(distance / (0.5 * width), 0.0, 1.0)
    lane_score = 1.0 - _clamp((edge - 0.25) / 0.75, 0.0, 1.0) ** 2
    # Straight: keep the centre line. Tight bend: an apex line must stay affordable.
    apex = _clamp(bend / 55.0, 0.0, 1.0)
    lane_keep = 0.75 - 0.45 * apex
    quality = 0.1 + 1.8 * direction_score + 0.6 * steering_score + 0.5 * lane_score
    quality *= 0.25 + lane_keep * lane_score
    safe_speed = _target_speed(points, projected, following, direction, speed)
    # Heading alone is not a reason to slow down; heading pointed at the edge is.
    # At 4.3 m/s a 15-degree error leaves a 0.76m track in about 0.1s, so the
    # remaining width, not the angle, sets the ceiling.
    room = max(0.0, 0.5 * width - distance)
    lateral = abs(speed * math.sin(math.radians(error)))
    if lateral > 1e-6:
        safety = _clamp((room / lateral) / 0.8, 0.0, 1.0)
        safe_speed = min(safe_speed, MIN_SPEED + (MAX_SPEED - MIN_SPEED) * safety)
    target_match = math.exp(-(((speed - safe_speed) / 1.2) ** 2))
    overspeed = max(0.0, speed - safe_speed)
    speed_score = (0.25 + 1.5 * target_match) * math.exp(-((overspeed / 0.9) ** 2))
    # A bounded cumulative progress/steps term discourages slow reward farming.
    # Missing steps are accepted for offline callers; the simulator supplies it.
    pace = _clamp((progress / steps) / 0.8, 0.0, 1.5) if steps > 0 else 1.0
    reward = quality * speed_score * (0.4 + 0.6 * pace)
    # Crawling on a clear straight is the habit v3 accidentally taught.
    if abs(error) < 12.0 and speed < safe_speed * 0.85:
        reward *= 0.85
    if progress >= 100.0:
        reward += 10.0 + 15.0 * pace
    if not params.get("all_wheels_on_track", True):
        reward *= 0.10
    return float(_clamp(reward, MIN_REWARD, MAX_REWARD))
