"""All-speed +30% continuation: same centerline reward, scaled speed envelope.

Pure geometry scoring, not a hard controller or a guaranteed 8-second policy.
Keep the old fixed-speed model as a completion/lap-time baseline.
Design references: AWS action-space/reward documentation and pure-pursuit geometry.
Only math is used; no I/O, external imports, or cross-episode state.
"""
import math

SPEED_SCALE = 1.3
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


def _target_speed(points, projected, following, direction):
    """Heuristic lateral acceleration/braking envelope, not calibrated physics.

    Inspect 1.75m ahead. Curvature implies lower corner speed; distance to
    that corner permits a higher current speed while leaving braking room.
    """
    spacing = 0.35
    samples = [projected] + [
        _point_ahead(points, projected, following, direction, spacing * i)
        for i in range(1, 6)
    ]
    headings = [_bearing(samples[i], samples[i + 1]) for i in range(5)]
    target = 3.0 * SPEED_SCALE
    for i in range(4):
        curvature = abs(math.radians(_angle_difference(headings[i + 1], headings[i]))) / spacing
        corner_speed = _clamp(math.sqrt(1.8 * SPEED_SCALE ** 2 / max(curvature, 1e-6)), 1.0 * SPEED_SCALE, 3.0 * SPEED_SCALE)
        # Conservative distance to the bend; slow before reaching its apex.
        permitted = math.sqrt(corner_speed ** 2 + 2.0 * 1.8 * SPEED_SCALE ** 2 * spacing * i)
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
        if not all(math.isfinite(v) for v in (x, y, heading, steering, width, distance, speed, progress)):
            return MIN_REWARD
        if width <= 0 or distance < 0 or speed <= 0:
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
    # Higher speed needs more reaction distance, but tight bends limit lookahead.
    lookahead = _clamp((0.7 * width + 0.10 * speed) / (1.0 + bend / 45.0), 0.25, 0.95)
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
    direction_score = math.exp(-((error / 30.0) ** 2))
    steering_score = math.exp(-(((steering - desired) / 15.0) ** 2))
    edge = _clamp(distance / (0.5 * width), 0.0, 1.0)
    lane_score = 1.0 - edge ** 2
    quality = (0.1 + 1.8 * direction_score + 0.6 * steering_score + 0.5 * lane_score)
    quality *= 0.25 + 0.75 * lane_score
    safe_speed = _target_speed(points, projected, following, direction)
    # Near edge or badly aligned: do not encourage full-speed recovery.
    safe_speed = min(safe_speed, _clamp(3.0 * SPEED_SCALE * direction_score * (0.5 + 0.5 * lane_score), 1.0 * SPEED_SCALE, 3.0 * SPEED_SCALE))
    overspeed = max(0.0, speed - safe_speed)
    speed_score = (_clamp(speed, 0.1, 3.0 * SPEED_SCALE) / (1.2 * SPEED_SCALE)) ** 1.5
    speed_score *= math.exp(-((overspeed / (0.45 * SPEED_SCALE)) ** 2))
    reward = quality * speed_score
    if progress >= 100.0:
        reward += 5.0
    if not params.get("all_wheels_on_track", True):
        reward *= 0.10
    return float(_clamp(reward, MIN_REWARD, 17.0))
