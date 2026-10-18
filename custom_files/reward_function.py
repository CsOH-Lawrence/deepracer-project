"""Advanced v1: 前瞻路点 + 方向对齐 + 安全边界，适配当前固定 0.6m/s 动作空间。

这是独立实现的待对照实验版本，不是已验证的冠军奖励函数。
设计参考（借鉴思路，不复制实现）：
https://github.com/VilemR/AWS_DeepRacer
https://github.com/dp770/aws_deepracer_worksheet
https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/create-a-model.html

仅依赖 math；不读取文件、不访问网络、不维护跨回合状态。
"""
import math

MIN_REWARD = 0.001
# 从当前 deepracer_single_cam.urdf 的前后轮轴位置核实：0.163974m。
WHEELBASE = 0.164
MAX_STEERING = 30.0


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _angle_difference(target, heading):
    """有符号角度差，正确处理 -180/180 度跨界。"""
    return (target - heading + 180.0) % 360.0 - 180.0


def _bearing(start, end):
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _point_ahead(points, origin, next_index, direction, distance):
    """沿中心线按米数前瞻并插值，而不是固定跳几个路点。"""
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
    return current  # 防止异常、重合路点产生死循环。


def reward_function(params):
    # 完全出界、碰撞：最低奖励。部分轮子越界在后面单独降分。
    if params.get("is_offtrack", False) or params.get("is_crashed", False):
        return MIN_REWARD

    try:
        x, y = float(params["x"]), float(params["y"])
        heading = float(params["heading"])
        steering = float(params["steering_angle"])
        width = float(params["track_width"])
        distance = float(params["distance_from_center"])
        progress = float(params.get("progress", 0.0))
        values = (x, y, heading, steering, width, distance, progress)
        if not all(math.isfinite(v) for v in values) or width <= 0 or distance < 0:
            return MIN_REWARD
        points = [(float(p[0]), float(p[1])) for p in params["waypoints"]]
        if len(points) < 3 or not all(math.isfinite(v) for p in points for v in p):
            return MIN_REWARD
        # 闭合赛道经常重复首点；去掉重复点，保证起终点附近前瞻正确。
        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) < 1e-9:
            points = points[:-1]
        if len(points) < 3:
            return MIN_REWARD
        previous, following = [int(i) % len(points) for i in params["closest_waypoints"]]
    except (KeyError, ValueError, TypeError, OverflowError, IndexError):
        return MIN_REWARD

    # 依据前后路点顺序判断行驶方向，不把 is_reversed=True 错当成倒车。
    delta = (following - previous) % len(points)
    if delta == 0:
        return MIN_REWARD
    direction = 1 if delta <= len(points) / 2 else -1
    behind, ahead = points[previous], points[following]
    dx, dy = ahead[0] - behind[0], ahead[1] - behind[1]
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-12:
        return MIN_REWARD
    fraction = _clamp(((x - behind[0]) * dx + (y - behind[1]) * dy) / length_squared, 0.0, 1.0)
    projected = (behind[0] + fraction * dx, behind[1] + fraction * dy)

    # 先估计前方弯曲程度：弯急时缩短前瞻，避免目标点跨过弯心。
    tangent_heading = _bearing(behind, ahead)
    preview = _point_ahead(points, projected, following, direction, 0.8 * width)
    bend = abs(_angle_difference(_bearing(projected, preview), tangent_heading))
    lookahead = _clamp(0.7 * width / (1.0 + bend / 45.0), 0.20, 0.80)
    target = _point_ahead(points, projected, following, direction, lookahead)
    target_distance = math.hypot(target[0] - x, target[1] - y)
    if target_distance < 1e-9:
        return MIN_REWARD
    heading_error = _angle_difference(_bearing((x, y), target), heading)
    if abs(heading_error) > 100.0:
        return MIN_REWARD

    # Pure-pursuit 几何仅用于评分，不直接控制小车；策略仍由 PPO 学习。
    desired_steering = math.degrees(math.atan2(
        2.0 * WHEELBASE * math.sin(math.radians(heading_error)), target_distance))
    desired_steering = _clamp(desired_steering, -MAX_STEERING, MAX_STEERING)
    direction_score = math.exp(-((heading_error / 30.0) ** 2))
    steering_score = math.exp(-(((steering - desired_steering) / 15.0) ** 2))

    # 连续边缘降分取代原来的三档跳变；不是要求每一步死守中心线。
    normalized_distance = _clamp(distance / (0.5 * width), 0.0, 1.0)
    lane_score = 1.0 - normalized_distance ** 2
    reward = 0.1 + 1.8 * direction_score + 0.6 * steering_score + 0.5 * lane_score
    reward *= 0.25 + 0.75 * lane_score

    # 单回合末尾奖励；不使用总 progress/steps，避免随机起点或历史效率干扰。
    if progress >= 100.0:
        reward += 1.0
    if not params.get("all_wheels_on_track", True):
        reward *= 0.15  # 留恢复信号，但不鼓励长期压着边线跑。

    # 全部动作速度固定，当前故意不加无效的加速/减速奖励。
    return float(_clamp(reward, MIN_REWARD, 4.0))
