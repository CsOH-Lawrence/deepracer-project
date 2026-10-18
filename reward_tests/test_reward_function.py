import copy
import math
import random
import unittest
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / 'custom_files/reward_function.py'
_spec = importlib.util.spec_from_file_location('lookahead_reward', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MIN_REWARD = _module.MIN_REWARD
_angle_difference = _module._angle_difference
_point_ahead = _module._point_ahead
reward_function = _module.reward_function


def straight(**changes):
    data = dict(x=0.5, y=0.0, heading=0.0, steering_angle=0.0,
                track_width=0.8, distance_from_center=0.0, progress=10.0,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                is_reversed=False, speed=0.6, steps=50,
                waypoints=[(0, 0), (1, 0), (2, 0), (2, 3), (0, 3)],
                closest_waypoints=[0, 1])
    data.update(changes)
    return data


class RewardTests(unittest.TestCase):
    def test_center_beats_edge(self):
        self.assertGreater(reward_function(straight()), reward_function(straight(distance_from_center=0.36)))

    def test_partial_wheels_are_penalized_not_terminal(self):
        base = reward_function(straight())
        partial = reward_function(straight(all_wheels_on_track=False))
        self.assertAlmostEqual(partial, base * 0.15)
        self.assertGreater(partial, MIN_REWARD)

    def test_offtrack_and_crash(self):
        for field in ("is_offtrack", "is_crashed"):
            self.assertEqual(reward_function(straight(**{field: True})), MIN_REWARD)

    def test_wrong_heading(self):
        self.assertGreater(reward_function(straight()), reward_function(straight(heading=50)))
        self.assertEqual(reward_function(straight(heading=180)), MIN_REWARD)

    def test_straight_steering(self):
        self.assertGreater(reward_function(straight()), reward_function(straight(steering_angle=30)))

    def test_appropriate_large_turn_is_rewarded(self):
        # 位于左弯入口；30度左转比直行或反向右转得分高。
        curve = straight(x=0.9, waypoints=[(0, 0), (1, 0), (1.2, 0.3), (1.2, 1), (0, 1)])
        left = reward_function(dict(curve, steering_angle=30))
        self.assertGreater(left, reward_function(dict(curve, steering_angle=0)))
        self.assertGreater(left, reward_function(dict(curve, steering_angle=-30)))

    def test_finish_bonus(self):
        self.assertAlmostEqual(reward_function(straight(progress=100)), reward_function(straight(progress=99)) + 1.0)

    def test_no_false_reverse_penalty(self):
        self.assertEqual(reward_function(straight(is_reversed=True)), reward_function(straight()))

    def test_reverse_index_order(self):
        reversed_state = straight(heading=180, closest_waypoints=[1, 0], is_reversed=True)
        self.assertAlmostEqual(reward_function(reversed_state), reward_function(straight()))

    def test_closed_duplicate_and_finish_wrap(self):
        regular = straight()
        duplicate = copy.deepcopy(regular)
        duplicate["waypoints"].append(duplicate["waypoints"][0])
        self.assertEqual(reward_function(regular), reward_function(duplicate))
        wrap = straight(x=0, y=0.05, heading=-90, closest_waypoints=[4, 0])
        self.assertTrue(math.isfinite(reward_function(wrap)))

    def test_angle_wrap(self):
        self.assertEqual(_angle_difference(-179, 179), 2)
        self.assertEqual(_angle_difference(179, -179), -2)

    def test_arc_length_interpolation(self):
        p = _point_ahead([(0, 0), (1, 0), (1, 1)], (0.5, 0), 1, 1, 0.75)
        self.assertEqual(p, (1.0, 0.25))

    def test_no_input_mutation(self):
        params = straight()
        before = copy.deepcopy(params)
        reward_function(params)
        self.assertEqual(params, before)

    def test_bad_data(self):
        for state in ({}, straight(track_width=0), straight(heading=float("nan")),
                      straight(waypoints=[(0, 0)]), straight(closest_waypoints=[0, 0]),
                      straight(waypoints=[(0, 0), (0, 0), (0, 0)])):
            self.assertEqual(reward_function(state), MIN_REWARD)

    def test_random_states_are_positive_finite_bounded(self):
        rng = random.Random(20261012)
        for _ in range(3000):
            state = straight(x=rng.uniform(0, 1), y=rng.uniform(-0.4, 0.4),
                             heading=rng.uniform(-180, 180), steering_angle=rng.choice([-30, -15, 0, 15, 30]),
                             distance_from_center=rng.uniform(0, 0.45), progress=rng.uniform(0, 100),
                             all_wheels_on_track=rng.choice([True, False]))
            result = reward_function(state)
            self.assertIsInstance(result, float)
            self.assertTrue(math.isfinite(result))
            self.assertGreaterEqual(result, MIN_REWARD)
            self.assertLessEqual(result, 4.0)


if __name__ == "__main__":
    unittest.main()
