"""Pace v4 behaviour tests: the speed ceiling must move, the safety must stay.

Numbers here are comparative, not golden: they describe which action the
reward prefers in a given geometry. Offline preferences are not proof of
simulator behaviour; they only catch a reward that points the wrong way.
"""
import importlib.util
import json
import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "custom_files/speed_2018_pace_v4/reward_function.py"
spec = importlib.util.spec_from_file_location("pace_v4_reward", MODULE)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def straight(speed, heading=0.0, offset=0.0):
    points = [(i * 0.25, 0.0) for i in range(120)]
    return dict(x=10.0, y=offset, heading=heading, steering_angle=0.0, speed=speed,
                track_width=0.76, distance_from_center=abs(offset), progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[40, 41], waypoints=points, track_length=17.7)


def arc(radius, speed, steering=5.0):
    points = [(radius * math.cos(i * 2 * math.pi / 120),
               radius * math.sin(i * 2 * math.pi / 120)) for i in range(120)]
    return dict(x=radius, y=0.0, heading=90.0, steering_angle=steering, speed=speed,
                track_width=0.76, distance_from_center=0.0, progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[0, 1], waypoints=points, track_length=17.7)


def score(params):
    return reward.reward_function(params)


class PaceV4Tests(unittest.TestCase):
    def test_action_space_is_unchanged(self):
        metadata = json.loads((MODULE.parent / "model_metadata.json").read_text())
        self.assertEqual(len(metadata["action_space"]), 11)
        self.assertEqual(sorted({a["speed"] for a in metadata["action_space"]}), [1.25, 1.55, 2.55, 4.3])

    def test_straight_rewards_top_speed(self):
        # The whole point of v4: 4.3 m/s on a clear straight must dominate.
        self.assertGreater(score(straight(4.3)), 3 * score(straight(1.55)))

    def test_small_heading_error_does_not_throttle(self):
        # v3 collapsed to 0.01 here because of the steering-angle speed cap.
        self.assertGreater(score(straight(4.3, heading=5.0)), score(straight(1.55, heading=5.0)))

    def test_heading_at_the_edge_still_slows_down(self):
        # 15 degrees of error near the edge is a crash, not a corner entry.
        fast = score(straight(4.3, heading=15.0, offset=0.30))
        slow = score(straight(1.55, heading=15.0, offset=0.30))
        self.assertLess(fast, slow)

    def test_tight_corner_caps_speed(self):
        tight = [score(arc(2.0, v)) for v in (1.25, 1.55, 2.55, 4.3)]
        self.assertEqual(max(range(4), key=lambda i: tight[i]), 2)  # 2.55 wins

    def test_medium_corner_allows_one_tier_more_than_v3(self):
        # v3 pinned this geometry to 2.55; v4 should prefer 3.6.
        medium = [score(arc(3.0, v)) for v in (2.55, 3.6)]
        self.assertGreater(medium[1], medium[0])

    def test_off_track_and_crash_are_terminal(self):
        params = straight(4.3)
        params["is_offtrack"] = True
        self.assertEqual(score(params), reward.MIN_REWARD)
        params = straight(4.3)
        params["is_crashed"] = True
        self.assertEqual(score(params), reward.MIN_REWARD)

    def test_invalid_inputs_do_not_raise(self):
        for broken in (dict(straight(4.3), speed=0.0),
                       dict(straight(4.3), track_width=0.0),
                       dict(straight(4.3), waypoints=[(0, 0), (1, 0)]),
                       dict(straight(4.3), closest_waypoints=[0, 0])):
            self.assertEqual(score(broken), reward.MIN_REWARD)

    def test_finish_bonus_is_bounded(self):
        params = straight(4.3, heading=0.0)
        params["progress"] = 100.0
        self.assertLessEqual(score(params), reward.MAX_REWARD)

    def test_fuzz_stays_bounded(self):
        random.seed(7)
        for _ in range(400):
            params = dict(
                x=random.uniform(-5, 5), y=random.uniform(-1, 1),
                heading=random.uniform(-180, 180), steering_angle=random.uniform(-30, 30),
                speed=random.choice([1.25, 1.55, 2.55, 4.3]),
                track_width=0.76, distance_from_center=random.uniform(0, 0.38),
                progress=random.uniform(0, 100), steps=random.randint(1, 400),
                all_wheels_on_track=random.random() > 0.1,
                is_offtrack=random.random() < 0.05, is_crashed=random.random() < 0.05,
                closest_waypoints=[random.randint(0, 119), random.randint(0, 119)],
                waypoints=[(random.uniform(-6, 6), random.uniform(-6, 6)) for _ in range(24)],
                track_length=17.7)
            value = reward.reward_function(params)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, reward.MIN_REWARD)
            self.assertLessEqual(value, reward.MAX_REWARD)


if __name__ == "__main__":
    unittest.main()
