"""Geometry/scoring tests, not claims of racing performance."""
import importlib.util
import json
import math
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "custom_files/speed_2018_v1/reward_function.py"
spec = importlib.util.spec_from_file_location("speed_reward", MODULE)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def straight():
    return dict(x=1.0, y=0.0, heading=0.0, steering_angle=0,
                track_width=0.76, distance_from_center=0.0, speed=1.2,
                progress=50.0, all_wheels_on_track=True,
                closest_waypoints=[1, 2], waypoints=[(0, 0), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (5, 4), (0, 4)])


def circle():
    points = [(math.cos(i * math.pi / 60), math.sin(i * math.pi / 60)) for i in range(120)]
    return dict(straight(), x=1.0, heading=90.0, steering_angle=15,
                waypoints=points, closest_waypoints=[0, 1])


class SpeedRewardTests(unittest.TestCase):
    def test_actions_unique_valid(self):
        meta = json.loads((MODULE.parent / "model_metadata.json").read_text())
        actions = [(a["steering_angle"], a["speed"]) for a in meta["action_space"]]
        self.assertEqual(len(actions), 11)
        self.assertEqual(len(set(actions)), 11)
        self.assertTrue(all(-30 <= a <= 30 and 1 <= s <= 3 for a, s in actions))

    def test_safe_straight_rewards_speed(self):
        p = straight()
        scores = [reward.reward_function(dict(p, speed=s)) for s in (1.2, 2.0, 3.0)]
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_corner_discourages_overspeed(self):
        p = circle()
        self.assertGreater(reward.reward_function(dict(p, speed=1.2)),
                           reward.reward_function(dict(p, speed=3.0)))

    def test_edge_penalty(self):
        p = straight()
        self.assertLess(reward.reward_function(dict(p, y=0.34, distance_from_center=0.34)),
                        reward.reward_function(p))
        self.assertLess(reward.reward_function(dict(p, all_wheels_on_track=False)),
                        reward.reward_function(p))

    def test_terminal_invalid(self):
        p = straight()
        for field in ("is_offtrack", "is_crashed"):
            self.assertEqual(reward.reward_function(dict(p, **{field: True})), 0.001)
        for field in ("speed", "heading", "x", "track_width"):
            self.assertEqual(reward.reward_function(dict(p, **{field: float("nan")})), 0.001)
        self.assertEqual(reward.reward_function({}), 0.001)
        self.assertEqual(reward.reward_function(dict(p, speed=0)), 0.001)

    def test_wrong_direction(self):
        self.assertEqual(reward.reward_function(dict(straight(), heading=180)), 0.001)

    def test_reverse_supported(self):
        p = circle()
        forward = reward.reward_function(p)
        backward = reward.reward_function(dict(p, closest_waypoints=[0, 119], heading=-90,
                                               steering_angle=-15, is_reversed=True))
        self.assertAlmostEqual(forward, backward)

    def test_finish_bonus(self):
        self.assertGreater(reward.reward_function(dict(straight(), progress=100)),
                           reward.reward_function(straight()))

    def test_random_finite(self):
        rng = random.Random(2018)
        for _ in range(3000):
            p = dict(circle(), speed=rng.uniform(0.1, 4),
                     steering_angle=rng.uniform(-30, 30), heading=rng.uniform(-180, 180),
                     distance_from_center=rng.uniform(0, 0.6), progress=rng.uniform(0, 100))
            value = reward.reward_function(p)
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value) and 0.001 <= value <= 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
