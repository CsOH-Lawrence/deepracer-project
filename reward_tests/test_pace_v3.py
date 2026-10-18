"""Offline safety/geometry tests; not proof of race performance."""
import importlib.util
import json
import math
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "custom_files/speed_2018_pace_v3/reward_function.py"
spec = importlib.util.spec_from_file_location("corner_reward", MODULE)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def straight():
    return dict(x=1.0, y=0.0, heading=0.0, steering_angle=0,
                track_width=0.76, distance_from_center=0.0, speed=2.55,
                progress=50.0, steps=100, all_wheels_on_track=True,
                closest_waypoints=[1, 2], waypoints=[(0, 0), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (5, 4), (0, 4)])


def circle():
    points = [(math.cos(i * math.pi / 60), math.sin(i * math.pi / 60)) for i in range(120)]
    return dict(straight(), x=1.0, heading=90.0, steering_angle=15,
                waypoints=points, closest_waypoints=[0, 1])


class PaceTests(unittest.TestCase):
    def test_action_head_and_angles_preserved(self):
        new = json.loads((MODULE.parent / "model_metadata.json").read_text())
        self.assertEqual(len(new["action_space"]), 11)
        self.assertEqual([-30, -15, 0, 15, 30, -15, 0, 15, -5, 0, 5],
                         [a["steering_angle"] for a in new["action_space"]])
        self.assertEqual([a["speed"] for a in new["action_space"]],
                         [1.25, 1.55, 1.55, 1.55, 1.25, 2.55, 2.55, 2.55, 4.3, 4.3, 4.3])
        expected = dict(sensor=["FRONT_FACING_CAMERA"], neural_network="DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
                        training_algorithm="clipped_ppo", action_space_type="discrete", version="5")
        for key, value in expected.items():
            self.assertEqual(value, new[key])

    def test_straight_can_choose_fast_action(self):
        p = straight()
        scores = [reward.reward_function(dict(p, speed=s)) for s in (1.55, 2.55, 4.3)]
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_curve_discourages_fast_action(self):
        self.assertGreater(reward.reward_function(dict(circle(), speed=1.55)),
                           reward.reward_function(dict(circle(), speed=4.3)))

    def test_brakes_before_corner(self):
        points = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        earlier = reward._target_speed(points, (0.5, 0), 1, 1)
        closer = reward._target_speed(points, (1.7, 0), 2, 1)
        self.assertLess(earlier, 4.3)
        self.assertLess(closer, earlier)

    def test_edge_partial_wheels_and_heading_penalty(self):
        p = straight()
        self.assertLess(reward.reward_function(dict(p, y=.34, distance_from_center=.34)), reward.reward_function(p))
        self.assertLess(reward.reward_function(dict(p, all_wheels_on_track=False)), reward.reward_function(p))
        self.assertLess(reward.reward_function(dict(p, heading=40)), reward.reward_function(p))

    def test_pace_and_finish(self):
        p = straight()
        self.assertGreater(reward.reward_function(dict(p, steps=80)), reward.reward_function(dict(p, steps=200)))
        self.assertGreater(reward.reward_function(dict(p, progress=100)), reward.reward_function(p))

    def test_reverse_and_closed_duplicate(self):
        p = circle()
        self.assertAlmostEqual(reward.reward_function(p), reward.reward_function(dict(
            p, closest_waypoints=[0, 119], heading=-90, steering_angle=-15, is_reversed=True)))
        self.assertAlmostEqual(reward.reward_function(p), reward.reward_function(dict(p, waypoints=p["waypoints"] + [p["waypoints"][0]])))

    def test_invalid_and_terminal(self):
        p = straight()
        self.assertEqual(reward.reward_function({}), .001)
        for field in ("is_offtrack", "is_crashed"):
            self.assertEqual(reward.reward_function(dict(p, **{field: True})), .001)
        for field in ("speed", "heading", "steps", "x", "track_width"):
            self.assertEqual(reward.reward_function(dict(p, **{field: float("nan")})), .001)
        for bad in ({"steps": -1}, {"progress": 101}, {"speed": 0}, {"track_width": 0}, {"heading": 180}):
            self.assertEqual(reward.reward_function(dict(p, **bad)), .001)

    def test_fuzz_bounded(self):
        rng = random.Random(20261017)
        p = circle()
        for _ in range(4000):
            result = reward.reward_function(dict(p, speed=rng.uniform(.1, 5),
                heading=rng.uniform(-180, 180), steering_angle=rng.uniform(-30, 30),
                distance_from_center=rng.uniform(0, .6), progress=rng.uniform(0, 100), steps=rng.randint(1, 500)))
            self.assertTrue(math.isfinite(result) and .001 <= result <= 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
