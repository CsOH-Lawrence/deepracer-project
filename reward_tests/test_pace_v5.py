"""Pace v5 hypothesis tests: each test pins one of H1-H3 from the docstring."""
import importlib.util
import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5 = load(ROOT / "custom_files/speed_2018_pace_v5/reward_function.py", "v5")
v4 = load(ROOT / "custom_files/speed_2018_pace_v4/reward_function.py", "v4")


def straight(speed, heading=0.0, steering=0.0, offset=0.0):
    points = [(i * 0.25, 0.0) for i in range(120)]
    return dict(x=10.0, y=offset, heading=heading, steering_angle=steering, speed=speed,
                track_width=1.07, distance_from_center=abs(offset), progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[40, 41], waypoints=points, track_length=23.12)


def arc(radius, speed, steering=5.0):
    points = [(radius * math.cos(i * 2 * math.pi / 120),
               radius * math.sin(i * 2 * math.pi / 120)) for i in range(120)]
    return dict(x=radius, y=0.0, heading=90.0, steering_angle=steering, speed=speed,
                track_width=1.07, distance_from_center=0.0, progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[0, 1], waypoints=points, track_length=23.12)


class PaceV5Tests(unittest.TestCase):
    def test_action_ladder_matches_the_running_model(self):
        # Pins the ladder so a metadata edit cannot silently undo the speed work.
        import json
        metadata = json.loads((Path(__file__).resolve().parents[1]
                               / "custom_files/speed_2018_pace_v5/model_metadata.json").read_text())
        ladder = [(a["steering_angle"], a["speed"]) for a in metadata["action_space"]]
        self.assertEqual(ladder, [(-30, 2.0), (-15, 2.7), (0, 2.7), (15, 2.7), (30, 2.0),
                                  (-15, 3.6), (0, 3.6), (15, 3.6), (-5, 4.3), (0, 4.3), (5, 4.3)])

    def test_h1_medium_corner_is_faster_than_v4(self):
        # Same geometry, same speed: the higher grip target must pay more for 3.6 m/s.
        params = arc(3.0, 3.6)
        self.assertGreater(v5.reward_function(params), v4.reward_function(params))

    def test_h1_does_not_unlock_absurd_corner_speed(self):
        # A 2m radius corner is still a 2m radius corner.
        tight = {v: v5.reward_function(arc(2.0, v)) for v in (2.0, 2.7, 3.6)}
        self.assertEqual(max(tight, key=tight.get), 2.7)

    def test_h2_braking_zone_is_shorter(self):
        # With more braking authority the same corner can be approached faster.
        horizon_v4 = 4.3 ** 2 / (2 * 2.5) + 1.0
        horizon_v5 = 4.3 ** 2 / (2 * 2.8) + 1.0
        self.assertLess(horizon_v5, horizon_v4)

    def test_h3_aligned_but_steering_hard_is_penalised(self):
        straight_steer = v5.reward_function(straight(4.3, steering=30.0))
        straight_clean = v5.reward_function(straight(4.3, steering=0.0))
        self.assertLess(straight_steer, straight_clean)
        # The same action in a real corner must not be punished this way.
        corner_steer = v5.reward_function(arc(2.0, 2.0, steering=30.0))
        self.assertGreater(corner_steer, reward_floor(v5))

    def test_off_track_is_terminal(self):
        params = straight(4.3)
        params["is_offtrack"] = True
        self.assertEqual(v5.reward_function(params), v5.MIN_REWARD)

    def test_straight_still_prefers_top_speed(self):
        self.assertGreater(v5.reward_function(straight(4.3)), 3 * v5.reward_function(straight(2.0)))

    def test_fuzz_stays_bounded(self):
        random.seed(11)
        for _ in range(300):
            params = dict(
                x=random.uniform(-5, 5), y=random.uniform(-1, 1),
                heading=random.uniform(-180, 180), steering_angle=random.uniform(-30, 30),
                speed=random.choice([1.8, 2.4, 3.3, 4.3]),
                track_width=1.07, distance_from_center=random.uniform(0, 0.53),
                progress=random.uniform(0, 100), steps=random.randint(1, 500),
                all_wheels_on_track=random.random() > 0.1,
                is_offtrack=random.random() < 0.05, is_crashed=random.random() < 0.05,
                closest_waypoints=[random.randint(0, 119), random.randint(0, 119)],
                waypoints=[(random.uniform(-8, 8), random.uniform(-8, 8)) for _ in range(24)],
                track_length=23.12)
            value = v5.reward_function(params)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, v5.MIN_REWARD)
            self.assertLessEqual(value, v5.MAX_REWARD)


def reward_floor(module):
    return module.MIN_REWARD


if __name__ == "__main__":
    unittest.main()
