"""Pace v6: the top speed must be both reachable and steerable.

Two things were wrong at once:

  * the braking model assumed 2.8 m/s^2 while the simulator changes speed
    within one 65ms step, so every corner sat inside a 3-5m braking horizon
    and the target speed never reached 4.3 on a track with 2m straights;
  * the 4.3 m/s tier only existed at +/-5 and 0 degrees of steering, so it
    could not be used anywhere that needed a real turn.

v6 fixes the first with BRAKE_ACCEL = 12.0 and a shorter horizon clamp, and
the second with a rectangular 5 speeds x 5 steering grid (25 actions), which
means the model can no longer warm start -- it has to be trained from zero.
"""
import importlib.util
import json
import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "custom_files/speed_2018_pace_v6/reward_function.py"
spec = importlib.util.spec_from_file_location("v6", MODULE)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)


def straight(speed, steering=0.0, heading=0.0):
    points = [(i * 0.25, 0.0) for i in range(160)]
    return dict(x=10.0, y=0.0, heading=heading, steering_angle=steering, speed=speed,
                track_width=1.07, distance_from_center=0.0, progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[40, 41], waypoints=points, track_length=23.12)


def arc(radius, speed, steering=0.0):
    points = [(radius * math.cos(i * 2 * math.pi / 160),
               radius * math.sin(i * 2 * math.pi / 160)) for i in range(160)]
    return dict(x=radius, y=0.0, heading=90.0, steering_angle=steering, speed=speed,
                track_width=1.07, distance_from_center=0.0, progress=50.0, steps=100,
                all_wheels_on_track=True, is_offtrack=False, is_crashed=False,
                closest_waypoints=[0, 1], waypoints=points, track_length=23.12)


class PaceV6Tests(unittest.TestCase):
    def test_action_space_is_a_full_grid(self):
        metadata = json.loads((MODULE.parent / "model_metadata.json").read_text())
        pairs = {(a["steering_angle"], a["speed"]) for a in metadata["action_space"]}
        speeds = sorted({s for _, s in pairs})
        steers = sorted({st for st, _ in pairs})
        self.assertEqual(pairs, {(st, sp) for sp in speeds for st in steers})
        self.assertEqual(len(pairs), 25)
        # The 4.3 tier must be steerable, not straight-line only.
        self.assertIn((15, 4.3), pairs)
        self.assertIn((-30, 4.3), pairs)

    def test_braking_horizon_is_shorter_than_before(self):
        self.assertLess(v6.BRAKE_ACCEL, 20.0)
        self.assertGreater(v6.BRAKE_ACCEL, 2.8 * 3)
        old = 4.3 ** 2 / (2 * 2.8) + 1.0
        new = min(2.4, 4.3 ** 2 / (2 * v6.BRAKE_ACCEL) + 0.6)
        self.assertLess(new, old / 2)

    def test_top_speed_wins_on_a_clear_straight(self):
        top = v6.reward_function(straight(4.3))
        mid = v6.reward_function(straight(2.2))
        self.assertGreater(top, 3 * mid)

    def test_corner_cap_follows_curvature_not_top_speed(self):
        # k = 0.5 demands sqrt(LATERAL_ACCEL / k) = 3.0 m/s. The cap must sit
        # there: above it the overspeed penalty bites, and 4.3 must lose badly.
        scores = {v: v6.reward_function(arc(2.0, v, steering=30.0)) for v in (1.4, 2.2, 3.0, 4.3)}
        limit = math.sqrt(v6.LATERAL_ACCEL / 0.5)
        best = max(scores, key=scores.get)
        self.assertEqual(best, min(scores, key=lambda v: abs(v - limit)))
        self.assertLess(scores[4.3], scores[3.0] / 5)

    def test_off_track_is_terminal(self):
        params = straight(4.3)
        params["is_offtrack"] = True
        self.assertEqual(v6.reward_function(params), v6.MIN_REWARD)

    def test_fuzz_stays_bounded(self):
        random.seed(3)
        speeds = [1.4, 2.2, 3.0, 3.8, 4.3]
        for _ in range(300):
            params = dict(
                x=random.uniform(-5, 5), y=random.uniform(-1, 1),
                heading=random.uniform(-180, 180), steering_angle=random.choice([-30, -15, 0, 15, 30]),
                speed=random.choice(speeds), track_width=1.07,
                distance_from_center=random.uniform(0, 0.53),
                progress=random.uniform(0, 100), steps=random.randint(1, 500),
                all_wheels_on_track=random.random() > 0.1,
                is_offtrack=random.random() < 0.05, is_crashed=random.random() < 0.05,
                closest_waypoints=[random.randint(0, 159), random.randint(0, 159)],
                waypoints=[(random.uniform(-8, 8), random.uniform(-8, 8)) for _ in range(30)],
                track_length=23.12)
            value = v6.reward_function(params)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, v6.MIN_REWARD)
            self.assertLessEqual(value, v6.MAX_REWARD)


if __name__ == "__main__":
    unittest.main()
