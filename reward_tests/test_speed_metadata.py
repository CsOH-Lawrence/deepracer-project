import json
import math
from pathlib import Path
import unittest
from speed_metadata import scale_all_speeds


class SpeedScalingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "custom_files/speed_2018_v1/model_metadata.json"
        self.metadata = json.loads(path.read_text())

    def test_all_four_speed_levels(self):
        new = scale_all_speeds(self.metadata, 1.3)
        self.assertEqual(sorted({a["speed"] for a in new["action_space"]}), [1.3, 1.56, 2.6, 3.9])

    def test_angles_indices_and_network_preserved(self):
        new = scale_all_speeds(self.metadata, 1.3)
        self.assertEqual(len(new["action_space"]), 11)
        self.assertEqual([a["steering_angle"] for a in new["action_space"]], [a["steering_angle"] for a in self.metadata["action_space"]])
        for key in self.metadata.keys() - {"action_space"}:
            self.assertEqual(new[key], self.metadata[key])

    def test_source_not_mutated(self):
        before = json.dumps(self.metadata, sort_keys=True)
        new = scale_all_speeds(self.metadata, 1.3)
        new["sensor"].append("TEST")
        self.assertEqual(json.dumps(self.metadata, sort_keys=True), before)

    def test_invalid_factors(self):
        for factor in (0, -1, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                scale_all_speeds(self.metadata, factor)

    def test_invalid_source(self):
        for speed in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                scale_all_speeds({"action_space": [{"speed": speed}]}, 1.3)
        with self.assertRaises(ValueError):
            scale_all_speeds({"action_space": []}, 1.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
