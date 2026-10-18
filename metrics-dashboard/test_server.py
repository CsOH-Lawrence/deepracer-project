import unittest
import threading
from unittest.mock import Mock
from botocore.exceptions import ClientError
from server import Metrics, normalize


class MetricsTests(unittest.TestCase):
    def test_phases_and_simulation_time(self):
        sample = {"metrics": [
            {"episode": 20, "phase": "evaluation", "reward_score": 10, "completion_percentage": 50, "metric_time": 3000},
            {"episode": 1, "phase": "training", "reward_score": 19, "completion_percentage": 12, "metric_time": 1000, "elapsed_time_in_milliseconds": 5322},
            {"episode": 20, "phase": "evaluation", "reward_score": 11, "completion_percentage": 60, "metric_time": 4000},
        ]}
        rows = normalize(sample)
        self.assertEqual(len(rows), 3)  # Keep distinct evaluation trials.
        self.assertEqual(rows[0]["seconds"], 5.322)
        self.assertEqual(rows[0]["episode"], 1)
        self.assertEqual(rows[1]["phase"], "evaluation")

    def test_bad_rows_and_empty(self):
        self.assertEqual(normalize({}), [])
        self.assertEqual(normalize({"metrics": [{}, {"episode": 1, "reward_score": "nan", "completion_percentage": 4}]}), [])

    def test_clock_rewind_does_not_reorder_episodes(self):
        sample = {"metrics": [
            {"episode": 466, "reward_score": 90, "completion_percentage": 0, "metric_time": 3429028},
            {"episode": 467, "reward_score": 30, "completion_percentage": 0, "metric_time": 27266},
            {"episode": 468, "reward_score": 702, "completion_percentage": 100, "metric_time": 3441598},
        ]}
        rows = normalize(sample)
        self.assertEqual([r["episode"] for r in rows], [466, 467, 468])
        self.assertEqual(rows[1]["sim_time_ms"], 27266)  # Raw anomaly remains visible.

    def test_training_before_same_episode_evaluations(self):
        sample = {"metrics": [
            {"episode": 440, "phase": "evaluation", "reward_score": 461, "completion_percentage": 100},
            {"episode": 440, "phase": "training", "reward_score": 375, "completion_percentage": 100},
            {"episode": 440, "phase": "evaluation", "reward_score": 437, "completion_percentage": 100},
        ]}
        rows = normalize(sample)
        self.assertEqual([r["phase"] for r in rows], ["training", "evaluation", "evaluation"])
        self.assertEqual([r["reward"] for r in rows], [375, 461, 437])

    def make_metrics(self, code):
        metrics = Metrics.__new__(Metrics)
        metrics.client = Mock()
        metrics.client.get_object.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
        metrics.bucket, metrics.prefix = "bucket", "new-model"
        metrics.lock, metrics.cached, metrics.checked = threading.Lock(), None, 0
        return metrics

    def test_new_model_waits_without_writing_s3(self):
        result = self.make_metrics("NoSuchKey").get()
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["model"], "new-model")
        self.assertEqual(result["rows"], [])

    def test_auth_failure_is_not_disguised_as_waiting(self):
        with self.assertRaises(ClientError):
            self.make_metrics("AccessDenied").get()


if __name__ == "__main__":
    unittest.main()
