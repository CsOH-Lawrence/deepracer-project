import io
import json
import time
import unittest
import threading
from datetime import datetime, timezone
from unittest.mock import Mock
from botocore.exceptions import ClientError
from server import Metrics, normalize


def sample_document(count=5):
    return {"metrics": [
        {"episode": i, "phase": "training", "reward_score": i, "completion_percentage": 100,
         "metric_time": i * 1000, "elapsed_time_in_milliseconds": 1000,
         "episode_status": "Lap complete"}
        for i in range(1, count + 1)
    ]}


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
        # A list-shaped document is not the format this dashboard understands.
        self.assertEqual(normalize([{"episode": 1}]), [])

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

    def make_metrics(self, code=None, document=None, max_rows=100, cache_seconds=8.0):
        metrics = Metrics.__new__(Metrics)
        metrics.client = Mock()
        if code:
            metrics.client.get_object.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
        else:
            # A fresh body per call: the server closes it after reading, and a
            # reused BytesIO would fail on the second refresh.
            metrics.client.get_object.side_effect = lambda **_kwargs: {
                "Body": io.BytesIO(json.dumps(document or sample_document()).encode()),
                "LastModified": datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc),
                "ETag": '"v1"',
            }
        metrics.bucket, metrics.prefix = "bucket", "new-model"
        metrics.max_rows, metrics.cache_seconds = max_rows, cache_seconds
        metrics._lock, metrics._refreshing = threading.Lock(), threading.Lock()
        metrics._cached, metrics._checked = None, 0.0
        return metrics

    def test_new_model_waits_without_writing_s3(self):
        result = self.make_metrics("NoSuchKey").get()
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["model"], "new-model")
        self.assertEqual(result["rows"], [])
        self.assertFalse(result["truncated"])

    def test_auth_failure_is_not_disguised_as_waiting(self):
        with self.assertRaises(ClientError):
            self.make_metrics("AccessDenied").get()

    def test_cache_answers_without_a_second_read(self):
        metrics = self.make_metrics()
        first = metrics.get()
        second = metrics.get()
        self.assertEqual(first["rows"], second["rows"])
        self.assertEqual(metrics.client.get_object.call_count, 1)

    def test_refresh_is_conditional_on_the_previous_etag(self):
        metrics = self.make_metrics()
        metrics.get()
        metrics._checked = time.monotonic() - 100  # force a refresh
        metrics.get()
        request = metrics.client.get_object.call_args.kwargs
        self.assertEqual(request["IfNoneMatch"], '"v1"')

    def test_not_modified_reuses_rows_instead_of_failing(self):
        metrics = self.make_metrics()
        first = metrics.get()
        metrics._checked = time.monotonic() - 100
        metrics.client.get_object.side_effect = ClientError({"Error": {"Code": "304"}}, "GetObject")
        second = metrics.get()
        self.assertEqual(second["rows"], first["rows"])
        self.assertEqual(second["modified"], first["modified"])
        self.assertNotEqual(second["fetched"], first["fetched"])

    def test_stale_cache_is_served_while_a_refresh_is_running(self):
        metrics = self.make_metrics()
        stale = metrics.get()
        metrics._checked = time.monotonic() - 100
        metrics._refreshing.acquire()  # simulate another request talking to S3
        try:
            self.assertEqual(metrics.get(), stale)
        finally:
            metrics._refreshing.release()
        self.assertEqual(metrics.client.get_object.call_count, 1)

    def test_row_cap_keeps_the_latest_episodes(self):
        metrics = self.make_metrics(document=sample_document(50), max_rows=10)
        result = metrics.get()
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_rows"], 50)
        self.assertEqual(len(result["rows"]), 10)
        self.assertEqual([r["episode"] for r in result["rows"]], list(range(41, 51)))

    def test_full_history_is_not_flagged_as_truncated(self):
        metrics = self.make_metrics(document=sample_document(5), max_rows=10)
        result = metrics.get()
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total_rows"], 5)


if __name__ == "__main__":
    unittest.main()
