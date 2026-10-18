import json
import io
from pathlib import Path
import tempfile
import unittest

from sync_logs import Redactor, Store, model_prefix, overlap_since, trainer_prefix

TEST_ROOT = Path(__file__).resolve().parent / ".test-tmp"
TEST_ROOT.mkdir(exist_ok=True)

class SyncTests(unittest.TestCase):
    def test_secret_values_and_keyed_secrets_are_redacted(self):
        redactor = Redactor(["example-login-secret"])
        redactor.add_env(["AWS_SECRET_ACCESS_KEY=example-api-secret"])
        result = redactor.text("example-login-secret example-api-secret API_KEY=abc123")
        self.assertNotIn("example-login-secret", result)
        self.assertNotIn("example-api-secret", result)
        self.assertNotIn("abc123", result)

    def test_nonsecret_training_log_is_preserved(self):
        text = "2026-10-17T10:00:01Z Training iteration=5 reward=271 checkpoint=62"
        self.assertEqual(Redactor().text(text), text)

    def test_cursor_uses_overlap_across_date_boundary(self):
        self.assertEqual(overlap_since("2026-10-17T00:00:01.123456789Z"), "2026-10-16T23:59:59+00:00")
        self.assertIsNone(overlap_since(None))

    def test_model_selection_uses_exact_prefix(self):
        self.assertEqual(model_prefix(["SM_HP_S3_PREFIX=model-v3"]), "model-v3")
        self.assertEqual(model_prefix(['SM_HPS={"s3_prefix":"model-v4"}']), "model-v4")
        self.assertIsNone(model_prefix(["SM_HPS=not-json"]))

    def test_sagemaker_model_is_read_from_input_mount(self):
        class FakeSFTP:
            def open(self, path, mode):
                self_test.assertEqual(path, "/tmp/sagemaker/job/input/config/hyperparameters.json")
                return io.BytesIO(b'{"s3_prefix":"model-v3"}')
            def close(self):
                pass
        class FakeClient:
            def open_sftp(self):
                return FakeSFTP()
        self_test = self
        container = {"Name": "/example-algo-1-trainer", "Config": {"Image": "awsdeepracercommunity/deepracer-simapp:6.0.6-gpu", "Env": []},
                     "Mounts": [{"Type": "bind", "Source": "/tmp/sagemaker/job/input", "Destination": "/opt/ml/input"}]}
        self.assertEqual(trainer_prefix(FakeClient(), container), "model-v3")

    def test_unrelated_container_is_not_read(self):
        container = {"Name": "/unrelated", "Config": {"Image": "other:latest", "Env": []}}
        self.assertIsNone(trainer_prefix(None, container))

    def test_metrics_saved_on_change_only(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            root = Path(tmp)
            store = Store(root, "test-model", Redactor())
            document = {"model": "test-model", "etag": "etag1", "rows": [{"phase": "training", "episode": 1}]}
            store.metric(document)
            store.metric({**document, "fetched": "later"})
            store.save()
            self.assertEqual(len(list((root / "metrics/snapshots").glob("*.json"))), 1)
            self.assertEqual(json.loads((root / "metrics/latest.json").read_text())["etag"], "etag1")
            self.assertEqual(json.loads((root / "status.json").read_text())["training_episodes"], 1)
            self.assertEqual(Store(root, "test-model", Redactor()).state["metrics_snapshots"], 1)

    def test_wrong_model_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            store = Store(Path(tmp), "right", Redactor())
            with self.assertRaises(ValueError):
                store.metric({"model": "wrong", "rows": []})
            self.assertFalse((Path(tmp) / "metrics/latest.json").exists())


if __name__ == "__main__":
    unittest.main()
