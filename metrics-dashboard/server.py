"""Read-only DeepRacer S3 metrics dashboard, restricted to loopback."""
import argparse
import json
import math
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def normalize(document):
    rows = []
    for item in document.get("metrics", []):
        try:
            row = {
                "episode": int(item["episode"]),
                "phase": str(item.get("phase", "training")),
                "reward": float(item["reward_score"]),
                "progress": float(item["completion_percentage"]),
                "seconds": float(item.get("elapsed_time_in_milliseconds", 0)) / 1000,
                "status": str(item.get("episode_status", "")),
                "sim_time_ms": float(item.get("metric_time", 0)),
            }
            if row["episode"] < 0 or not all(math.isfinite(row[k]) for k in ("reward", "progress", "seconds", "sim_time_ms")):
                continue
            if row["phase"] in ("training", "evaluation"):
                rows.append(row)
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    # Simulator time may rewind (reset or a second /clock publisher).
    # Episodes define this model's progression; never reorder history by /clock.
    # Python's stable sort retains trial order within each evaluation group.
    return sorted(rows, key=lambda row: (row["episode"], row["phase"] == "evaluation"))


class Metrics:
    def __init__(self, args):
        self.client = boto3.Session(profile_name=args.profile).client(
            "s3", endpoint_url=args.endpoint,
            config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}),
        )
        self.bucket, self.prefix = args.bucket, args.model_prefix.rstrip("/")
        self.lock = threading.Lock()
        self.cached = None
        self.checked = 0

    def get(self):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.checked < 5:
                return self.cached
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=self.prefix + "/metrics/TrainingMetrics.json")
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") != "NoSuchKey":
                    raise
                # A new model must remain absent in S3 until the user starts it.
                self.cached = {
                    "model": self.prefix, "rows": [], "status": "waiting",
                    "source": self.prefix + "/metrics/TrainingMetrics.json",
                    "modified": None, "fetched": datetime.now(timezone.utc).isoformat(),
                    "etag": "",
                }
                self.checked = time.monotonic()
                return self.cached
            body = response["Body"]
            try:
                document = json.loads(body.read())
            finally:
                body.close()
            rows = normalize(document)
            self.cached = {
                "model": self.prefix, "rows": rows,
                "source": self.prefix + "/metrics/TrainingMetrics.json",
                "modified": response["LastModified"].isoformat(),
                "fetched": datetime.now(timezone.utc).isoformat(),
                "etag": response.get("ETag", ""),
            }
            self.checked = time.monotonic()
            return self.cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--profile", default="minio")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000")
    parser.add_argument("--bucket", default="bucket")
    parser.add_argument("--model-prefix", default="rl-deepracer-sagemaker")
    args = parser.parse_args()
    metrics = Metrics(args)
    page = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route == "/":
                status, content_type, payload = 200, "text/html; charset=utf-8", page
            elif route == "/api/metrics":
                try:
                    payload = json.dumps(metrics.get(), ensure_ascii=False, allow_nan=False).encode()
                    status, content_type = 200, "application/json; charset=utf-8"
                except Exception as exc:
                    # Do not expose credentials, tracebacks or service details to browsers.
                    print("Metrics read failed:", type(exc).__name__, flush=True)
                    payload = json.dumps({"error": "暂时无法读取训练统计，页面将自动重试。"}, ensure_ascii=False).encode()
                    status, content_type = 503, "application/json; charset=utf-8"
            elif route == "/health":
                status, content_type, payload = 200, "text/plain", b"OK"
            else:
                status, content_type, payload = 404, "text/plain", b"Not found"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Read-only dashboard on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
