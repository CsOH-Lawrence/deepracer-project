"""Read-only DeepRacer S3 metrics dashboard, restricted to loopback."""
import argparse
import gzip
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

# The page refreshes every 10s; a slightly shorter TTL keeps repeated reads
# from the same browser on the cached copy.
CACHE_SECONDS = 8.0
# Upper bound on rows serialized per request. A long run keeps appending to
# TrainingMetrics.json, and the browser never needs the whole history.
MAX_ROWS = 20000
GZIP_MIN_BYTES = 2048
METRICS_KEY = "/metrics/TrainingMetrics.json"


def normalize(document):
    if not isinstance(document, dict):
        return []
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
        self.max_rows = args.max_rows
        self.cache_seconds = args.cache_seconds
        # Cache reads and refreshes use separate locks: a slow S3 call must
        # never hold the lock that HTTP handlers wait on.
        self._lock = threading.Lock()
        self._refreshing = threading.Lock()
        self._cached = None
        self._checked = 0.0

    @property
    def source(self):
        return self.prefix + METRICS_KEY

    def get(self):
        with self._lock:
            cached, checked = self._cached, self._checked
        if cached is not None and time.monotonic() - checked < self.cache_seconds:
            return cached
        # Single flight: one request refreshes, the rest reuse what is cached.
        # N browsers must not stack up N slow S3 calls behind a timeout.
        if not self._refreshing.acquire(blocking=False):
            if cached is not None:
                return cached
            self._refreshing.acquire()
        try:
            with self._lock:
                if self._cached is not None and time.monotonic() - self._checked < self.cache_seconds:
                    return self._cached
                previous = self._cached
            result = self._fetch(previous)
            with self._lock:
                self._cached, self._checked = result, time.monotonic()
            return result
        finally:
            self._refreshing.release()

    def _waiting(self):
        # A new model must remain absent in S3 until the user starts it.
        return {
            "model": self.prefix, "rows": [], "status": "waiting",
            "source": self.source, "modified": None,
            "fetched": datetime.now(timezone.utc).isoformat(),
            "etag": "", "total_rows": 0, "truncated": False,
        }

    def _fetch(self, previous):
        request = {"Bucket": self.bucket, "Key": self.source}
        if previous and previous.get("etag"):
            request["IfNoneMatch"] = previous["etag"]
        try:
            response = self.client.get_object(**request)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("304", "NotModified"):
                # Object unchanged: keep the parsed rows, refresh only the check time.
                return dict(previous, fetched=datetime.now(timezone.utc).isoformat())
            if code != "NoSuchKey":
                raise
            return self._waiting()
        body = response["Body"]
        try:
            document = json.loads(body.read())
        finally:
            body.close()
        rows = normalize(document)
        total = len(rows)
        truncated = total > self.max_rows
        if truncated:
            rows = rows[-self.max_rows:]
        return {
            "model": self.prefix, "rows": rows,
            "source": self.source,
            "modified": response["LastModified"].isoformat(),
            "fetched": datetime.now(timezone.utc).isoformat(),
            "etag": response.get("ETag", ""),
            "total_rows": total, "truncated": truncated,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--profile", default="minio")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000")
    parser.add_argument("--bucket", default="bucket")
    parser.add_argument("--model-prefix", default="rl-deepracer-sagemaker")
    parser.add_argument("--cache-seconds", type=float, default=CACHE_SECONDS)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = parser.parse_args()
    metrics = Metrics(args)
    page = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

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
            body, encoding = payload, None
            if len(payload) >= GZIP_MIN_BYTES and "gzip" in self.headers.get("Accept-Encoding", "").lower():
                body, encoding = gzip.compress(payload, 6), "gzip"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Read-only dashboard on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
