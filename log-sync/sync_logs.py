"""Read-only SSH log receiver. Secrets stay in memory; no server files are edited."""
import argparse
import collections
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import socket
import threading
import time

CST = timezone(timedelta(hours=8))
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z) ")
SENSITIVE = re.compile(r"(?:password|passwd|secret|token|access.?key|private.?key)", re.I)
KEYED_VALUE = re.compile(
    r"(?i)((?:[A-Z0-9_]*(?:PASSWORD|SECRET_ACCESS_KEY|ACCESS_KEY_ID|ACCESS_TOKEN|AUTH_TOKEN|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)[\"']?\s*[:=]\s*[\"']?)([^\s,\"'}]+)"
)


def now():
    return datetime.now(CST).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def overlap_since(stamp):
    if not stamp:
        return None
    # Overlap intentionally provides at-least-once delivery after reconnect.
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (parsed - timedelta(seconds=2)).isoformat(timespec="seconds")


class Redactor:
    def __init__(self, secrets=()):
        self.lock = threading.Lock()
        self.secrets = set(value for value in secrets if isinstance(value, str) and len(value) >= 4)

    def add_env(self, environment):
        with self.lock:
            for entry in environment:
                key, separator, value = entry.partition("=")
                if separator and SENSITIVE.search(key) and len(value) >= 4:
                    self.secrets.add(value)

    def text(self, value):
        with self.lock:
            secrets = sorted(self.secrets, key=len, reverse=True)
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return KEYED_VALUE.sub(lambda match: match.group(1) + "[REDACTED]", value)


class Store:
    def __init__(self, root, model, redactor):
        self.root = root
        self.model = model
        self.redactor = redactor
        root.mkdir(parents=True, exist_ok=True)
        (root / "containers").mkdir(exist_ok=True)
        (root / "metrics" / "snapshots").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        state_path = root / "state.json"
        self.state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "model": model, "containers": {}, "metrics_snapshots": 0, "metrics_etag": None
        }
        if self.state["model"] != model:
            raise ValueError("The output directory belongs to a different model")
        self.status = {"pid": os.getpid(), "model": model, "started": now(), "connection": "starting", "metrics_status": "waiting", "directory": str(root)}

    def event(self, kind, **fields):
        record = {"time": now(), "event": kind, **fields}
        # No exception objects, remote environment, or credentials are serialized.
        with self.lock, (self.root / "sync-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(self.redactor.text(json.dumps(record, ensure_ascii=False)) + "\n")

    def save(self):
        with self.lock:
            self.status["updated"] = now()
            self.status["containers"] = self.state["containers"]
            self.status["metrics_snapshots"] = self.state["metrics_snapshots"]
            atomic_json(self.root / "state.json", self.state)
            atomic_json(self.root / "status.json", self.status)

    def metric(self, document):
        if document.get("model") != self.model:
            raise ValueError("Metrics endpoint points to another model; refusing to mix histories")
        if "error" in document:
            raise ValueError("Metrics endpoint returned an error")
        with self.lock:
            self.status["last_metrics_check"] = now()
            self.status["metrics_status"] = document.get("status", "active")
            rows = document.get("rows", [])
            self.status["training_episodes"] = sum(row.get("phase") == "training" for row in rows)
            self.status["evaluation_trials"] = sum(row.get("phase") == "evaluation" for row in rows)
            self.status["remote_metrics_modified"] = document.get("modified")
            digest = document.get("etag") or hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            if digest == self.state["metrics_etag"]:
                return
            # The API exposes read-only normalized metrics, not the raw S3 object.
            safe_document = json.loads(self.redactor.text(json.dumps(document, ensure_ascii=False)))
            name = datetime.now(CST).strftime("%Y%m%d-%H%M%S-%f") + ".json"
            atomic_json(self.root / "metrics" / "snapshots" / name, safe_document)
            atomic_json(self.root / "metrics" / "latest.json", safe_document)
            self.state["metrics_etag"] = digest
            self.state["metrics_snapshots"] += 1
            self.status["last_metrics_saved"] = now()


def model_prefix(environment):
    env = dict(entry.split("=", 1) for entry in environment if "=" in entry)
    prefix = env.get("SM_HP_S3_PREFIX") or env.get("S3_PREFIX")
    if prefix:
        return prefix.strip("\"")
    try:
        return json.loads(env.get("SM_HPS", "{}" )).get("s3_prefix")
    except (ValueError, TypeError):
        return None


def trainer_prefix(client, container, fallback=None):
    prefix = model_prefix(container["Config"].get("Env", []))
    if prefix:
        return prefix
    name = container["Name"].lstrip("/")
    if "-algo-1-" not in name or "deepracer-simapp" not in container["Config"].get("Image", ""):
        return None
    # SageMaker creates SM_HP_* only inside its entrypoint, not Docker Config.Env.
    # Its bind-mounted hyperparameters are available without entering the trainer.
    for mount in container.get("Mounts", []):
        if mount.get("Type") != "bind":
            continue
        destination = mount.get("Destination", "").rstrip("/")
        if destination == "/opt/ml/input":
            path = posixpath.join(mount["Source"], "config", "hyperparameters.json")
        elif destination == "/opt/ml/input/config":
            path = posixpath.join(mount["Source"], "hyperparameters.json")
        else:
            continue
        sftp = client.open_sftp()
        try:
            with sftp.open(path, "rb") as handle:
                values = json.load(handle)
            value = values.get("s3_prefix")
            if isinstance(value, str) and value:
                return value.strip('"')
        except (OSError, ValueError, KeyError):
            pass
        finally:
            sftp.close()
    return (fallback or {}).get(name)


def command_output(client, command):
    stdin, stdout, stderr = client.exec_command(command, timeout=20)
    stdin.close()
    output = stdout.read()
    error = stderr.read()
    result = stdout.channel.recv_exit_status()
    if result:
        # Remote stderr can contain secrets, so only return the exit status.
        raise RuntimeError(f"Read-only SSH command exited with status {result}")
    return output


def discover(client, config, redactor):
    ids = command_output(client, "docker ps -aq").decode().split()
    if not ids:
        return []
    if any(not re.fullmatch(r"[a-f0-9]{12,64}", cid) for cid in ids):
        raise ValueError("Invalid container identifier")
    objects = json.loads(command_output(client, "docker inspect " + shlex.join(ids)))
    models = {config["model"], *config.get("source_models", [])}
    selected = []
    for obj in objects:
        name = obj["Name"].lstrip("/")
        environment = obj["Config"].get("Env", [])
        prefix = trainer_prefix(client, obj, config.get("known_trainers"))
        if name not in config["outer_containers"] and prefix not in models:
            continue
        redactor.add_env(environment)
        selected.append({"id": obj["Id"], "name": name, "running": obj["State"]["Running"], "model": prefix})
    return selected


def remote_metrics(client, port):
    transport = client.get_transport()
    channel = transport.open_channel("direct-tcpip", ("127.0.0.1", port), ("127.0.0.1", 0), timeout=10)
    channel.settimeout(10)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.sock = channel
    try:
        connection.request("GET", "/api/metrics", headers={"Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Metrics HTTP status {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


class LogFollower(threading.Thread):
    def __init__(self, client, container, store, rotate_bytes):
        super().__init__(daemon=True, name="docker-log-" + container["id"][:12])
        self.client, self.container, self.store = client, container, store
        self.rotate_bytes = rotate_bytes
        self.channel = None
        self.error_type = None

    def run(self):
        container, store = self.container, self.store
        cid = container["id"]
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", container["name"])[:100] + "-" + cid[:12]
        handle = None
        try:
            with store.lock:
                state = store.state["containers"].setdefault(cid, {"name": container["name"], "model": container["model"], "lines": 0, "last_timestamp": None, "part": 1})
                state["backfill_complete"] = False
                since = overlap_since(state["last_timestamp"])
                part = state["part"]
            command = "docker logs --follow --timestamps " + ("--since " + shlex.quote(since) + " " if since else "") + cid
            channel = self.client.get_transport().open_session(timeout=10)
            self.channel = channel
            channel.set_combine_stderr(True)
            channel.exec_command(command)
            path = store.root / "containers" / f"{base}.part{part:03d}.log"
            handle = path.open("a", encoding="utf-8", newline="\n")
            size = path.stat().st_size
            handle.write(f"# local sync opened {now()}, since={since or 'all retained logs'}; reconnect overlap may repeat lines\n")
            handle.flush()
            store.event("log_stream_open", container=container["name"], container_id=cid[:12], since=since)
            for raw in channel.makefile("rb"):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                safe_line = store.redactor.text(line)
                encoded_size = len(safe_line.encode("utf-8")) + 1
                if size >= self.rotate_bytes:
                    handle.close()
                    part += 1
                    path = store.root / "containers" / f"{base}.part{part:03d}.log"
                    handle = path.open("a", encoding="utf-8", newline="\n")
                    size = path.stat().st_size
                handle.write(safe_line + "\n")
                handle.flush()
                size += encoded_size
                match = STAMP.match(line)
                with store.lock:
                    state["part"] = part
                    state["lines"] += 1
                    state["last_received"] = now()
                    if match:
                        value = match.group(1)
                        if not state["last_timestamp"] or value > state["last_timestamp"]:
                            state["last_timestamp"] = value
            code = channel.recv_exit_status()
            with store.lock:
                state["backfill_complete"] = code == 0
                state["last_stream_exit_code"] = code
            store.event("log_stream_closed", container=container["name"], exit_code=code)
            if code != 0:
                self.error_type = "RemoteLogCommandFailed"
        except Exception as error:
            self.error_type = type(error).__name__
            store.event("log_stream_interrupted", container=container["name"], error_type=self.error_type)
        finally:
            if handle:
                handle.close()
            if self.channel:
                self.channel.close()

    def close(self):
        if self.channel:
            self.channel.close()


class InstanceLock:
    def __init__(self, path):
        self.handle = path.open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        self.handle.close()


def run(config_path, force_reconnect_after=0):
    import paramiko
    logging.getLogger("paramiko").addHandler(logging.NullHandler())
    logging.getLogger("paramiko").propagate = False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / config["log_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    known_hosts = (config_path.parent / config["known_hosts"]).resolve()
    instance = InstanceLock(root / ".sync.lock")
    password = getpass.getpass("SSH login password (memory only): ")
    redactor = Redactor([password])
    store = Store(root, config["model"], redactor)
    stop_file = root / "STOP_SYNC"
    if stop_file.exists():
        raise RuntimeError("STOP_SYNC exists; remove it manually before restarting")
    client, followers = None, {}
    retry = 5
    reconnect_test_pending = bool(force_reconnect_after)
    store.event("sync_started", pid=os.getpid(), model=config["model"])
    print("LOCAL LOG SYNC: " + str(root), flush=True)
    try:
        while not stop_file.exists():
            if shutil.disk_usage(root).free < config["minimum_free_bytes"]:
                store.status["connection"] = "stopped_low_disk_space"
                store.event("low_disk_space")
                break
            try:
                if client is None or not client.get_transport() or not client.get_transport().is_active():
                    if client:
                        client.close()
                    for follower in followers.values():
                        follower.close()
                        follower.join(timeout=2)
                    followers.clear()
                    client = paramiko.SSHClient()
                    client.load_host_keys(str(known_hosts))
                    client.set_missing_host_key_policy(paramiko.RejectPolicy())
                    client.connect(config["host"], port=config.get("port", 22), username=config["username"], password=password,
                                   look_for_keys=False, allow_agent=False, timeout=10, banner_timeout=15, auth_timeout=15)
                    client.get_transport().set_keepalive(20)
                    store.status["connection"] = "connected"
                    store.status["last_connected"] = now()
                    store.event("ssh_connected", host_key_verified=True)
                    print("SSH connected; pinned host key verified", flush=True)
                    retry, next_discovery, next_metrics = 5, 0, 0
                    connected_at = time.monotonic()
                if time.monotonic() >= next_discovery:
                    for container in discover(client, config, redactor):
                        cid = container["id"]
                        existing = followers.get(cid)
                        if existing and existing.is_alive():
                            continue
                        if existing:
                            existing.join(timeout=1)
                        state = store.state["containers"].get(cid, {})
                        if state.get("backfill_complete") and not container["running"]:
                            continue
                        follower = LogFollower(client, container, store, config["rotate_bytes"])
                        followers[cid] = follower
                        follower.start()
                    next_discovery = time.monotonic() + config["discovery_interval_seconds"]
                if time.monotonic() >= next_metrics:
                    try:
                        store.metric(remote_metrics(client, int(config["metrics_remote_port"])))
                    except Exception as error:
                        store.status["metrics_status"] = "retrying"
                        store.event("metrics_read_failed", error_type=type(error).__name__)
                    next_metrics = time.monotonic() + config["metrics_interval_seconds"]
                store.save()
                if reconnect_test_pending and time.monotonic() - connected_at >= force_reconnect_after:
                    reconnect_test_pending = False
                    store.event("controlled_reconnect_test")
                    client.close()  # Only the local read-only SSH connection is closed.
                time.sleep(2)
            except (paramiko.AuthenticationException, paramiko.BadHostKeyException) as error:
                store.status["connection"] = "stopped_authentication_or_host_key_error"
                store.event("authentication_or_host_key_error", error_type=type(error).__name__)
                print("Stopped: authentication or pinned host key could not be verified", flush=True)
                break
            except Exception as error:
                store.status["connection"] = "reconnecting"
                store.event("ssh_reconnecting", error_type=type(error).__name__, retry_seconds=retry)
                store.save()
                if client:
                    client.close()
                print(f"Connection unavailable ({type(error).__name__}); retrying in {retry}s", flush=True)
                for _ in range(retry):
                    if stop_file.exists():
                        break
                    time.sleep(1)
                retry = min(retry * 2, 60)
    except KeyboardInterrupt:
        store.event("local_stop_requested")
    finally:
        for follower in followers.values():
            follower.close()
        if client:
            client.close()
        for follower in followers.values():
            follower.join(timeout=2)
        if store.status["connection"] not in {"stopped_low_disk_space", "stopped_authentication_or_host_key_error"}:
            store.status["connection"] = "stopped"
        store.save()
        instance.close()
        password = None
        print("Local log sync stopped; remote training was not modified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--test-reconnect-after", type=int, default=0,
                        help="Close this logger's SSH connection once to test catch-up, without touching training")
    args = parser.parse_args()
    run(args.config.resolve(), args.test_reconnect_after)
