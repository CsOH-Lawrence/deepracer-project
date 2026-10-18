#!/usr/bin/env bash
# Ubuntu entry point. Run with bash; no chmod or curl-pipe-shell needed.
set -e
if ! command -v python3 >/dev/null 2>&1; then
    echo '需要 Python 3；请先选择 Ubuntu 22.04 预装 Docker 的虚拟机。' >&2
    exit 1
fi
DR_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DR_SETUP_ROOT/tools/deploy.py" "$@"
