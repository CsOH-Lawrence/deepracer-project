#!/usr/bin/env bash
# Administrator-only, dedicated Ubuntu VM. Does not start training.
set -e
DR_SCHOOL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DR_SCHOOL_ROOT/deploy.sh" --arch cpu --school "$@"
