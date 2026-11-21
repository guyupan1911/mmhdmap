#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
exec docker compose --project-directory "$legacy_root" -f "$legacy_root/docker-compose.yaml" run --rm vma_legacy "$@"
