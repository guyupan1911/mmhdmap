#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
# Match the repo selected in Compose, including .env and shell overrides.
vma_source="$(compose config --format json | python -c 'import json,sys; c=json.load(sys.stdin); print(next(v["source"] for v in c["services"]["vma_legacy"]["volumes"] if v["target"]=="/workspace/VMA"))')"
mkdir -p "$work_dirs/build"
git -C "$vma_source" rev-parse HEAD > "$work_dirs/build/upstream_commit.txt"
git -C "$vma_source" status --short > "$work_dirs/build/upstream_status.txt"
compose build --progress plain vma_legacy 2>&1 | tee "$work_dirs/build/build.log"
docker image inspect vma_legacy:blackwell-cu128 --format '{{.Id}}' > "$work_dirs/build/image_id.txt"
