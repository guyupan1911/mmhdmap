#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
prepare_if_needed
run_experiment overfit python /opt/vma-support/validate.py overfit \
    --index 26 --steps 300 --lr 0.00002 --disable-dropout --cosine-lr "$@"
