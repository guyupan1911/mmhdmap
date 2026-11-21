#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
prepare_if_needed
run_experiment inference python /opt/vma-support/validate.py infer "$@"
