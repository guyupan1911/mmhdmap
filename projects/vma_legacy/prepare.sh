#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
mkdir -p "$work_dirs"
if [[ -f "$work_dirs/crops/sd_data_line_dict.json" ]]; then
    echo 'Prepared crops already exist: work_dirs/crops'
    exit 0
fi
compose run --rm -T vma_legacy python tools/custom/generate_line_final_from_origin.py \
    /data/sd_data/line/origin_data/line_data_6k.json /data/sd_data/line/origin_data \
    /work_dirs/crops --visualize 2>&1 | tee "$work_dirs/prepare.log"
