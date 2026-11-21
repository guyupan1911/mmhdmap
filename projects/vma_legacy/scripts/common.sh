#!/usr/bin/env bash
set -euo pipefail
legacy_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)"
export VMA_UID="${VMA_UID:-$(id -u)}" VMA_GID="${VMA_GID:-$(id -g)}"
compose() {
    docker compose --project-directory "$legacy_root" -f "$legacy_root/docker-compose.yaml" "$@"
}
work_dirs="$(compose config --format json | python -c 'import json,sys; c=json.load(sys.stdin); print(next(v["source"] for v in c["services"]["vma_legacy"]["volumes"] if v["target"]=="/work_dirs"))')"
mkdir -p "$work_dirs"
prepare_if_needed() {
    if [[ ! -f "$work_dirs/crops/sd_data_line_dict.json" ]]; then
        bash "$legacy_root/prepare.sh"
    fi
}
run_experiment() {
    local kind="$1"
    shift
    local output_dir
    output_dir="$(mktemp -d "$work_dirs/${kind}_$(date +%Y%m%d_%H%M%S)_XXXX")"
    printf '%q ' "$@" > "$output_dir/command.txt"
    printf '\n' >> "$output_dir/command.txt"
    echo "Results: $output_dir"
    compose run --rm -T vma_legacy "$@" --output "/work_dirs/$(basename "$output_dir")" > "$output_dir/run.log" 2>&1 &
    local run_pid=$!
    # Show concise status while keeping the complete initialization log on disk.
    tail --pid="$run_pid" -n +1 -F "$output_dir/run.log" 2>/dev/null | sed -u -n '/^CHECKPOINT_AUDIT /p; /^INFERENCE /p; /^TRAIN /p; /^EVAL /p' &
    local tail_pid=$!
    local status=0
    wait "$run_pid" || status=$?
    wait "$tail_pid" || true
    if (( status != 0 )); then
        tail -n 60 "$output_dir/run.log" >&2
    fi
    echo "Results: $output_dir"
    return "$status"
}
