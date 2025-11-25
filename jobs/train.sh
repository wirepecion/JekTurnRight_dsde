#!/usr/bin/env bash
# /home/sirav/JekTurnRight_dsde/jobs/train.sh
# Simple orchestration: train -> tune_threshold -> (optional) push & deploy
# Usage: ./jobs/train.sh [--push | --no-push] [--yes] [--dry-run]

set -euo pipefail

PUSH=""
AUTO_NO_PROMPT=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--push | --no-push] [--yes] [--dry-run]
    --push       Push to hub and deploy after training/tuning
    --no-push    Skip push/deploy (default if neither provided and not confirmed)
    --yes        Do not prompt; accept chosen push/no-push
    --dry-run    Print commands without executing
EOF
    exit 1
}

# parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --push) PUSH=1; shift ;;
        --no-push) PUSH=0; shift ;;
        --yes) AUTO_NO_PROMPT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

run_cmd() {
    echo "+ $*"
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    eval "$@"
}

confirm_push() {
    if [ -n "$PUSH" ]; then
        return $((1 - PUSH)) # return 0 if PUSH=1, return 1 if PUSH=0
    fi
    if [ "$AUTO_NO_PROMPT" -eq 1 ]; then
        # default to no push when auto-confirm and not specified
        PUSH=0
        return 1
    fi
    read -r -p "Push to hub and deploy after training? [y/N]: " resp
    case "$resp" in
        [yY]|[yY][eE][sS]) PUSH=1; return 0 ;;
        *) PUSH=0; return 1 ;;
    esac
}

echo "==> Starting training pipeline: $(date)"

# 1) Train
run_cmd python3 -m src.ds.train

# 2) Tune threshold (try common spelling variations)
if run_cmd python3 -m src.ds tune_threshold; then
    :
elif run_cmd python3 -m src.ds tune_thresould; then
    :
else
    echo "ERROR: threshold tuning command failed (tried tune_threshold and tune_thresould)"
    exit 2
fi

# 3) Optional push & deploy
if confirm_push; then
    echo "==> Pushing to hub..."
    # run_cmd python -m src.ds push_hub

    echo "==> Deploying..."
    run_cmd python3 -m src.ds deploy
else
    echo "==> Skipping push & deploy."
fi

echo "==> Pipeline complete: $(date)"