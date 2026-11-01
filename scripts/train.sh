#!/usr/bin/env bash
set -euo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export BALTO_RELEASE_ROOT="$release_root"
export PYTHONPATH="$release_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/train.sh <confiqa|ragtruth|financial> <model> <train.parquet> <test.parquet> [Hydra overrides...]" >&2
  exit 2
fi
balto_task=$1
export BALTO_MODEL_PATH=$2 BALTO_TRAIN_DATA=$3 BALTO_VAL_DATA=$4
shift 4
case "$balto_task" in
  confiqa|ragtruth|financial) ;;
  *) echo "Unknown task: $balto_task" >&2; exit 2 ;;
esac
: "${BALTO_JUDGE_BASE_URL:?Set BALTO_JUDGE_BASE_URL to an OpenAI-compatible /v1 base URL}"
: "${BALTO_JUDGE_MODEL:?Set BALTO_JUDGE_MODEL to the served Judge model name}"
balto_backend=${BALTO_BACKEND:-megatron}
case "$balto_backend" in
  megatron|fsdp) ;;
  *) echo "BALTO_BACKEND must be megatron or fsdp" >&2; exit 2 ;;
esac
cd "$release_root"
exec "${BALTO_PYTHON:-python}" -m verl.trainer.main_ppo \
  --config-path="$release_root/configs" --config-name="balto_$balto_backend" \
  "custom_reward_function.reward_kwargs.task=$balto_task" \
  "trainer.experiment_name=balto_$balto_task" "$@"
