#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/deploy_generation.sh <model> [SGLang options...]" >&2
  exit 2
fi
generation_model=$1
shift
exec "${BALTO_PYTHON:-python}" -m sglang.launch_server \
  --model-path "$generation_model" --host "${BALTO_SERVER_HOST:-127.0.0.1}" \
  --port "${BALTO_SERVER_PORT:-8000}" --tp "${BALTO_SERVER_TP:-1}" "$@"
