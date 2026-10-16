#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ERROR_FILE="${ERROR_FILE:-error.txt}"

: > "$ERROR_FILE"

configs="$(
  "$PYTHON_BIN" - <<'PY'
from configs.cifar10.classic_attack import CONFIGS as CLASSIC_ATTACK_CONFIGS
from configs.cifar10.no_attack import CONFIGS as NO_ATTACK_CONFIGS

for config_name in [*CLASSIC_ATTACK_CONFIGS.keys(), *NO_ATTACK_CONFIGS.keys()]:
    print(config_name)
PY
)"

while IFS= read -r config; do
  if [[ -z "$config" ]]; then
    continue
  fi

  echo "==== $config ===="
  if "$PYTHON_BIN" main.py --config "$config"; then
    echo "==== $config done ===="
  else
    echo "$config" >> "$ERROR_FILE"
    echo "==== $config failed; recorded in $ERROR_FILE ===="
  fi
done <<< "$configs"
