#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ERROR_FILE="${ERROR_FILE:-generated_attack_errors.txt}"

: > "$ERROR_FILE"

configs="$(
  "$PYTHON_BIN" - <<'PY'
from configs.cifar10.other_attacks import CONFIGS as OTHER_ATTACKS_CONFIGS
from configs.cifar10.pca_attacks import CONFIGS as PCA_ATTACKS_CONFIGS

for config_name in [*OTHER_ATTACKS_CONFIGS.keys(), *PCA_ATTACKS_CONFIGS.keys()]:
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
