#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-adaptiveFL}"
ERROR_FILE="${ERROR_FILE:-no_attack_errors.txt}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("$PYTHON_BIN")
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  PY_CMD=(conda run -n "$CONDA_ENV" python)
else
  PY_CMD=(python)
fi

: > "$ERROR_FILE"

CONFIG_LIST_CODE=$'from configs.cifar10.no_attack import CONFIGS as NO_ATTACK_CONFIGS\nfor config_name in sorted(NO_ATTACK_CONFIGS):\n    print(config_name)'

configs=()
while IFS= read -r config_name; do
  configs+=("$config_name")
done < <(
  "${PY_CMD[@]}" -c "$CONFIG_LIST_CODE"
)

echo "Python command: ${PY_CMD[*]}"
echo "Configs: ${#configs[@]}"
echo "Errors: $ERROR_FILE"

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No no-attack configs found." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' "${configs[@]}"
  exit 0
fi

for config in "${configs[@]}"; do
  if [[ -z "$config" ]]; then
    continue
  fi

  echo "==== $config ===="
  if "${PY_CMD[@]}" main.py --config "$config"; then
    echo "==== $config done ===="
  else
    echo "$config" >> "$ERROR_FILE"
    echo "==== $config failed; recorded in $ERROR_FILE ===="
  fi
done
