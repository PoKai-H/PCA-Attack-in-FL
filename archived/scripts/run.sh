#!/usr/bin/env bash
set -e

configs=(
  trimmed_mean_noniid
  trimmed_mean_noniid_ones
  trimmed_mean_noniid_min_max
  trimmed_mean_noniid_min_sum
  trimmed_mean_noniid_min_max_nnm
  trimmed_mean_noniid_min_sum_nnm
  prodigy_noniid
  prodigy_noniid_ones
  prodigy_noniid_min_max
  prodigy_noniid_min_sum
  prodigy_noniid_min_max_nnm
  prodigy_noniid_min_sum_nnm
)

for config in "${configs[@]}"; do
  echo "==== $config ===="
  python main.py --config "$config"
done