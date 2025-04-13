#!/bin/bash

imstack="python3 $(dirname $0)/imstack.py"

stacked_paths=()
for scenario_dir in "$@"; do
    stacked="${scenario_dir}/stacked.png"
    $imstack -o "$stacked" ${scenario_dir}/reconstruction0.png ${scenario_dir}/fake0-*.png
    stacked_paths+=("$stacked")
done

echo "${stacked_paths[@]}"
$imstack -H -o "experiment.png" "${stacked_paths[@]}"


