#!/bin/bash

imstack="python3 $(dirname $0)/imstack.py"

for x in $(seq 0 100); do
    if [[ ! -f "fake${x}-0.png" ]]; then
        break
    fi
    $imstack -o "fake${x}-stacked.png" fake${x}-*.png
done

$imstack -H -o "fake-stacked.png" fake*-stacked.png
