#!/bin/bash

for iterations in $(seq -w 2000 2000 10000); do
for mult in $(seq -w 0.0 0.2 2.0); do
dir="scenario_${mult}_${iterations}"
echo "$dir"
mkdir -p "$dir"
python3 singan.py -o "${dir}" --iterations $iterations --res_loss_mult $mult
done
done
