#!/usr/bin/env bash
set -e
for c in configs/ablation/seed_*.yaml; do python train.py --config "$c"; done
