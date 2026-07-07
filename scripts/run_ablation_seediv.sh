#!/usr/bin/env bash
set -e
for c in configs/ablation/seediv_*.yaml; do python train.py --config "$c"; done
