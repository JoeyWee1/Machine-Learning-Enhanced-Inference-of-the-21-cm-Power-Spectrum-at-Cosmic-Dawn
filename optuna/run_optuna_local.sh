#!/bin/bash

python optuna_search.py \
    --data-dir simulations \
    --output-dir optuna_outputs \
    --n-trials 25 \
    --epochs 1000 \
    --patience 100 \
    --device cpu \
    --log-power