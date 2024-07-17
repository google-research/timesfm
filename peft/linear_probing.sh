#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 finetune.py \
    --model-name="google/timesfm-1.0-200m" \
    --backend="gpu" \
    --horizon-len=128 \
    --context-len=512 \
    --freq="15min" \
    --data-path="../datasets/ETT-small/ETTm1.csv" \
    --num-epochs=100 \
    --learning-rate=1e-2 \
    --adam-epsilon=1e-7 \
    --adam-clip-threshold=1e2 \
    --early-stop-patience=10 \
    --datetime-col="date" \
    --boundaries=1000 46080 57600 \
    --use-linear-probing \
    --cos-initial-decay-value=1e-4 \
    --cos-decay-steps=40000 \
    --cos-final-decay-value=1e-5 \
    --ema-decay=0.9999