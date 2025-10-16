#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /data/home/Yifan/Hallu/Search-R1-phase1-main/models/Qwen/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8002 \
  --api-key "yifan" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 2048 \
  --trust-remote-code
