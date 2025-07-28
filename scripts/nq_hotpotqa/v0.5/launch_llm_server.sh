#!/bin/bash
# PPO
export VLLM_DISABLE_MEMORY_PROFILING=1
while true; do
    echo "Starting vLLM API server..."
    python -m vllm.entrypoints.openai.api_server \
        --model /home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/qwen2.5-7b-instruct-1m \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --gpu-memory-utilization 0.2 \
        --cpu-offload-gb 80 \
        --max-num-seqs 128 \
        --max-model-len 2048 \
        --block-size 16 \
        --enable-prefix-caching \
        --distributed-executor-backend mp \
        --port 8001
    
    code=$?
    echo "Server exited with code ${code}, cleaning up…"
    python -c "import torch; torch.cuda.empty_cache()"
    pkill -f "vllm.entrypoints.openai.api_server" || true
    find /dev/shm -name '*nvidia_ipc*' -delete 2>/dev/null || true
    rm -f /tmp/nvidia-ipc-* 2>/dev/null || true
    echo "Restarting in 2 seconds..."
    sleep 2
done


#!/bin/bash
# GRPO
"""
export VLLM_DISABLE_MEMORY_PROFILING=1
while true; do
    echo "Starting vLLM API server..."
    python -m vllm.entrypoints.openai.api_server \
        --model /home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/qwen2.5-7b-instruct-1m \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --gpu-memory-utilization 0.2 \
        --cpu-offload-gb 100 \
        --max-num-seqs 96 \
        --max-model-len 2048 \
        --block-size 16 \
        --enable-prefix-caching \
        --distributed-executor-backend mp \
        --port 8001
    
    code=$?
    echo "Server exited with code ${code}, cleaning up…"
    python -c "import torch; torch.cuda.empty_cache()"
    pkill -f "vllm.entrypoints.openai.api_server" || true
    find /dev/shm -name '*nvidia_ipc*' -delete 2>/dev/null || true
    rm -f /tmp/nvidia-ipc-* 2>/dev/null || true
    echo "Restarting in 2 seconds..."
    sleep 2
done
"""
