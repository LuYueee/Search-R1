# Grounded-R1 Training Guide

This document provides the environment setup and training instructions for Grounded-R1. The environment follows the Search-R1 setup, while the training entry scripts are specific to Grounded-R1.

Grounded-R1 uses two runtime environments:

- `grounded-r1`: for reinforcement learning (RL) training.
- `retriever`: for the local retrieval service.

The retrieval service and the training process should be launched in two separate terminal sessions.

---

## 1. Environment Setup

### 1.1 Create the Grounded-R1 Training Environment

```bash
conda create -n grounded-r1 python=3.9
conda activate grounded-r1
```

Install PyTorch:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Install vLLM:

```bash
pip3 install vllm==0.6.3
```

Install the local package:

```bash
pip install -e .
```

Install FlashAttention and Weights & Biases:

```bash
pip3 install flash-attn --no-build-isolation
pip install wandb
```

---

## 2. Retriever Environment

If a local retriever is used as the search engine, create a separate Conda environment for retrieval.

```bash
conda create -n retriever python=3.10
conda activate retriever
```

Install PyTorch with CUDA support:

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install retrieval dependencies:

```bash
pip install transformers datasets pyserini
```

Install FAISS-GPU:

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

Install API service dependencies:

```bash
pip install uvicorn fastapi
```

---

## 3. Data and Index Preparation

Download the Wikipedia corpus and E5 index:

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

Process the Natural Questions data:

```bash
python scripts/data_process/nq_search.py
```

---

## 4. Start the Retrieval Service

Open a dedicated terminal for the retrieval service.

```bash
conda activate retriever
cd /your-path-to
bash retrieval_launch.sh
```

Keep this terminal running during training.

---

## 5. Before Training

Open a new terminal for RL training.

Before executing any training script, update the model paths, data paths, output paths, and experiment names in the corresponding `.sh` files.

Example variables:

```bash
export DATA_DIR=data/${data_name}
export BASE_MODEL=models/Qwen/Qwen2.5-7B-Instruct
export EXPERIMENT_NAME=v4-grpo
```

Please also check the following items when applicable:

- training data path
- base model path
- checkpoint output path
- retriever endpoint
- judge model path
- Weights & Biases project name
- experiment name

---

## 6. Run Grounded-R1 Training

Make sure the retrieval service is running before launching training.

---

### 6.1 Run v0.4 PPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.4/
bash train_ppo_format_retrieval.sh
```

---

### 6.2 Run v0.4 GRPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.4/
bash train_grpo_format_retrieval.sh
```

---

### 6.3 Run v0.5 PPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.5/
bash launch_llm_server.sh
bash train_ppo_format_retrieval_semantic_score.sh
```

---

### 6.4 Run v0.5 GRPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.5/
bash launch_llm_server.sh
bash train_grpo_format_retrieval_semantic_score.sh
```

---

### 6.5 Run v0.6 PPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.6/
bash vllm_72b.sh
bash train_ppo_format.sh
```

---

### 6.6 Run v0.6 GRPO Training

```bash
conda activate grounded-r1
cd /your-path-to/scripts/nq_hotpotqa/v0.6/
bash vllm_72b.sh
bash train_grpo_format.sh
```

---

## 7. Training Versions

| Version | Stage | Main purpose | Script directory |
| --- | --- | --- | --- |
| `v0.4` | Phase I | Format reward and retrieval-cost penalties | `scripts/nq_hotpotqa/v0.4/` |
| `v0.5` | Phase I | Semantic answer-quality reward with LLM-as-a-judge scoring | `scripts/nq_hotpotqa/v0.5/` |
| `v0.6` | Phase II | Sentence-level retrieval trigger timing optimization | `scripts/nq_hotpotqa/v0.6/` |

---

## 8. Notes

- The retrieval service must be active before RL training starts.
- `v0.5` and `v0.6` require an additional judge model service.
- Use separate terminals for retrieval, judge model serving, and RL training when necessary.
- Make sure the GPU memory is sufficient before launching the judge model and training process together.
- Use different experiment names for different runs to avoid overwriting checkpoints and logs.
- Training logs and checkpoints are saved according to the paths configured in each training script.
