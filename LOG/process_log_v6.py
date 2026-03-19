import re
import json

# log_file = "/data/home/Yifan/Hallu/Search-R1-phase1-main/nq_search-r1-ppo-qwen2.5-3b-it-em-format-retrieval-0924.log"
log_file = "/data/home/Yifan/Hallu/Search-R1-main/v06-grpo--v05-ppo-1016-step200--rind07-continue75-0129.log"
output_file = "logs/v06-grpo--v05-ppo-1016-step200--rind07-continue75-0129.jsonl"

# 正则表达式
reward_pattern = re.compile(r"Batch reward stats - mean:\s*([-+]?\d*\.?\d+), min:\s*([-+]?\d*\.?\d+), max:\s*([-+]?\d*\.?\d+)")
epoch_pattern = re.compile(r"epoch\s+(\d+), step\s+(\d+)")

latest_reward = None
results = []

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # 匹配 reward stats
        reward_match = reward_pattern.search(line)
        if reward_match:
            latest_reward = {
                "mean": float(reward_match.group(1)),
                "min": float(reward_match.group(2)),
                "max": float(reward_match.group(3)),
            }
        
        # 匹配 epoch/step
        epoch_match = epoch_pattern.search(line)
        if epoch_match and latest_reward:
            epoch = int(epoch_match.group(1))
            step = int(epoch_match.group(2))
            entry = {"epoch": epoch, "step": step, **latest_reward}
            results.append(entry)

# 写入 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"解析完成，结果已保存到 {output_file}，共提取 {len(results)} 条记录")
