import re
import json
from collections import defaultdict

def extract_training_metrics(log_file_path, output_jsonl_path):
    """
    从训练日志文件中提取指定指标，保存为JSONL格式
    :param log_file_path: 输入的日志文件路径
    :param output_jsonl_path: 输出的JSONL文件路径
    """
    # 定义正则表达式（匹配各类指标，兼容日志中的杂乱字符）
    patterns = {
        # 1. 样本平均奖励、奖励标准差
        'sample_return': re.compile(
            r'\[DEBUG\]\[Step\] sample/return_mean:\s*([-+]?\d+\.\d+),\s*/std:\s*([-+]?\d+\.\d+)'
        ),
        # 1. 补充：样本句子数 len_mean
        'sample_len_mean': re.compile(
            r'/len_mean:\s*([-+]?\d+\.\d+)'
        ),
        # 2. EM奖励均值
        'em_reward_mean': re.compile(
            r'\[DEBUG\]\[EM\+Format\] Batch reward stats - mean:\s*([-+]?\d+\.\d+)'
        ),
        # 3. Macro-Dec 混淆矩阵（TP%/FN%/FP%/TN%/Precision/Recall/F1）
        'macro_dec_confusion': re.compile(
            r'\[DEBUG\]\[Confusion\]\[Macro-Dec\] TP%:([-+]?\d+\.\d+),\s*FN%:([-+]?\d+\.\d+),\s*FP%:([-+]?\d+\.\d+),\s*TN%:([-+]?\d+\.\d+),\s*Precision:([-+]?\d+\.\d+),\s*Recall:([-+]?\d+\.\d+),\s*F1:([-+]?\d+\.\d+)'
        ),
        # 4. Macro-All 混淆矩阵（含DecisionCoverage）
        'macro_all_confusion': re.compile(
            r'\[DEBUG\]\[Confusion\]\[Macro-All\] TP%:([-+]?\d+\.\d+),\s*FN%:([-+]?\d+\.\d+),\s*FP%:([-+]?\d+\.\d+),\s*TN%:([-+]?\d+\.\d+),\s*Precision:([-+]?\d+\.\d+),\s*Recall:([-+]?\d+\.\d+),\s*F1:([-+]?\d+\.\d+),\s*DecisionCoverage:([-+]?\d+\.\d+)'
        ),
        # 5. Sentence-Decision 正负奖励比例
        'sentence_decision_rate': re.compile(
            r'\[DEBUG\]\[Sentence\]\[Decision\] pos_rate:\s*([-+]?\d+\.\d+),\s*neg_rate:\s*([-+]?\d+\.\d+)'
        ),
        # 6. Sentence-All 正负奖励比例
        'sentence_all_rate': re.compile(
            r'\[DEBUG\]\[Sentence\]\[All\] pos_rate:\s*([-+]?\d+\.\d+),\s*neg_rate:\s*([-+]?\d+\.\d+)'
        )
    }

    # 存储每个step的所有指标（默认空值）
    step_data = defaultdict(lambda: {
        'step': 0,
        'reward_mean': None,       # 样本平均奖励
        'reward_std': None,        # 奖励标准差
        'len_mean': None,          # 样本句子数
        'em_reward_mean': None,    # EM奖励均值
        # Macro-Dec 指标
        'macro_dec_tp_pct': None,
        'macro_dec_fn_pct': None,
        'macro_dec_fp_pct': None,
        'macro_dec_tn_pct': None,
        'macro_dec_precision': None,
        'macro_dec_recall': None,
        'macro_dec_f1': None,
        # Macro-All 指标
        'macro_all_tp_pct': None,
        'macro_all_fn_pct': None,
        'macro_all_fp_pct': None,
        'macro_all_tn_pct': None,
        'macro_all_precision': None,
        'macro_all_recall': None,
        'macro_all_f1': None,
        'macro_all_decision_coverage': None,
        # 正负奖励比例
        'sentence_decision_pos_rate': None,
        'sentence_decision_neg_rate': None,
        'sentence_all_pos_rate': None,
        'sentence_all_neg_rate': None
    })

    current_step = 1  # step从1开始计数

    # 读取并解析日志文件
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 1. 匹配样本奖励和句子数
            return_match = patterns['sample_return'].search(line)
            if return_match:
                step_data[current_step]['reward_mean'] = float(return_match.group(1))
                step_data[current_step]['reward_std'] = float(return_match.group(2))
                
                # 提取同行的len_mean
                len_match = patterns['sample_len_mean'].search(line)
                if len_match:
                    step_data[current_step]['len_mean'] = float(len_match.group(1))

            # 2. 匹配EM奖励均值
            em_match = patterns['em_reward_mean'].search(line)
            if em_match:
                step_data[current_step]['em_reward_mean'] = float(em_match.group(1))

            # 3. 匹配Macro-Dec混淆矩阵（检索决策TP/FP/FN/TN+精度/召回/F1）
            dec_confusion_match = patterns['macro_dec_confusion'].search(line)
            if dec_confusion_match:
                step_data[current_step]['macro_dec_tp_pct'] = float(dec_confusion_match.group(1))
                step_data[current_step]['macro_dec_fn_pct'] = float(dec_confusion_match.group(2))
                step_data[current_step]['macro_dec_fp_pct'] = float(dec_confusion_match.group(3))
                step_data[current_step]['macro_dec_tn_pct'] = float(dec_confusion_match.group(4))
                step_data[current_step]['macro_dec_precision'] = float(dec_confusion_match.group(5))  # 检索触发精度
                step_data[current_step]['macro_dec_recall'] = float(dec_confusion_match.group(6))     # 检索触发召回
                step_data[current_step]['macro_dec_f1'] = float(dec_confusion_match.group(7))         # 检索触发F1

            # 4. 匹配Macro-All混淆矩阵
            all_confusion_match = patterns['macro_all_confusion'].search(line)
            if all_confusion_match:
                step_data[current_step]['macro_all_tp_pct'] = float(all_confusion_match.group(1))
                step_data[current_step]['macro_all_fn_pct'] = float(all_confusion_match.group(2))
                step_data[current_step]['macro_all_fp_pct'] = float(all_confusion_match.group(3))
                step_data[current_step]['macro_all_tn_pct'] = float(all_confusion_match.group(4))
                step_data[current_step]['macro_all_precision'] = float(all_confusion_match.group(5))
                step_data[current_step]['macro_all_recall'] = float(all_confusion_match.group(6))
                step_data[current_step]['macro_all_f1'] = float(all_confusion_match.group(7))
                step_data[current_step]['macro_all_decision_coverage'] = float(all_confusion_match.group(8))

            # 5. 匹配Sentence-Decision正负奖励比例
            dec_rate_match = patterns['sentence_decision_rate'].search(line)
            if dec_rate_match:
                step_data[current_step]['sentence_decision_pos_rate'] = float(dec_rate_match.group(1))
                step_data[current_step]['sentence_decision_neg_rate'] = float(dec_rate_match.group(2))

            # 6. 匹配Sentence-All正负奖励比例
            all_rate_match = patterns['sentence_all_rate'].search(line)
            if all_rate_match:
                step_data[current_step]['sentence_all_pos_rate'] = float(all_rate_match.group(1))
                step_data[current_step]['sentence_all_neg_rate'] = float(all_rate_match.group(2))

            # 关键：每收集完一组完整指标，step+1（可根据日志实际节奏调整，这里按指标完整性判断）
            # 简单判断：至少包含奖励均值+EM奖励+Macro-Dec精度，视为一个step完成
            if (step_data[current_step]['reward_mean'] is not None and
                step_data[current_step]['em_reward_mean'] is not None and
                step_data[current_step]['macro_dec_precision'] is not None):
                step_data[current_step]['step'] = current_step
                current_step += 1

    # 将数据写入JSONL文件（过滤空step）
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for step in sorted(step_data.keys()):
            data = step_data[step]
            if data['step'] > 0:  # 只保存有效step
                # 移除值为None的字段（可选，也可保留为null）
                cleaned_data = {k: v for k, v in data.items() if v is not None}
                out_f.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')

    print(f"数据提取完成！已保存至 {output_jsonl_path}")
    print(f"共提取 {current_step-1} 个训练步的数据")


if __name__ == "__main__":
    log_files = [
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v6-grpo-rind07-0127.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v6-grpo-rind056-0117.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v6-grpo-rind056-0120.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v6-ppo-rind07-0108.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v6-ppo-rind056-0108.txt',
    ]

    # 批量处理所有文件
    for log_file in log_files:
        output_file = log_file.replace('.txt', '.jsonl')
        extract_training_metrics(log_file, output_file)