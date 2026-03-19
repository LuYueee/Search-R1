import re
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_info_from_log(log_path, output_path):
    """
    从日志文件中提取epoch、step、env/number_of_valid_search和response_length/mean信息
    
    参数:
        log_path: 日志文件路径
        output_path: 输出JSONL文件路径
    """
    # 用于匹配epoch和step的正则表达式，考虑可能的乱码
    epoch_step_pattern = re.compile(r'epoch (\d+), step (\d+)', re.IGNORECASE)
    
    # 存储最近出现的指标值
    last_valid_search = None
    last_response_mean = None
    results = []
    
    # try:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        # 遍历所有行
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # 检查是否是env/number_of_valid_search字段
            if line == "env/number_of_valid_search" and i + 1 < len(lines):
                try:
                    # 下一行是数值
                    value = float(lines[i+1].strip())
                    last_valid_search = value
                    logging.debug(f"找到env/number_of_valid_search: {value}")
                except (ValueError, TypeError):
                    logging.warning(f"无法解析env/number_of_valid_search的值: {lines[i+1].strip()}")
            
            # 检查是否是response_length/mean字段
            elif line == "response_length/mean" and i + 1 < len(lines):
                try:
                    # 下一行是数值
                    value = float(lines[i+1].strip())
                    last_response_mean = value
                    logging.debug(f"找到response_length/mean: {value}")
                except (ValueError, TypeError):
                    logging.warning(f"无法解析response_length/mean的值: {lines[i+1].strip()}")
            
            # 检查是否包含epoch和step信息
            epoch_step_match = epoch_step_pattern.search(line)
            if epoch_step_match:
                epoch = int(epoch_step_match.group(1))
                step = int(epoch_step_match.group(2))
                
                # 保存结果
                results.append({
                    'epoch': epoch,
                    'step': step,
                    'env/number_of_valid_search': last_valid_search,
                    'response_length/mean': last_response_mean
                })
                logging.debug(f"提取到记录: epoch={epoch}, step={step}")
                
                # 重置最近的指标值，避免重复使用
                last_valid_search = None
                last_response_mean = None
    
    # 写入JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    
    logging.info(f"处理完成，共提取 {len(results)} 条记录，已保存至 {output_path}")

if __name__ == "__main__":
    log_files =[
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda1.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda2.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda3.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda4.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda5.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda6.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-ppo-lambda6-2.txt',

        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-grpo-1202.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v4-grpo-1203.txt',

        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v05-ppo-1016.txt',
        '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/wandb/txts/v05-ppo-1018.txt',
    ]

    for log_file in log_files:
        output_file = log_file.replace('.txt', '.jsonl')
        extract_info_from_log(log_file, output_file)
    