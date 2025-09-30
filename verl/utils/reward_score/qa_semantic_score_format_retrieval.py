# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import string
import random
import httpx
## for F1
from collections import Counter

# 定义全局 vLLM 服务地址
LLM_API_URL = "http://127.0.0.1:8001/v1/completions"  # MODIFIED

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def f1_check(prediction, golden_answers):
    """
    Compute the maximum token-level F1 overlap between `prediction` and one or more `golden_answers`.
    Returns a float in [0.0, 1.0].
    """
    # Ensure list of answers
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    # Tokenize normalized prediction
    pred_tokens = normalize_answer(prediction).split()
    pred_counts = Counter(pred_tokens)

    best_f1 = 0.0
    for ga in golden_answers:
        gold_tokens = normalize_answer(ga).split()
        gold_counts = Counter(gold_tokens)

        # Count common tokens
        common = pred_counts & gold_counts
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall    = num_same / len(gold_tokens)
            f1        = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def llm_semantic_check(prediction, golden_answers, model_path):
    """
    Uses a locally loaded LLM to score the semantic similarity between
    a candidate prediction and the golden answer, returning a normalized
    score in [0.0, 1.0].
    """
    # Construct the evaluation prompt
    prompt = (
        "# Role  \n"
        "You are an objective evaluator comparing a candidate response to a golden answer.\n\n"
        "# Instructions  \n"
        "1. Rate on five dimensions (0–5 integers):  \n"
        "   - **70%** Semantic Accuracy  \n"
        "   - **7.5%** Completeness  \n"
        "   - **7.5%** Logical Coherence  \n"
        "   - **7.5%** Clarity  \n"
        "   - **7.5%** Fluency  \n"
        "2. Compute overall similarity (0–100):  \n"
        "   overall = (0.70×SA + 0.075×(CMP+LC+CLR+FL)) × 20  \n"
        "3. **Output only the overall score** No other fields or text or sql statements.\n\n"
        "# Few-Shot  \n"
        "Golden: Apple founded by Jobs/Wozniak (1976)  \n"
        "Candidate: Jobs & Wozniak co‑founded Apple (1976), launching the PC era  \n"
        "→ `100`  \n\n"
        "Golden: Atmospheric layers: troposphere/stratosphere/mesosphere/thermosphere/exosphere  \n"
        "Candidate: Atmosphere: troposphere & stratosphere  \n"
        "→ `20`  \n\n"
        "Golden: Deep learning stages: data preprocessing/model design/training/evaluation  \n"
        "Candidate: DL workflow: data cleaning/network design/training/testing  \n"
        "→ `100`  \n\n"
        "Golden: EV benefits: zero emissions/high efficiency/low maintenance  \n"
        "Candidate: EVs produce no pollutants but require charging networks  \n"
        "→ `67`  \n\n"
        "# Evaluation  \n"
        f"Candidate: {prediction}  \n"
        f"Golden: {golden_answers}  \n"
    )

    # 调用 HTTP 接口，增加重试／异常捕获
    try:
        response = httpx.post(
            LLM_API_URL,
            json={
                "model": model_path,
                "prompt": prompt,
                "max_tokens": 10,       # 对应 max_new_tokens=10
                "temperature": 0.01,    # 对应 temperature=0.01
                "top_p": 1.0,           # 等同不截断采样
                "n": 1
            },
            timeout=None
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["text"].strip()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"[WARNING] LLM service request failed: {e!r}")
        raw = ""  # 退回空，下面会变 score100=0
    except ValueError as e:  # JSONDecodeError 属于 ValueError
        print(f"[WARNING] LLM service returned invalid JSON: {response.text!r}")
        raw = ""

    # 3) 提取数字并清洗
    m = re.search(r"\d{1,3}", raw)
    if not m:
        score100 = 0
    else:
        v = int(m.group())
        score100 = v if 0 <= v <= 100 else 0
    
    # 4) 归一化到 0–1
    semantic_score = score100 / 100.0
    return semantic_score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def count_repeat_information(text: str) -> int:
    # Count repeated information blocks
    info_blocks = extract_information_blocks(text)
    seen = set()
    repeat_count = 0
    for block in info_blocks:
        if block in seen:  # check if exactly match
            repeat_count += 1
        else:
            seen.add(block)
    return repeat_count


def count_search_tags(text: str) -> int:
    """
    Count the number of complete <search>...</search> blocks in the text.
    Only fully closed tags are counted.
    """
    pattern = r"<search>(.*?)</search>"
    matches = re.findall(pattern, text, re.DOTALL)
    return len(matches)


def compute_score_em(solution_str, ground_truth, model_path, structure_format_score=0, final_format_score=0, lambda_task=2, lambda_search_num=0.1, lambda_repeat_search_num=0.1, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)

    response_str = solution_str[solution_str.find('<|im_start|>assistant') + 21:] if '<|im_start|>assistant' in solution_str else solution_str
    answer = extract_solution(response_str)

    ###################
    do_print = random.randint(1, 64) == 1
    #do_print=1
    ##################
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
   
    llm_score = 0
    f1_score = 0
    final_em_format_score = 0
    if answer is None:
        if is_valid_format:
            final_em_format_score = structure_format_score
        else:
            final_em_format_score = 0
    else:
        # semantic_score ∈ [0,1]
        # is_valid_format ∈ {0,1}
        # structure_format_score = lambda_f ∈ [0,1]
        semantic_score = em_check(answer, ground_truth['target'])
        if not semantic_score:
            llm_score = llm_semantic_check(answer, ground_truth['target'], model_path)
            f1_score = f1_check(answer, ground_truth['target'])
            semantic_score = max(f1_score, llm_score)

        # 统一公式实现（加入 final_format_score 兜底项）：
        final_em_format_score = (
            semantic_score
            + structure_format_score * (
                is_valid_format * (1 - semantic_score)
                - (1 - is_valid_format) * semantic_score
            )
            + (1 - semantic_score) * (1 - is_valid_format) * final_format_score  # ← 新增final_format_score这一项
        )
        # 为了防止数值漂移，可再做一次裁剪，确保 0 ≤ score ≤ 1
        final_em_format_score = max(0.0, min(1.0, final_em_format_score))
    
    # Rewards for redundant retrieval
    n_search = count_search_tags(response_str)
    n_repeat = count_repeat_information(response_str)  

    # Apply penalties and calculate final reward
    #final_score = max(0, lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat)
    final_score = lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat
    
    if do_print:
        print(f"EM Score: {em_check(answer, ground_truth['target'])}")
        print(f"LLM Semantic Score: {llm_score}")
        print(f"F-1 Score: {f1_score}")
        print(f"Format Valid: {is_valid_format}")
        print(f"Lambda task: {lambda_task}")
        print(f"Base Reward: {final_em_format_score}")
        print(f"Lambda search num: {lambda_search_num}")
        print(f"Search Count: {n_search} (Penalty: -{lambda_search_num * n_search})")
        print(f"Lambda repeat search num: {lambda_repeat_search_num}")
        print(f"Repeat Count: {n_repeat} (Penalty: -{lambda_repeat_search_num * n_repeat})")
        print(f"Final Reward: {final_score}\n")
    return final_score
