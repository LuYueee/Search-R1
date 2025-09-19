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
## for F1
from collections import Counter
# for SF LLM score
import requests  
import json
import time

LLM_API_URL = "http://llm-model-hub-apis.sf-express.com"  # http://llm-model-hub-apis.int.sfcloud.local:1080
API_KEY = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJhcGkyX2VkMjM2YWI4LTJhZDMtNDE2Yy05ZTMyLWU4OWJjZGVjYjYyNyIsImVudiI6InByZCIsImp0aSI6MjA4NDIsInByb2plY3RfaWQiOjQ2Niwic3lzdGVtS2V5IjoiYTVmMDI4NjItMmRjYS00YzJmLTk3MDktZjdmMThhYjMzMzNlIn0.tpkfC0aaxiiVklYT6tXq-OxbKxnyN4y-IW8NtSdCYAU"  # === MODIFIED: 填上申请到的 token

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


def llm_semantic_check(prediction, golden_answers, model_path: str = "aiplat/Qwen3-30B-A3B"):
    """
    Uses a locally loaded LLM to score the semantic similarity between
    a candidate prediction and the golden answer, returning a normalized
    score in [0.0, 1.0].
    """
    # Construct the evaluation prompt
    prompt = (
    "You are a strict semantic matching judge.\n"
    "Judge ONLY semantic equivalence between a Candidate and a Golden reference.\n"
    "Ignore style, wording, order, fluency, grammar, and formatting.\n"
    "\n"
    "# What to compare\n"
    "- Decide whether Candidate and Golden express the SAME set of atomic propositions (facts/claims).\n"
    "- Use bidirectional entailment:\n"
    "  * Golden ⇒ Candidate (recall of Golden’s facts)\n"
    "  * Candidate ⇒ Golden (penalize extra/new claims not supported by Golden)\n"
    "- Treat paraphrases, synonyms, reordering, casing, and minor function words as equivalent.\n"
    "- Numbers, dates, and named entities must match semantically (e.g., \"the capital of the UK\" ≡ \"the capital of the United Kingdom\").\n"
    "- Do NOT reward surface word overlap without meaning match.\n"
    "- If Candidate is empty and Golden is non-empty → score 0.\n"
    "- If content is unrelated or contradicts Golden → score 0 (or very low).\n"
    "\n"
    "# Scoring = semantic equivalence first (wording/order/style irrelevant). Rules:\n"
    "1) Unrelated or off-topic → 0.\n"
    "2) Empty or whitespace candidate → 0.\n"
    "3) Contradiction on key facts/entities/numbers → ≤25.\n"
    "4) Missing key entity/attribute/time/number → 20–80 depending on severity.\n"
    "5) Paraphrase / same meaning (including word order change, synonyms) → 95–100.\n"
    "6) Minor wording differences only → 100.\n"
    "7) Case and punctuation are ignored.\n"
    "\n"
    "# Primary semantic alignment (SA, 0–100)\n"
    "1) Extract atomic propositions from Golden: N.\n"
    "2) S = number of Golden propositions supported by Candidate (entailment or clear paraphrase).\n"
    "3) C = number of Golden propositions contradicted by Candidate.\n"
    "4) E = number of extra propositions asserted by Candidate not entailed by Golden.\n"
    "5) Raw = (S - 0.5*C - 0.25*E) / max(N,1); clamp to [0,1].\n"
    "6) SA = round(100 * Raw).\n"
    "\n"
    "# Secondary readability factors (10% cap)\n"
    "Rate four dimensions ONLY if they materially affect understanding; otherwise set to 100.\n"
    "- Completeness (CMP): coverage of Golden’s required facts.\n"
    "- Logical Coherence (LC): internally consistent.\n"
    "- Clarity (CLR): unambiguous phrasing.\n"
    "- Fluency (FL): readable text.\n"
    "Each in {0, 50, 100}; average them to get M (0–100).\n"
    "\n"
    "# Final score (0–100, integer)\n"
    "Overall = round(0.90 * SA + 0.10 * M).\n"
    "Return ONLY the integer Overall (0–100). No other text.\n"
    "\n"
    "# Few-shot sanity checks (return only the number):\n"
    "Golden: Apple founded by Jobs/Wozniak (1976)\n"
    "Candidate: Apple was founded by Jobs.\n"
    "55\n"
    "\n"
    "Golden: Apple founded by Jobs/Wozniak (1976)\n"
    "Candidate: Giant pandas live in Sichuan and eat bamboo.\n"
    "0\n"
    "\n"
    "Golden: hello world\n"
    "Candidate: Hello, world!\n"
    "100\n"
    "\n"
    "Golden: the cat\n"
    "Candidate:\n"
    "0\n"
    "\n"
    "Golden: Apple founded by Jobs/Wozniak (1976)  \n"
    "Candidate: Jobs & Wozniak co-founded Apple (1976), launching the PC era  \n"
    "→ `100`  \n"
    "\n"
    "Golden: Atmospheric layers: troposphere/stratosphere/mesosphere/thermosphere/exosphere  \n"
    "Candidate: Atmosphere: troposphere & stratosphere  \n"
    "→ `20`  \n"
    "\n"
    "Golden: Deep learning stages: data preprocessing/model design/training/evaluation  \n"
    "Candidate: DL workflow: data cleaning/network design/training/testing  \n"
    "→ `100`  \n"
    "\n"
    "Golden: EV benefits: zero emissions/high efficiency/low maintenance  \n"
    "Candidate: EVs produce no pollutants but require charging networks  \n"
    "→ `67`  \n"
    "\n"
    "# Additional examples for missing facts\n"
    "Golden: The law of universal gravity was proposed by Isaac Newton, and it describes the attraction between objects.\n"
    "Candidate: Newton discovered the law of universal gravity.\n"
    "50\n"
    "\n"
    "Golden: Under standard atmospheric pressure, the common phases of water are solid, liquid, and gas, with liquid being the most common.\n"
    "Candidate: Water exists in liquid form at room temperature.\n"
    "50\n"
    "\n"
    "# Evaluation\n"
    )
        
    user_content = (
        f"Candidate: {prediction}\n"
        f"Golden: {golden_answers}"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": user_content},
    ]

    url = f"{LLM_API_URL}/v1/chat/completions"
    headers = {
        "Authorization": API_KEY,                         # === MODIFIED
        "Content-Type":  "application/json",
    }
    '''
    payload = {
        "model":  model_path,                             # === MODIFIED: 使用公有模型名
        "messages": messages,
        "max_tokens":  10,    # 最多生成 10 个 token
        "temperature": 0.01,  # 几乎是贪心解码
        "top_p":       1.0,   # 不做截断采样
        "n":           1,     # 返回一个候选
        "stream":      False, # 关闭流式
    }
    '''
    payload = {
        "model":  model_path,                             # === MODIFIED: 使用公有模型名
        "messages": messages,
        "max_tokens":  10,    # 最多生成 10 个 token
        "temperature": 0.01,  # 几乎是贪心解码
        "top_p":       1.0,   # 不做截断采样
        "n":           1,     # 返回一个候选
        "stream":      False, # 关闭流式
        "chat_template_kwargs": {"enable_thinking":False} # === MODIFIED: 关闭思考模式
    }
    # === MODIFIED: 发起请求
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=None)
        time.sleep(0.1)
        resp.raise_for_status()
        data = resp.json()
        # 从 choices[0].message.content 里提取数字
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARNING] LLM API call failed: {e}")
        content = ""

    # 提取 0-100 整数并清洗
    m = re.search(r"\d{1,3}", content or "")
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

    answer = extract_solution(solution_str=solution_str)

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
            llm_score = llm_semantic_check(answer, ground_truth['target'], "aiplat/Qwen3-30B-A3B")
            f1_score = f1_check(answer, ground_truth['target'])
            semantic_score = max(f1_score, llm_score)

        # 统一公式实现：
        final_em_format_score = (
            semantic_score
            + structure_format_score * (
                is_valid_format * (1 - semantic_score)
                - (1 - is_valid_format) * semantic_score
            )
        )
        # 为了防止数值漂移，可再做一次裁剪，确保 0 ≤ score ≤ 1
        final_em_format_score = max(0.0, min(1.0, final_em_format_score))
    
    # Rewards for redundant retrieval
    response_str = solution_str[solution_str.find('<|im_start|>assistant') + 21:] if '<|im_start|>assistant' in solution_str else solution_str
    n_search = count_search_tags(response_str)
    n_repeat = count_repeat_information(response_str)  

    # Apply penalties and calculate final reward
    final_score = lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat
    #final_score = max(0, lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat)
    
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
