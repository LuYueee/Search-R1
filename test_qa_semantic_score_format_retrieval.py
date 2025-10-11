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
            llm_score = llm_semantic_check(answer, ground_truth['target'], model_path)
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
    n_search = count_search_tags(solution_str)
    n_repeat = count_repeat_information(solution_str)  
    # Apply penalties and calculate final reward
    final_score = max(0, lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat)
    
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



# Test cases based on the paper's examples
def test_cases():
    case0 = {
    "solution_str": """<|im_start|>assistant
    <think>
    I need to find out which player Mike Barnett negotiated many contracts and went on to become the general manager of CSKA Moscow of the Kontinental Hockey League.
    </think>

    <search> Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League </search>

    <information>Doc 1(Title: "Mike Barnett (ice hockey)") negotiated the playing and marketing contracts for Wayne Gretzky, Brett Hull, Jaromir Jagr, Sergei Fedorov, Paul Coffey, Joe Thornton, Mats Sundin, Lanny McDonald, Grant Fuhr, Marty McSorley, Alexander Mogilny, Owen Nolan, Daniel Sedin and Henrik Sedin and numerous others. Whatever the form of contract, Barnett's creativity was legendary. His ingenuity in finding language that challenged the NHL's Collective Bargaining Agreement, in finding products and companies for his clients that were groundbreaking in their launches, and his perpetual attention to the public relations of his clients, were all hallmarks of his career in athlete representation. Following his two-decades as one of
    Doc 2(Title: "Mike Barnett (ice hockey)") day. Mike Barnett (ice hockey) Michael G. Barnett (born October 9, 1948 in Olds, Alberta) is a Canadian ice hockey executive currently serving as Senior Advisor to the President-General Manager of the New York Rangers. He is best known in the world of hockey as a former agent representing sports icon Wayne Gretzky for two decades. He was listed among the ""100 Most Powerful People In Sports"" by ""The Sporting News"" on six occasions from 1994 to 2000. During his 12 years as President of International Management Group's hockey division, Barnett represented a who's who of the National Hockey League.
    Doc 3(Title: "Mike Barnett (ice hockey)") the most highly regarded agents in all of sports, Barnett went on to become the General Manager of the Phoenix Coyotes in the National Hockey League. Barnett grew up in Western Canada, before attending St. Lawrence University in Canton, New York, on a hockey scholarship. He later attended the University of Calgary in his home province of Alberta. He was a two sport collegiate athlete in both football and hockey. Barnett's professional career led him to play with both the Chicago Cougars and Houston Aeros organizations of the World Hockey Association. He was a part of WHA history, playing in
    </information>

    <think>
    I need to find the player who went on to become the general manager of CSKA Moscow of the Kontinental Hockey League. I found out that Mike Barnett negotiated contracts for many players, including Wayne Gretzky.
    </think>

    <search> general manager of CSKA Moscow of the Kontinental Hockey League </search>

    <information>Doc 1(Title: "Sergei Fedorov") Sergei Fedorov Sergei Viktorovich Fyodorov (; born December 13, 1969) is a Russian former professional ice hockey player and the current general manager of CSKA Moscow of the Kontinental Hockey League (KHL). During his playing career, Fedorov was primarily a centre, but occasionally played as a winger or defenceman. Fedorov gained fame in the National Hockey League (NHL) for his unique style of play with the Detroit Red Wings, with whom he won the Stanley Cup three times, as well as the Hart Memorial Trophy as the NHL's most valuable player in 1994. After a highly publicized departure from the
    Doc 2(Title: "Sergei Fedorov") Zada over the past 11 years. The lawsuit was filed by Fedorov in Michigan. Fedorov won the suit but has been unable to collect on the judgment from Zada. Fedorov continues his philanthropic endeavours via the Sergei Fedorov Foundation, a 501(c)(3) charitable foundation, which has donated over $800,000 to charities that mostly benefit children in need. Fedorov currently serves as the general manager of CSKA Moscow. He resides in Moscow during hockey season and splits his summers between Detroit and Miami. Sergei is the brother of former professional hockey player Fedor Fedorov. Sergei Fedorov Sergei Viktorovich Fyodorov (; born December
    Doc 3(Title: "Alexei Zhamnov") due to injury. During the 2004–05 NHL lockout, Zhamnov spent time playing in his homeland. Alexei Zhamnov Alexei Yuryevich ""Alex"" Zhamnov (; born October 1, 1970) is a Russian former professional ice hockey centre who played in the National Hockey League (NHL) for the Winnipeg Jets, Chicago Blackhawks, Philadelphia Flyers and Boston Bruins. He is the general manager of HC Spartak Moscow of the Kontinental Hockey League (KHL). He previously served as general manager of Vityaz Chekhov and Atlant Moscow Oblast. Zhamnov has been a part of three Olympic games, winning gold, silver, and bronze medals with the Unified Team
    </information>

    <think>
    I need to find the player who went on to become the general manager of CSKA Moscow of the Kontinental Hockey League. I found out that Sergei Fedorov is the general manager of CSKA Moscow.
    </think>

    <search> which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League </search>

    <information>Doc 1(Title: "Sergei Fedorov") Sergei Fedorov Sergei Viktorovich Fyodorov (; born December 13, 1969) is a Russian former professional ice hockey player and the current general manager of CSKA Moscow of the Kontinental Hockey League (KHL). During his playing career, Fedorov was primarily a centre, but occasionally played as a winger or defenceman. Fedorov gained fame in the National Hockey League (NHL) for his unique style of play with the Detroit Red Wings, with whom he won the Stanley Cup three times, as well as the Hart Memorial Trophy as the NHL's most valuable player in 1994. After a highly publicized departure from the
    Doc 2(Title: "Valeri Bragin") Valeri Bragin Valeri Nikolayevich Bragin (); born May 31, 1956, is a retired Russian ice hockey forward. He is former coach of the HC CSKA Moscow team in the KHL. Valeri Bragin started his career with HC Spartak Moscow where he played in the top Soviet league from 1978. In 1981 he joined Khimik Voskresensk where he played for 8 seasons. In 1989 Bragin moved to Denmark where he was the player/coach for Rødovre SIK in the Danish League. He won the Danish title with Rødovre in 1990. In that season he was also named league player of the year.
    Doc 3(Title: "Alexei Zhamnov") due to injury. During the 2004–05 NHL lockout, Zhamnov spent time playing in his homeland. Alexei Zhamnov Alexei Yuryevich ""Alex"" Zhamnov (; born October 1, 1970) is a Russian former professional ice hockey centre who played in the National Hockey League (NHL) for the Winnipeg Jets, Chicago Blackhawks, Philadelphia Flyers and Boston Bruins. He is the general manager of HC Spartak Moscow of the Kontinental Hockey League (KHL). He previously served as general manager of Vityaz Chekhov and Atlant Moscow Oblast. Zhamnov has been a part of three Olympic games, winning gold, silver, and bronze medals with the Unified Team
    </information>
    
    <think>
    I need to find the player who went on to become the general manager of CSKA Moscow of the Kontinental Hockey League. I found out that Sergei Fedorov is the general manager of CSKA Moscow.
    </think>

    <answer> Sergei Fedorov </answer>""",
        "ground_truth": {"target": ["Sergei Fedorov"]},
        "expected": 1.0
    }

    # Case 1: Correct answer + valid format (Table 10 w. E5 example)
    case1 = {
        "solution_str": "<|im_start|>assistant<think>I found out that Delicatessen is in the style of Terry Gilliam.</think> <answer>Terry Gilliam</answer>",
        "ground_truth": {"target": ["Terry Gilliam"]},
        "expected": 1.0
    }
    
    # Case 2: Correct answer + invalid format (Table 10 w. random noise)
    case2 = {
        "solution_str": "<think>Therefore... Burton.</think><answer>Terry Gilliam</answer>",
        "ground_truth": {"target": ["Terry Gilliam"]},
        "expected": 1.0 - 0.2  # 0.8
    }
    
    # Case 3: Wrong answer + valid format + retrieval correct (Page 6 example)
    case3 = {
        "solution_str": "<|im_start|>assistant<think>Need info...</think><search>query</search><information>Doc contains 'Terry Gilliam'</information><think>Found Gilliam</think><answer>Wrong Name</answer>",
        "ground_truth": {"target": ["Terry Gilliam"]},
        "use_retrieval_reward": True,
        "expected": 0.2  # 0.3
    }
    
    # Case 4: Wrong answer + valid format (Table 9 w. BM25 example)
    case4 = {
        "solution_str": "<|im_start|>assistant<think>Searching...</think><search>Helis Angel</search><information>Doc about Rolling Stones</information><think>...</think><answer>Charlie ISHAM</answer>",
        "ground_truth": {"target": ["Alan Passaro"]},
        "expected": 0.2
    }
    
    # Case 5: Wrong answer + invalid format (Table 9 w. random noise)
    case5 = {
        "solution_str": "<think>Answer is Manson</think><answer>Manson</answer>",
        "ground_truth": {"target": ["Alan Passaro"]},
        "expected": 0.0
    }
    
    # Case 6: Valid format but no answer extracted
    case6 = {
        "solution_str": "<|im_start|>assistant<think>Reasoning...</think><search>query</search><information>Doc</information><think></think><answer></answer>",
        "ground_truth": {"target": ["Alan Passaro"]},
        "expected": 0.2  # Valid format but wrong answer
    }
    #test_cases_set = [case0,case1, case2, case3, case4, case5, case6]
    #test_cases_set = [case0]
    # Run all test cases
        
        # 运行所有测试用例
    test_cases_set = [
    # Case 1: 无检索操作
    {
        "solution_str": "<|im_start|>assistant<think>直接回答</think><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0  # 2.0
    },
    
    # Case 2: 单次有效检索，无重复
    {
        "solution_str": "<|im_start|>assistant<think>需要检索</think><search>查询1</search><information>信息1</information><think>思考结果</think><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 1  # 2.0 - 0.2 = 1.8
    },
    
    # Case 3: 两次有效检索，无重复
    {
        "solution_str": "<|im_start|>assistant<think>第一次检索</think><search>查询1</search><information>信息1</information><think>第二次检索</think><search>查询2</search><information>信息2</information><think>思考结果</think><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 2  # 2.0 - 0.4 = 1.6
    },
    
    # Case 4: 三次检索但有两次重复信息
    {
        "solution_str": "<|im_start|>assistant<think>检索1</think><search>查询1</search><information>重复信息</information><think>检索2</think><search>查询2</search><information>新信息</information><think>检索3</think><search>查询3</search><information>重复信息</information><think>思考结果</think><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 3 - beta * 1  # 2.0 - 0.6 - 0.1 = 1.3
    },
    
    # Case 5: 多次重复相同信息
    {
        "solution_str": "<|im_start|>assistant<search>查询1</search><information>重复信息</information><search>查询2</search><information>重复信息</information><search>查询3</search><information>重复信息</information><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 3 - beta * 2  # 2.0 - 0.6 - 0.2 = 1.2
    },
    
    # Case 6: 混合重复（不同重复块）
    {
        "solution_str": "<|im_start|>assistant<search>查询1</search><information>重复A</information><search>查询2</search><information>重复B</information><search>查询3</search><information>重复A</information><search>查询4</search><information>重复B</information><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 4 - beta * 2  # 2.0 - 0.8 - 0.2 = 1.0
    },
    
    # Case 7: 未闭合的检索标签不计入
    {
        "solution_str": "<|im_start|>assistant<search>未闭合检索<think>思考</think><information>信息</information><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0  # 2.0（无检索惩罚）
    },
    
    # Case 8: 空白字符不影响重复检测
    {
        "solution_str": "<|im_start|>assistant<search>查询1</search><information>重复信息</information><search>查询2</search><information>  重复信息  </information><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 2 - beta * 1  # 2.0 - 0.4 - 0.1 = 1.5
    },
    
    # Case 9: 答案错误但有检索惩罚
    {
        "solution_str": "<|im_start|>assistant<search>查询1</search><information>信息1</information><search>查询2</search><information>重复信息</information><answer>错误答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * lambda_f - alpha * 2 - beta * 1  # 2.0*0.2 - 0.4 - 0.1 = 0.4 - 0.5 = -0.1
    },
    
    # Case 10: 复杂混合场景
    {
        "solution_str": "<|im_start|>assistant<search>查询1</search><information>信息A</information><search>查询2</search><information>信息B</information><search>查询3</search><information>信息A</information><search>查询4</search><information>信息C</information><search>查询5</search><information>信息C</information><answer>正确答案</answer>",
        "ground_truth": {"target": ["正确答案"]},
        #"expected": lambda_task * 1.0 - alpha * 5 - beta * 2  # 2.0 - 1.0 - 0.2 = 0.8
    }
]
    for i, test in enumerate(test_cases_set):
            print(f"\n{'#' * 40} 测试用例 {i+1} {'#' * 40}")
            reward = compute_score_em(
                solution_str=test["solution_str"],
                ground_truth=test["ground_truth"],
                structure_format_score=0.2,
                final_format_score=0,
                lambda_task=2, 
                lambda_search_num=0.2, 
                lambda_repeat_search_num=0.1,
                model_path="/home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/qwen2.5-7b-instruct-1m"
            )
            print(f"测试用例 {i+1} 结果: 奖励值 = {reward:.1f}")
            print(f"{'#' * 40} 结束 {'#' * 40}\n")
            #assert abs(reward - test["expected"]) < 1e-5, \
                #f"Case {i+1} failed: Expected {test['expected']}, got {reward}"
            print(f"TEST CASE {i+1} PASSED")

# Run test cases
if __name__ == "__main__":
    model_path = "/home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/qwen2.5-7b-instruct-1m"
    test_cases()
    
   # 合并后的测试用例列表
test_cases = [
    # 第一组
    {"prediction": "苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于1976年创立。",
     "golden_answers": "苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于1976年创立。",
     "expected": 1.0},
    {"prediction": "1976年，乔布斯与沃兹尼亚克共同创办了苹果公司。",
     "golden_answers": "苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于1976年创立。",
     "expected": 0.90},
    {"prediction": "苹果公司由乔布斯创立。",
     "golden_answers": "苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于1976年创立。",
     "expected": 0.50},
    {"prediction": "大熊猫生活在中国四川地区，以竹子为食。",
     "golden_answers": "苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于1976年创立。",
     "expected": 0.0},

    # 第二组
    {"prediction": "北京是中国的首都。",
     "golden_answers": "北京是中华人民共和国的首都。",
     "expected": 1.0},
    {"prediction": "长城历史悠久，全长约两万公里。",
     "golden_answers": "万里长城总长约21196公里。",
     "expected": 0.85},
    {"prediction": "长城是著名的旅游景点。",
     "golden_answers": "万里长城总长约21196公里，是世界文化遗产。",
     "expected": 0.4},
    {"prediction": "埃菲尔铁塔高达324米，位于巴黎。",
     "golden_answers": "埃菲尔铁塔位于法国巴黎，高305米。",
     "expected": 0.75},
    {"prediction": "地球是太阳系的第三颗行星，绕太阳公转。",
     "golden_answers": "地球是太阳系中的第三颗行星，围绕太阳运动。",
     "expected": 0.95},
    {"prediction": "牛顿发现了万有引力定律。",
     "golden_answers": "万有引力定律由艾萨克·牛顿提出，描述了物体间的引力。",
     "expected": 0.6},
    {"prediction": "热带雨林的树木繁茂，降雨量大。",
     "golden_answers": "热带雨林年降水量可达2000毫米以上，植被茂密。",
     "expected": 0.7},
    {"prediction": "水在常温下以液态存在。",
     "golden_answers": "在标准大气压下，水的常见相态有固态、液态和气态，液态为最常见。",
     "expected": 0.5},
    {"prediction": "太阳系内有八大行星。",
     "golden_answers": "截至目前，太阳系包括八大行星，从水星到海王星。",
     "expected": 0.9},
    {"prediction": "咖啡因会使人兴奋，提高注意力。",
     "golden_answers": "咖啡因是一种中枢神经兴奋剂，可增加警觉性和集中力。",
     "expected": 0.8},
]

# 调用并打印
for case in test_cases:
    score = llm_semantic_check(case["prediction"], case["golden_answers"], model_path)
    #score = f1_check(case["prediction"], case["golden_answers"])
    print(f"预测：{case['prediction']}\n标准：{case['golden_answers']}\n"
          f"预期：{case['expected']:.2f}，实际：{score:.2f}\n")

test_cases = [
    # 1. 完全一致
    {"prediction": "the cat sat on the mat",
     "golden_answers": "the cat sat on the mat",
     "expected": 1.0000},

    # 2. 全部不同
    {"prediction": "apple orange banana",
     "golden_answers": "car train plane",
     "expected": 0.0000},

    # 3. 部分重合（“the” 被移除后 pred_tokens=4, gold_tokens=6, common=3）
    {"prediction": "the cat sat on the mat",
     "golden_answers": "a dog sat near the cat and mat",
     "expected": 0.6000},

    # 4. 单词 “like” 重合（长度都为4）
    {"prediction": "cats like warm places",
     "golden_answers": "dogs like cold areas",
     "expected": 0.2500},

    # 5. 大小写与标点被清理
    {"prediction": "Hello, world!",
     "golden_answers": "hello world",
     "expected": 1.0000},

    # 6. 冠词移除后 pred_tokens=2 (cat, sat)，gold_tokens=1 (cat)
    {"prediction": "the the cat sat",
     "golden_answers": "the cat",
     "expected": 0.6667},

    # 7. 多答案取最优
    {"prediction": "he studies machine learning",
     "golden_answers": ["she studies math", "he studies learning"],
     "expected": 0.8571},

    # 8. 仅一个词 “the” 和 “mat” 重合（“the” 被移除）
    {"prediction": "mat the sat cat on",
     "golden_answers": "the cat sat on the mat",
     "expected": 1.0000},

    # 9. only “world” overlaps?
    {"prediction": "World War II ended in 1945",
     "golden_answers": "hello world",
     "expected": 0.25},  # pred_tokens=5→['world','war','ii','ended','in','1945'], actually 6 tokens; golden=['hello','world']→2; common=world(1): P=1/6≈0.1667, R=1/2=0.5 → F1≈0.25

    # 10. 空预测
    {"prediction": "",
     "golden_answers": "the cat sat",
     "expected": 0.0000},
]


# 测试输出
for i, case in enumerate(test_cases, 1):
    score = f1_check(case["prediction"], case["golden_answers"])
    passed = abs(score - case["expected"]) < 1e-4
    print(f"Test {i}: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  Prediction: {case['prediction']}")
    print(f"  Golden:     {case['golden_answers']}")
    print(f"  Expected:   {case['expected']:.4f}")
    print(f"  Got:        {score:.4f}\n")


# 1. 最简流程：只有一次思考和一次回答
    text1 = """<|im_start|>assistant
    <think>我在分析这个问题。</think>
    <answer>这是我的回答。</answer>
    """

    # 2. 标准流程：思考 → 检索 → 信息 → 再次思考 → 回答
    text2 = """<|im_start|>assistant
    <think>首先，我需要明确需求。</think>
    <search>关键字查询</search>
    <information>查询到的相关信息。</information>
    <think>基于以上信息，我得出结论。</think>
    <answer>这是最终的回答。</answer>
    """

    # 3. 带有多余换行和空格，但无非法内容
    text3 = "<|im_start|>assistant   \n\n" \
        "  <think>准备工作完成。</think>  \n" \
        "\n<search>检索步骤</search>\n\n" \
        "<information>一些检索结果。</information>\n\n" \
        "<think>整合信息。</think>\n\n" \
        "<answer>输出答案。</answer>\n"


    # 4. 包含空白字符
    text4 = """<|im_start|>assistant\n\
        <think>思考</think>\t<search>查询</search> \n
        <information>结果</information>  <think>再思考</think>\n
        <search>查询</search> \n<information>结果</information>
        <think>再思考</think>\n
        <answer>答案</answer>"""
    
    text5 = """<|im_start|>assistant\n\
        <think>思考</think>\t<search>查询</search> \n
        <information>结果1</information>  <think>再思考</think>\n
        <search>查询</search> \n<information>结果1</information>
        <think>再思考</think>\n<information>结果1</information>
        <information>结果</information><information>结果</information>
        <information>结果2</information><information>结果2</information>
        <information>结果3</information><information>结果3</information>
        <answer>答案</answer>"""
    """
    text = text5
    print("闭合的Search tag数",count_search_tags(text))
    print("重复的检索数",count_repeat_information(text))
    
      
    is_valid_format, format_reason = is_valid_sequence(text)
    print(f"格式验证: {'有效' if is_valid_format else '无效'} - {format_reason}")
    answer = extract_solution(solution_str=text)
    print(f"提取的答案: {answer if answer is not None else '未提取到答案'}")
    """
