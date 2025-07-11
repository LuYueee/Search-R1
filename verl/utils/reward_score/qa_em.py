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
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


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

def compute_score_em(solution_str, ground_truth, structure_format_score=0, final_format_score=0, lambda_task=2, lambda_search_num=0.1, lambda_repeat_search_num=0.1, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)

    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
      
    # 计算base得分
    final_em_format_score = 0
    if answer is None:
        if is_valid_format:
            final_em_format_score = structure_format_score
        else:
            final_em_format_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                final_em_format_score = score
            else:
                final_em_format_score = score - structure_format_score
        elif is_valid_format:
            final_em_format_score = structure_format_score
        else:
            final_em_format_score = final_format_score
    
    # Rewards for redundant retrieval
    n_search = count_search_tags(solution_str)
    n_repeat = count_repeat_information(solution_str)  
    # Apply penalties and calculate final reward
    final_score = lambda_task * final_em_format_score - lambda_search_num * n_search - lambda_repeat_search_num * n_repeat
    
    if do_print:
        print(f"EM Score: {em_check(answer, ground_truth['target'])}")
        print(f"Format Valid: {is_valid_format}")
        print(f"Lambda task: {lambda_task}")
        print(f"Base Reward: {final_em_format_score}")
        print(f"Lambda search num: {lambda_search_num}")
        print(f"Search Count: {n_search} (Penalty: -{lambda_search_num * n_search})")
        print(f"Lambda repeat search num: {lambda_repeat_search_num}")
        print(f"Repeat Count: {n_repeat} (Penalty: -{lambda_repeat_search_num * n_repeat})")
        print(f"Final Reward: {final_score}\n")
    return final_score
