from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
import json
from openai import OpenAI
import re

def calculate_metrics(model_response, correct_references, incorrect_references):
    # BLEU
    model_tokens = model_response.split()
    correct_refs_tokens = correct_references.split()
    incorrect_refs_tokens = incorrect_references.split()

    smooth = SmoothingFunction().method1

    bleu_correct = sentence_bleu(correct_refs_tokens, model_tokens, smoothing_function=smooth)
    bleu_incorrect = sentence_bleu(incorrect_refs_tokens, model_tokens, smoothing_function=smooth)

    bleu_diff = bleu_correct - bleu_incorrect
    bleu_acc = 1 if bleu_correct > bleu_incorrect else 0
    
    # ROUGE
    rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_correct = rouge_calculator.score(correct_references, model_response)['rouge1'].fmeasure
    rouge1_incorrect = rouge_calculator.score(incorrect_references, model_response)['rouge1'].fmeasure
    
    rouge2_correct = rouge_calculator.score(correct_references, model_response)['rouge2'].fmeasure
    rouge2_incorrect = rouge_calculator.score(incorrect_references, model_response)['rouge2'].fmeasure

    rougeL_correct = rouge_calculator.score(correct_references, model_response)['rougeL'].fmeasure
    rougeL_incorrect = rouge_calculator.score(incorrect_references, model_response)['rougeL'].fmeasure

    rouge1_diff = rouge1_correct - rouge1_incorrect
    rouge1_acc = 1 if rouge1_correct > rouge1_incorrect else 0

    rouge2_diff = rouge2_correct - rouge2_incorrect
    rouge2_acc = 1 if rouge2_correct > rouge2_incorrect else 0

    rougeL_diff = rougeL_correct - rougeL_incorrect
    rougeL_acc = 1 if rougeL_correct > rougeL_incorrect else 0

    # BLEURT
    from bleurt import score
    checkpoint = "/data/home/Yifan/Hallu/Search-R1-phase1-main/process/eval/bleurt/bleurt/test_checkpoint"
    scorer = score.BleurtScorer(checkpoint)
    bleurt_correct = scorer.score(references=[correct_references], candidates=[model_response])[0]
    bleurt_incorrect = scorer.score(references=[incorrect_references], candidates=[model_response])[0]

    bleurt_diff = bleurt_correct - bleurt_incorrect
    bleurt_acc = 1 if bleurt_correct > bleurt_incorrect else 0

    # EM
    if model_response == correct_references:
        em = 1
    else:
        em = 0

    return {
        'bleu_correct': bleu_correct,
        'bleu_incorrect': bleu_incorrect,
        'bleu_diff': bleu_diff,
        'bleu_acc': bleu_acc,
        
        'rouge1_correct': rouge1_correct,
        'rouge1_incorrect': rouge1_incorrect, 
        'rouge1_diff': rouge1_diff, 
        'rouge1_acc': rouge1_acc,

        'rouge2_correct': rouge2_correct,
        'rouge2_incorrect': rouge2_incorrect,
        'rouge2_diff': rouge2_diff,
        'rouge2_acc': rouge2_acc,

        'rougeL_correct': rougeL_correct,
        'rougeL_incorrect': rougeL_incorrect,
        'rougeL_diff': rougeL_diff,
        'rougeL_acc': rougeL_acc,

        'bleurt_correct': bleurt_correct,
        'bleurt_incorrect': bleurt_incorrect,
        'bleurt_diff': bleurt_diff,
        'bleurt_acc': bleurt_acc,

        'EM': em
    }


def cal(dataset_path, json_file, output_json_file):
    questions, reference_answers, incorrect_answers = [],[],[]   
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data['question'])
            reference_answers.append(data['right_answer'])
            incorrect_answers.append(data['hallucinated_answer'])

    with open(json_file, 'r', encoding='utf-8') as f:
        qa_results = json.load(f)

    qa_answers_list = list(qa_results.values())
    print(f"processing {len(qa_answers_list)} QA pairs")

    all_results = {}
    all_scores_list = [] 

    for index, question in enumerate(questions):
        print(f"\nprocessing {index+1}/{len(questions)} QA")

        model_response = qa_answers_list[index]

        if model_response is None:
            model_response = ""
        
        print(f"question: {question}")
        print(f"reference_answers: {reference_answers[index]}")
        print(f"incorrect_answers: {incorrect_answers[index]}")
        print(f"model_response: {model_response}")

        scores = calculate_metrics(model_response, reference_answers[index], incorrect_answers[index])
        print(f"scores: {scores}")

        all_results[question] = scores
        all_scores_list.append(scores)
    

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    if all_scores_list:
        scores_df = pd.DataFrame(all_scores_list)
        average_scores = scores_df.mean()

        print("\n--- AVG SCORES ---")
        for metric, avg_score in average_scores.items():
            print(f"{metric}: {avg_score:.4f}")

        # for metric, avg_score in average_scores.items():
        #     print(metric)

        for metric, avg_score in average_scores.items():
            print(f"{avg_score:.4f}")
    else:
        print("ERROR")


def get_llm_response_1(model_response, best_answer):
    client = OpenAI(
        base_url="http://10.98.36.100:8010/v1",
        api_key="qiaoyt",
    )

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
            f"Candidate: {model_response}  \n"
            f"Golden: {best_answer}  \n"
        )
    
    resp = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[{"role":"user","content":prompt}],
        max_tokens=128,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\d{1,3}", raw)
    if not m:
        score100 = 0
    else:
        v = int(m.group())
        score100 = v if 0 <= v <= 100 else 0
        
    semantic_score = score100 / 100.0
    return semantic_score

def get_llm_response_2(model_response, best_answer):
    client = OpenAI(
        base_url="http://10.98.36.100:8010/v1",
        api_key="qiaoyt",
    )

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
        f"Candidate: {model_response}\n"
        f"Golden: {best_answer}"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": user_content},
    ]
    
    resp = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=messages,
        max_tokens=128,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\d{1,3}", raw)
    if not m:
        score100 = 0
    else:
        v = int(m.group())
        score100 = v if 0 <= v <= 100 else 0
        
    semantic_score = score100 / 100.0
    return semantic_score

def get_qwen_response(question, knowledge, right_answer, model_response):
    client = OpenAI(
        base_url="http://10.98.36.100:8010/v1",
        api_key="qiaoyt",
    )

    messages = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge and right answer. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": "#Question#: " + question +
                                    "\n#World knowledge#: " + knowledge +
                                    "\n#Right answer#: " + right_answer +
                                    "\n#Answer#: " + model_response +
                                    "\n#Your Judgement#: "} 
    ]
    response = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=messages,
        max_tokens=128,
    ).choices[0].message.content
    return response

# def get_qwen_response(question, knowledge, right_answer, model_response):
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     model_name = "/data/home/Yifan/Hallu/Search-R1-phase1-main/models/Qwen/Qwen2.5-72B-Instruct"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     messages = [
#         {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge and right answer. The answer you provided MUST be \"Yes\" or \"No\""},
#         {"role": "user", "content": "#Question#: " + question +
#                                     "\n#World knowledge#: " + knowledge +
#                                     "\n#Right answer#: " + right_answer +
#                                     "\n#Answer#: " + model_response +
#                                     "\n#Your Judgement#: "} 
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response

def llm_judge(dataset_path, json_file, output_json_file):
    questions, right_answers, knowledges = [],[],[]   
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data['question'])
            right_answers.append(data['right_answer'])
            knowledges.append(data['knowledge'])

    print(f"processing {len(questions)} QA pairs")

    with open(json_file, 'r', encoding='utf-8') as f:
        qa_results = json.load(f)
    qa_answers_list = list(qa_results.values())
    print(f"processing {len(qa_answers_list)} QA pairs")

    no_halu = 0
    yes_halu = 0
    all_scores_1, all_scores_2 = [], []

    for index, question in enumerate(questions):
        print(f"\nprocessing {index+1}/{len(questions)} QA")

        model_response = qa_answers_list[index]

        if model_response is None:
            print(f"No qa result, skipping...")
            continue
        knowledge = knowledges[index]
        right_answer = right_answers[index]

        print(f"question: {question}")
        print(f"right_answer: {right_answer}")
        print(f"model_response: {model_response}")

        response = get_qwen_response(question, knowledge, right_answer, model_response)
        response = response.strip().lower().replace(".", "")
        print(f"llm judge: {response}")
        if "yes" in response:
            yes_halu += 1
        elif "no" in response:  
            no_halu += 1

        llm_score_1 = get_llm_response_1(model_response, right_answer)
        llm_score_2 = get_llm_response_2(model_response, right_answer)
        all_scores_1.append(llm_score_1)
        all_scores_2.append(llm_score_2)
   
        print(f"llm_score_1: {llm_score_1:.4f}")
        print(f"llm_score_2: {llm_score_2:.4f}")

    avg_score_1 = sum(all_scores_1) / len(all_scores_1) if all_scores_1 else 0
    avg_score_2 = sum(all_scores_2) / len(all_scores_2) if all_scores_2 else 0
    print(f"\n--- AVG LLM SCORE ---\n{avg_score_1:.4f}\n{avg_score_2:.4f}")  

    print(f"no halu rate: {no_halu/len(questions)}")

import os
if __name__ == "__main__":
    data_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/data/qa_data_sample_2k.json'
    json_files = ['lambda1-100', 'lambda2-100', 'lambda3-100', 'lambda4-100', 'lambda5-100', 'lambda6-100', 'baseline-ppo', 'baseline-ppo-v03']

    # data_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/data/qa_data.json'
    # json_files = ['baseline-2k', 'baseline3-2k']
    path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/'

    print(json_files)

    for json_file in json_files:
        output_json_file = path + json_file + '-bleu-rouge.json'
        json_file = path + json_file + '.json'
        print(f"Evaluating {json_file}...")
        print(f"Output will be saved to {output_json_file}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cal(data_path, json_file, output_json_file)
        llm_judge(data_path, json_file, output_json_file)

    # nohup python -u halueval-eval.py > halueval-eval.log 2>&1 &


