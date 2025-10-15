import transformers
import torch
import random
from datasets import load_dataset
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

import re
def extract_think_and_answer(output_text):
    """
    Extract the content of the <think> and <answer> tags from the model output text
    Parameters:
        output_text: The complete output text of the model
    Return:
        A dictionary containing think_content and answer_content
    """
    think_matches = re.findall(r'<think>(.*?)</think>', output_text, re.DOTALL)
    think_content = [match.strip() for match in think_matches] if think_matches else []
    
    answer_matches = re.findall(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    answer_content = [match.strip() for match in answer_matches] if answer_matches else []
    
    result = {
        "all_think_contents": think_content,
        "all_answer_contents": answer_content,
        "last_think": think_content[-1] if think_content else None,
        "last_answer": answer_content[-1] if answer_content else None
    }
    
    return result

import pandas as pd
import json
def generate_answer(model_id, save_path, dataset_path):
    dataset = []   
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'question' in data:
                dataset.append(data['question'])

    print(f"dataset: {len(dataset)} QA pairs.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        # device_map="auto", 
        device_map=device, 
        local_files_only=True
    )

    qa_data = {}

    for index, question in enumerate(dataset):
        print(f"\nprossing {index+1}/{len(dataset)} QA")
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        curr_eos = [151645, 151643] # for Qwen2.5 series models
        curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        # Prepare the message
        prompt = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        # Initialize the stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])


        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

        # print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        # print(prompt)

        # Encode the chat-formatted prompt and move it to the correct device
        all_outputs = []
        cnt = 0

        while cnt < 5:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate text with the stopping criteria
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            if outputs[0][-1].item() in curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # print(output_text)
                all_outputs.append(output_text)
                break
            
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_outputs.append(output_text)
            
            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                # print(f'searching "{tmp_query}"...')
                search_results = search(tmp_query)
            else:
                search_results = ''

            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1
            # print(f"{cnt} - search_text: {search_text}")

        full_output_text = " ".join(all_outputs)
        result = extract_think_and_answer(full_output_text)
        # print("all_think_contents:", result["all_think_contents"])
        # print("all_answer_contents:", result["all_answer_contents"])
        # print("last_think:", result["last_think"])
        answer = result["last_answer"]
        print(f"question: {question}\nanswer: {answer}")
        qa_data[question] = answer
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    dataset_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/data/qa_data_sample_2k.json'

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/baseline-ppo.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/baseline-ppo-v03.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/nq_search-r1-ppo-qwen2.5-3b-it-em-format-retrieval-0927/actor/global_step_100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda1-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/lambda2-step100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda2-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/lambda3-step100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda3-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/lambda4-step100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda4-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/nq_search-r1-ppo-qwen2.5-3b-it-em-format-retrieval-0917/actor/global_step_100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda5-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)  

    model_id = '/data/home/Yifan/Hallu/Search-R1-phase1-main/verl_checkpoints/nq_search-r1-ppo-qwen2.5-3b-it-em-format-retrieval-0924/actor/global_step_100'
    save_path = '/data/home/Yifan/Hallu/Search-R1-phase1-main/process/HaluEval/results/qa/lambda6-100.json'
    print(model_id)
    print(save_path)
    generate_answer(model_id, save_path=save_path, dataset_path=dataset_path)

# nohup python -u infer_extract_halueval.py > output-1009.log 2>&1 &