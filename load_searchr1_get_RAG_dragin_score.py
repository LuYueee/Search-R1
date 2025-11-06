import transformers
import torch
import random
from datasets import load_dataset
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import spacy
from scipy.special import softmax
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ RIND 计算配置 ------------------
ALLOWED_POS = {'NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'}

# ★ FIX: 引入与代码2一致的空白与标签正则 -----------------------
_WHITESPACE_MARKERS = {
    "\u2581": " ",   # SentencePiece/Metaspace
    "\u0120": " ",   # Byte BPE leading space
    "\u010A": "\n",  # Byte BPE newline
}
CLOSE_TAG_RE = re.compile(
    r"^\s*(?:</(?:think|answer|search|information)>)\s*$",
    re.IGNORECASE,
)
LEADING_CLOSE_RE = re.compile(
    r"^\s*(?:</(?:think|answer|search|information)>\s*)+",
    re.IGNORECASE,
)
TAG_BOUNDARY_RE = re.compile(
    r"(</(?:think|answer|search|information)>)|(<\s*(?:search|answer|information)\s*>)",
    re.IGNORECASE,
)
SEARCH_TAG_RE = re.compile(r"<\s*search\s*>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")
# ---------------------------------------------------

# ★ FIX: 与代码2一致的规范化与offset构建 -------------------------
def _normalize_piece(piece: str) -> str:
    for k, v in _WHITESPACE_MARKERS.items():
        piece = piece.replace(k, v)
    piece = re.sub(r"[ \t]+", " ", piece)
    return piece if piece.strip() == "" else piece.strip()

def build_offsets_from_ids(tokenizer, ids):
    """从生成的 token ids 反推文本与字符偏移，避免重分词错位。"""
    if ids is None:
        return "", []
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # 过滤非法id
    ids = [int(t) for t in ids if t is not None and t != -100]
    if not ids:
        return "", []

    pieces = tokenizer.convert_ids_to_tokens(ids) or []
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        return "", []

    resp_text = tokenizer.convert_tokens_to_string(pieces)
    offsets = []
    cursor = 0
    for tok in pieces:
        seg = tokenizer.convert_tokens_to_string([tok])
        seg = _normalize_piece(seg)
        idx = resp_text.find(seg, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = start + len(seg)
        offsets.append((start, end))
        cursor = end
    return resp_text, offsets
# ---------------------------------------------------

class RINDCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        
        # 子词合并时用到的空格标记
        if getattr(self.config, 'model_type', '') == 'llama':
            self.space_token = '▁'
        else:
            space_tokens = tokenizer.tokenize(' ')
            self.space_token = space_tokens[0] if space_tokens else " "

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # spaCy 用于 POS 判断
        self.nlp = spacy.load('en_core_web_sm')
        self.content_pos = ALLOWED_POS
        self.method = "dragin"  # 或 "attn_prob"

    def is_content_word(self, token_str):
        """语义指示器 s_i：非停用词且属于指定 POS 列表时返回 1，否则返回 0"""
        doc = self.nlp(token_str)
        if len(doc) == 0:
            return 0
        tok = doc[0]
        if tok.is_stop or tok.text.lower() in self.nlp.Defaults.stop_words:
            return 0
        return 1 if tok.pos_ in ALLOWED_POS else 0

    def compute_rind_for_generation(self, generation_outputs, generated_tokens_ids, solver='max'):
        """
        计算生成文本的RIND得分
        generation_outputs: 生成过程的输出（包含scores）
        generated_tokens_ids: 生成的token ID序列
        """
        # 1. 提取生成token序列
        gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
        gen_len = len(generated_tokens_ids)
        
        # 2. 复用生成过程的scores计算熵
        scores = generation_outputs.scores  # 元组，每个元素是(1, vocab_size)的tensor
        all_logits = torch.stack(scores, dim=1).squeeze(0).cpu().numpy()  # (gen_len, vocab_size)
        
        entropies = []
        for i in range(gen_len):
            probs = softmax(all_logits[i])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        # 3. 单独前向传播计算注意力（仅针对生成部分）
        input_ids = generated_tokens_ids.unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        attentions = outputs.attentions
        
        # 4. 获取最后一层注意力
        last_layer_attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        seq_len = last_layer_attn.shape[1]
        
        # 5. 聚合注意力
        if solver == "max":
            head_max, _ = torch.max(last_layer_attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_max, dim=0)  # [seq_len]
        elif solver == "avg":
            head_sum = torch.sum(last_layer_attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_sum, dim=0)  # [seq_len]     
            for i in range(seq_len):
                mean_atten[i] /= (seq_len - i)
        elif solver == "last_token":
            mean_atten = torch.mean(last_layer_attn[:, -1], dim=0)  # [seq_len]
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        # 6. 子词合并
        spans = []
        for i, t in enumerate(gen_tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens_ids[i] == 13 or (i > 0 and gen_tokens[i-1] == '</s>'):
                spans.append([i, i])
            else:
                spans[-1][1] = i
        
        # 7. 计算每个span的RIND
        rind_list = []
        for (start, end) in spans:
            L = end - start + 1
            
            common_prefixes = {'un', 're', 'in', 'im', 'dis', 'non', 'pre', 'mis', 'sub', 'inter', 'trans'}
            common_suffixes = {'ing', 'ed', 'ly', 'ion', 'able', 'ness', 'ment', 'ful', 'less', 'est', 'ous', 'ive', 's', 'es'}

            word = ''.join(gen_tokens[start:end+1]).replace(self.space_token, '')
            punct_count = sum(1 for tok in gen_tokens[start:end+1] if not tok.isalpha() and not tok.isalnum())
            prefix_count = 1 if any(word.lower().startswith(p) for p in common_prefixes) else 0
            suffix_count = 1 if any(word.lower().endswith(s) for s in common_suffixes) else 0
            L_eff = max(1, L - punct_count - prefix_count - suffix_count)
            
            attn_vals = mean_atten[start:end+1].tolist()
            attn_sum = sum(attn_vals)
            if attn_sum > 0:
                attn_vals = [v / attn_sum for v in attn_vals]
            else:
                attn_vals = [0.0] * len(attn_vals)
            max_attn = max(attn_vals) if attn_vals else 0.0
            
            if self.method == "dragin":
                weight_vals = entropies[start:end+1]
            else:
                weight_vals = [1.0] * L
            span_ent = sum(weight_vals) / L
            
            s = self.is_content_word(word)
            rind = max_attn * span_ent * s * L_eff
            
            pos_tag = self.nlp(word)[0].pos_ if len(self.nlp(word)) > 0 else ""
            rind_list.append((word, rind, max_attn, span_ent, L_eff, pos_tag))
            
        return rind_list

# ------------------ 主代码 ------------------
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
question = "Who was Ph.D. advisor of Yue Lu from UCR?"
#question = "Which Chinese city held the Olympic Games?"
#question = "Which city is the capital of the United Kingdom?"

# 模型路径
model_id = "/home/jovyan/work_vol90/RL+RAG/Search-R1-main/verl_checkpoints/nq_search-r1-ppo-qwen2.5-3b-it-em-format-retrieval/actor/global_step_100"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# 准备消息
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# 初始化tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto", 
    local_files_only=True
)

# 初始化RIND计算器
rind_calculator = RINDCalculator(model, tokenizer)

# 定义自定义停止条件
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
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

# 初始化停止条件
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0
full_context = prompt  # 保存完整上下文用于日志等

if tokenizer.chat_template:
    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    prompt = formatted_prompt

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)

# 主生成循环
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # 生成文本（启用output_scores以获取logits）
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 计算并打印当前生成的RIND得分（使用新的计算方法）
    rind_scores = rind_calculator.compute_rind_for_generation(
        outputs, 
        generated_tokens,
        solver='max'
    )
    
    print(f"\n=== Generated Text ===")
    print(output_text)
    
    print(f"\n=== RIND Scores ===")
    print(f"{'Word':<15}{'RIND':<10}{'MaxAttn':<10}{'AvgEnt':<10}{'EffLen':<10}{'POS':<8}")
    for word, rind, attn, ent, tok_num, pos in rind_scores:
        print(f"{word:<15}{rind:<10.4f}{attn:<10.4f}{ent:<10.4f}{tok_num:<10}{pos}")
    print("="*50 + "\n")
    
    ##### 奖励计算（严格对齐代码2） ################################
    THETA = 1.2

    # ★ FIX: 使用 ids -> 文本 与 offsets 的单一来源，避免错位
    resp_text, offsets = build_offsets_from_ids(tokenizer, generated_tokens)

    # ★ FIX: 按标签边界细分句子段（而非简单字符串分句）
    doc = rind_calculator.nlp(resp_text)
    raw_sents = [
        (span.text, span.start_char, span.end_char)
        for span in doc.sents
        if span.text.strip()
    ]

    # ★ FIX: 在每个句子内部按 TAG_BOUNDARY_RE 切分，并处理句首连续关闭标签
    sentences = []
    for text, s, e in raw_sents:
        parts = []
        cur = 0
        has_match = False
        for m in TAG_BOUNDARY_RE.finditer(text):
            has_match = True
            if m.start() > cur:
                parts.append((text[cur:m.start()], s + cur, s + m.start()))
            parts.append((m.group(0), s + m.start(), s + m.end()))
            cur = m.end()
        if cur < len(text):
            parts.append((text[cur:], s + cur, e))

        if not has_match:
            m = LEADING_CLOSE_RE.match(text)
            if m:
                close_part = text[: m.end()]
                close_end = s + len(close_part)
                if sentences:
                    prev_text, prev_s, prev_e = sentences[-1]
                    sentences[-1] = (prev_text + close_part, prev_s, close_end)
                else:
                    sentences.append((close_part, s, close_end))
                if m.end() < len(text):
                    sentences.append((text[m.end():], close_end, e))
                continue

        for seg_text, seg_s, seg_e in parts:
            if not seg_text.strip():
                continue
            if CLOSE_TAG_RE.fullmatch(seg_text):
                if sentences:
                    prev_text, prev_s, prev_e = sentences[-1]
                    sentences[-1] = (prev_text + seg_text, prev_s, seg_e)
                else:
                    sentences.append((seg_text, seg_s, seg_e))
            else:
                sentences.append((seg_text, seg_s, seg_e))

    # ★ FIX: 仅把 <search>/<information>/<answer> 的成对区块做整体跳过
    skip_spans = []
    for tag in ("search", "information", "answer"):
        for m in re.finditer(fr"<{tag}>(.*?)</{tag}>", resp_text, re.DOTALL):
            skip_spans.append((m.start(), m.end()))
    skip_spans.sort()
    merged = []
    for s, e in skip_spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    skip_spans = merged

    rewards = []
    L = len(generated_tokens)

    # ★ FIX: 遍历段落（非粗糙“整句”），并用 offsets → token 范围映射
    for i, (sent_text, start_pos, end_pos) in enumerate(sentences):
        sent = sent_text

        # 跳过：三类标签区块内/含三类标签自身/终止标记
        if (any(f"<{tag}" in sent for tag in ("search", "information", "answer"))
            or any(s <= start_pos < e for s, e in skip_spans)
            or any(marker in sent for marker in TERMINAL_MARKERS)):
            # print(f"Skip tagged sentence:\n {sent} -> reward 0")
            continue

        token_idxs = [idx for idx, (s, e) in enumerate(offsets) if 0 <= idx < L and s >= start_pos and e <= end_pos]
        if not token_idxs:
            # print(f"No tokens for sentence: {sent} -> reward 0")
            continue

        start_idx, end_idx = token_idxs[0], token_idxs[-1]
        start_idx = max(0, min(start_idx, L - 1))
        end_idx = max(0, min(end_idx, L - 1))
        if end_idx < start_idx:
            continue

        # 取出对应的 token_ids 和 scores（保持你原来的MiniOut调用方式）
        sent_tok_ids = generated_tokens[start_idx: end_idx + 1]
        mini_scores   = outputs.scores[start_idx: end_idx + 1]

        class MiniOut:
            def __init__(self, scores): self.scores = tuple(scores)
        mini_out = MiniOut(mini_scores)

        # 计算 RIND 并求 M
        rind_list = rind_calculator.compute_rind_for_generation(
            mini_out,
            sent_tok_ids,
            solver='max'
        )
        M = max((r for _, r, *_ in rind_list), default=0.0)

        # ★ FIX: 动作判定前跨过所有纯关闭标签段
        j = i + 1
        while j < len(sentences) and CLOSE_TAG_RE.fullmatch(sentences[j][0]):
            j += 1

        if j < len(sentences) and SEARCH_TAG_RE.search(sentences[j][0]):
            action = "SEARCH"
        elif j < len(sentences) and ANSWER_TAG_RE.search(sentences[j][0]):
            action = "ANSWER"
        else:
            action = "CONTINUE_THINK"

        # 奖励
        if M > THETA:
            reward = +2 if action == "SEARCH" else -2
        else:
            reward = +1 if action in ("CONTINUE_THINK", "ANSWER") else -1

        print(f"Sentence:\n {sent}")
        print(f"  MaxRIND = {M:.4f}, Action = {action}, Reward = {reward}")
        print("-" * 40)
        rewards.append((end_idx, reward))
    ###############################################################

    # 更新完整上下文（用于日志等）
    full_context += output_text
    
    if outputs.sequences[0][-1].item() in curr_eos:
        break

    tmp_query = get_query(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
    if tmp_query:
        search_results = search(tmp_query)
    else:
        search_results = ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
    full_context += search_text  # 更新完整上下文
    cnt += 1
    print(search_text)
