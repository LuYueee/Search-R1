import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import spacy
from scipy.special import softmax
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ 配置 ------------------
MODEL_PATH = "/home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/qwen2.5-7b-instruct-1m"
ALLOWED_POS = {'NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'}


class BasicGeneratorRIND:
    # 类变量，用于单例加载
    _tokenizer = None
    _model = None
    _config = None

    def __init__(self, model_path):
        # 仅在第一次实例化时加载
        if BasicGeneratorRIND._tokenizer is None:
            logger.info("Loading tokenizer and model for the first time...")
            BasicGeneratorRIND._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            BasicGeneratorRIND._config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code="falcon" in model_path
            )
            BasicGeneratorRIND._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code="falcon" in model_path,
                output_attentions=True
            )
            BasicGeneratorRIND._model.eval()

        # 将单例对象赋给实例属性
        self.tokenizer = BasicGeneratorRIND._tokenizer
        self.model = BasicGeneratorRIND._model
        config = BasicGeneratorRIND._config

        # 子词合并时用到的空格标记
        if getattr(config, 'model_type', '') == 'llama':
            self.space_token = '▁'
        else:
            space_tokens = self.tokenizer.tokenize(' ')
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

    def compute_rind(self, input_text: str, max_new_tokens: int = 20, solver: str = 'max'):
        # 1. 编码输入
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.model.device)
        input_len = input_ids.shape[1]
        
        # 2. 生成
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True
        )
        
        # 3. 提取生成的token
        generated_ids = outputs.sequences[:, input_len:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        text = self.tokenizer.decode(generated_ids[0])
        
        # 4. 子词合并 (与权威代码一致)
        spans = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_ids[0][i] == 13 or tokens[i-1] == '</s>':
                spans.append([i, i])
            else:
                spans[-1][1] = i
        
        # 5. 获取最后一层注意力 (与权威代码一致)
        # 注意：权威代码使用生成的token作为输入重新计算注意力
        attn_outputs = self.model(generated_ids, output_attentions=True)
        attn = attn_outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        
        # 6. 聚合注意力 (与权威代码一致)
        if solver == "max":
            # 每个头取最大 -> 头维度取平均
            head_max, _ = torch.max(attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_max, dim=0)  # [seq_len]
        elif solver == "avg":
            # 每个头求和 -> 头维度取平均
            head_sum = torch.sum(attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_sum, dim=0)  # [seq_len]     
            # 权威代码的特殊归一化
            seq_len = mean_atten.shape[0]
            for i in range(seq_len):
                mean_atten[i] /= (seq_len - i)
        elif solver == "last_token":
            # 只取最后一个token的注意力
            mean_atten = torch.mean(attn[:, -1], dim=0)  # [seq_len]
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        # 7. 处理特殊token的归一化 (与权威代码一致)
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            sum_after = torch.sum(mean_atten[1:])
            if sum_after > 0:
                mean_atten = mean_atten / sum_after
        
        # 8.1 计算熵 (与权威代码一致)
        # 收集所有logits
        logits_list = [score.cpu().numpy() for score in outputs.scores]
        
        # 使用scipy的softmax计算概率分布
        probs_list = softmax(logits_list, axis=-1)
        
        # 计算每个位置的熵
        entropies = []
        logprobs = []
        if self.method == "dragin":
            for probs in probs_list:
                # 添加epsilon防止log(0)
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                entropies.append(entropy[0])  # 取batch中第一个样本
        elif self.method == "attn_prob":
            # 8.2 计算对数概率（用于attn_prob方法）
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = [score.item() for score in transition_scores[0].cpu()]
        
        # 9. 选择权重计算方式
        weights = []
        if self.method == "dragin":
            weights = entropies
        elif self.method == "attn_prob":
            weights = [-lp for lp in logprobs]  # 负对数概率
        
        # 10. 重新计算RIND得分（与代码1一致）
        rind_list = []
        
        for (start, end) in spans:
            # 1. span 长度
            L = end - start + 1
            
            ########
            # 新增：定义高频前缀和后缀
            common_prefixes = {'un', 're', 'in', 'im', 'dis', 'non', 'pre', 'mis', 'sub', 'inter', 'trans'}
            common_suffixes = {'ing', 'ed', 'ly', 'ion', 'able', 'ness', 'ment', 'ful', 'less', 'est', 'ous', 'ive', 's', 'es'}

            #### 重新构造当前span对应的单词
            word = ''.join(tokens[start:end+1]).replace(self.space_token, '')

            # 新增：标点计数（如单独 token 为 "," "." "!" 等）
            punct_count = sum(1 for tok in tokens[start:end+1] if tok.isalpha() is False and tok.isalnum() is False)

            # 新增：前缀/后缀计数（是否以常见前缀或后缀开头/结尾）
            prefix_count = 1 if any(word.lower().startswith(p) for p in common_prefixes) else 0
            suffix_count = 1 if any(word.lower().endswith(s) for s in common_suffixes) else 0

            # 重新定义有效长度 L_eff
            L_eff = max(1, L - punct_count - prefix_count - suffix_count)  # 至少为1防止除0
            #######
            
            
            # 2. attention 列表
            attn_vals = mean_atten[start:end+1].tolist()
            # 对span内部的attention做归一化
            attn_sum = sum(attn_vals)
            if attn_sum > 0:
                attn_vals = [v / attn_sum for v in attn_vals]
            else:
                attn_vals = [0.0] * len(attn_vals)  # 防止除以0

            # 计算 max attentions
            max_attn = max(attn_vals)

            
            # 3. weight 列表（熵或负 logprob）
            if self.method == "dragin":
                weight_vals = entropies[start:end+1]
            else:
                weight_vals = [-lp for lp in logprobs[start:end+1]]
            # 计算 span 平均熵/平均对数概率
            span_ent = sum(weight_vals) / L
            
            
            # 4. 语义指示器 s_i（内容词检查）
            s = self.is_content_word(word)

            # 6. 最终 RIND 
            # rind = max_attn * span_ent * s
            ### 或者乘上 L_eff，放大多子词的影响
            rind = max_attn * span_ent * s * L_eff
            
            
            # 7. 记录：原始 word、RIND、max_attn、avg_ent、POS（可选）
            pos_tag = self.nlp(word)[0].pos_ if len(self.nlp(word)) > 0 else ""
            rind_list.append((word, rind, max_attn, span_ent, L_eff, pos_tag))
            

            '''
            # 2. attention 列表
            attn_vals = mean_atten[start:end+1].tolist()
            # 对span内部的attention做归一化
            attn_sum = sum(attn_vals)
            if attn_sum > 0:
                attn_vals = [v / attn_sum for v in attn_vals]
            else:
                attn_vals = [0.0] * len(attn_vals)  # 防止除以0

            # 3. weight 列表（熵或负 logprob）
            if self.method == "dragin":
                weight_vals = entropies[start:end+1]
            else:
                weight_vals = [-lp for lp in logprobs[start:end+1]]

            # 4. 原始：per-token value = attn * weight * span_length
            value_list = [attn_vals[i] * weight_vals[i] * L_eff for i in range(L)]
            ####
            # 新版：移除 * L 缩放，仅保留 attn * weight
            # value_list = [attn_vals[i] * weight_vals[i] for i in range(L)]
            ####

            
            # 5. 语义指示器 s_i（内容词检查）
            s = self.is_content_word(word)

            # 6. 最终 RIND = max(value_list) * s * L
            rind = max(value_list) * s
            ###
            # 新版：移除 L 缩放，看子词的平均情况
            # rind = sum(value_list) / L * s 
            ###
            
            # 7. 记录：原始 word、RIND、max_attn、avg_ent、POS（可选）
            span_ent = sum(weight_vals) / L
            pos_tag = self.nlp(word)[0].pos_ if len(self.nlp(word)) > 0 else ""
            rind_list.append((word, rind, max(attn_vals), span_ent, L_eff, pos_tag))
            '''
        return text, rind_list


if __name__ == '__main__':
    # 不管调用多少次，每次都是同一个 model/tokenizer
    gen = BasicGeneratorRIND(MODEL_PATH)

    out_text, rind_scores = gen.compute_rind(
        "Dr.Yue Lu was a Ph.D. student from UCR, her advisor",
        max_new_tokens=20,
        solver='max'
    )
    print("Generated:", out_text)
    print(f"{'Word':<12}{'RIND':<8}{'Attn':<8}{'Ent':<8}{'Tok#':<8}{'POS'}")
    for word, rind, attn, ent, tok_num, pos in rind_scores:
        print(f"{word:<12}{rind:<8.3f}{attn:<8.3f}{ent:<8.3f}{tok_num:<8.3f}{pos}")

    # 再次调用，不会重复加载模型
    out_text2, rind_scores2 = gen.compute_rind(
        "In ancient times, ships were powered by sails.",
        max_new_tokens=20,
        solver='max'
    )
    print("Generated:", out_text2)
    print(f"{'Word':<12}{'RIND':<8}{'Attn':<8}{'Ent':<8}{'Tok#':<8}{'POS'}")
    for word, rind, attn, ent, tok_num, pos in rind_scores2:
        print(f"{word:<12}{rind:<8.3f}{attn:<8.3f}{ent:<8.3f}{tok_num:<8.3f}{pos}")
