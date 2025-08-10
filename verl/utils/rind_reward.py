import torch
import spacy
import re
import gc


ALLOWED_POS = {"NOUN", "ADJ", "VERB", "PROPN", "NUM"}

class RINDCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        space_tokens = tokenizer.tokenize(" ")
        self.space_token = space_tokens[0] if space_tokens else "▁"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.nlp = spacy.load('en_core_web_sm')
        self.method = "dragin"

    def is_content_word(self, token_str):
        doc = self.nlp(token_str)
        if len(doc) == 0:
            return 0
        tok = doc[0]
        if tok.is_stop or tok.text.lower() in self.nlp.Defaults.stop_words:
            return 0
        return 1 if tok.pos_ in ALLOWED_POS else 0

    def compute_rind_from_attn_entropy(self, attn_tensor, generated_tokens_ids, entropies, solver='max'):
        """Compute RIND scores given attention weights and token entropies.

        Args:
            attn_tensor: Tensor of shape [num_heads, L, L]
            generated_tokens_ids: 1D tensor/list of token ids of length L
            entropies: Iterable of precomputed entropy values length L
        """
        gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
        gen_len = len(generated_tokens_ids)
        attn_tensor = attn_tensor.to(torch.float32)
        if torch.is_tensor(entropies):
            entropies = entropies.detach().cpu().tolist()
        else:
            entropies = list(entropies)

        if solver == 'max':
            head_max, _ = torch.max(attn_tensor, dim=1)
            mean_atten = torch.mean(head_max, dim=0)
        elif solver == 'avg':
            head_sum = torch.sum(attn_tensor, dim=1)
            mean_atten = torch.mean(head_sum, dim=0)
            for i in range(gen_len):
                mean_atten[i] /= (gen_len - i)
        elif solver == 'last_token':
            mean_atten = torch.mean(attn_tensor[:, -1], dim=0)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        mean_atten = mean_atten.cpu()
        spans = []
        for i, t in enumerate(gen_tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens_ids[i] == 13 or (
                i > 0 and gen_tokens[i - 1] == '</s>'
            ):
                spans.append([i, i])
            else:
                spans[-1][1] = i

        rind_list = []
        for (start, end) in spans:
            L = end - start + 1
            common_prefixes = {'un', 're', 'in', 'im', 'dis', 'non', 'pre', 'mis', 'sub', 'inter', 'trans'}
            common_suffixes = {'ing', 'ed', 'ly', 'ion', 'able', 'ness', 'ment', 'ful', 'less', 'est', 'ous', 'ive', 's', 'es'}
            word = ''.join(gen_tokens[start:end + 1]).replace(self.space_token, '')
            punct_count = sum(1 for tok in gen_tokens[start:end + 1] if not tok.isalpha() and not tok.isalnum())
            prefix_count = 1 if any(word.lower().startswith(p) for p in common_prefixes) else 0
            suffix_count = 1 if any(word.lower().endswith(s) for s in common_suffixes) else 0
            L_eff = max(1, L - punct_count - prefix_count - suffix_count)

            attn_vals = mean_atten[start:end + 1].tolist()
            attn_sum = sum(attn_vals)
            if attn_sum > 0:
                attn_vals = [v / attn_sum for v in attn_vals]
            else:
                attn_vals = [0.0] * len(attn_vals)
            max_attn = max(attn_vals) if attn_vals else 0.0

            if self.method == 'dragin':
                weight_vals = entropies[start:end + 1]
            else:
                weight_vals = [1.0] * L
            span_ent = sum(weight_vals) / L

            s = self.is_content_word(word)
            rind = max_attn * span_ent * s * L_eff
            pos_tag = self.nlp(word)[0].pos_ if len(self.nlp(word)) > 0 else ""
            rind_list.append((word, rind, max_attn, span_ent, L_eff, pos_tag))

        return rind_list

def compute_sentence_end_rewards(
    rind_calc,
    tokenizer,
    generated_tokens_ids,
    attn_full,
    entropies_full,
    theta=1.2,
    solver='max',
    debug=None,
):
    """Return a list of (token_idx, reward) for each sentence in the sequence.

    If ``debug`` is ``None`` the function will only print debugging information on
    rank 0 when torch.distributed is initialised.  This avoids having to thread
    the rank information through the callers.
    """
    if debug is None:
        if torch.distributed.is_initialized():
            debug = torch.distributed.get_rank() == 0
        else:
            debug = False
    resp_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
    doc = rind_calc.nlp(resp_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

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

    encoding = tokenizer(resp_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    rewards = []

    if torch.is_tensor(attn_full):
        attn_full = attn_full.to(torch.float32)
        attn_full = attn_full / attn_full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if torch.is_tensor(entropies_full):
        entropies_full = entropies_full.to(torch.float32)

    for sent in sentences:
        start_pos = resp_text.find(sent)
        end_pos = start_pos + len(sent)
        if any(f"<{tag}" in sent for tag in ("search", "information", "answer")) or any(s <= start_pos < e for s, e in skip_spans):
            if debug:
                print(f"Skip tagged sentence:\n {sent} -> reward 0")
            continue

        token_idxs = [i for i, (s, e) in enumerate(offsets) if s >= start_pos and e <= end_pos]
        if not token_idxs:
            continue
        start_idx, end_idx = token_idxs[0], token_idxs[-1]
        sub_attn = attn_full[:, start_idx: end_idx + 1, start_idx: end_idx + 1].to('cpu')
        sub_ents = entropies_full[start_idx: end_idx + 1].to('cpu')
        sent_ids = generated_tokens_ids[start_idx: end_idx + 1]
        rind_list = rind_calc.compute_rind_from_attn_entropy(sub_attn, sent_ids, sub_ents, solver=solver)
        M = max((r for _, r, *_ in rind_list), default=0.0)

        tail = resp_text[end_pos:]
        if re.match(r'\s*<search>', tail):
            action = 'SEARCH'
        elif re.match(r'\s*<answer>', tail):
            action = 'ANSWER'
        else:
            action = 'CONTINUE_THINK'

        if M > theta:
            reward = 2 if action == 'SEARCH' else -2
        else:
            reward = 1 if action in ('CONTINUE_THINK', 'ANSWER') else -1

        rewards.append((end_idx, reward))
        if debug:
            print(f"Sentence:\n {sent}\n  MaxRIND={M:.4f}, Action={action}, Reward={reward}")

    del attn_full, entropies_full
    torch.cuda.empty_cache()
    gc.collect()
    return rewards
