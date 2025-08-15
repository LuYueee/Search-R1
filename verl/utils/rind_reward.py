import torch
import spacy
import re
import gc

from verl.utils.torch_functional import entropy_from_logits


ALLOWED_POS = {"NOUN", "ADJ", "VERB", "PROPN", "NUM"}

_WHITESPACE_MARKERS = {
    "\u2581": " ",  # SentencePiece/Metaspace
    "\u0120": " ",  # Byte BPE leading space
    "\u010A": "\n",  # Byte BPE newline
}


def _normalize_piece(piece: str) -> str:
    for k, v in _WHITESPACE_MARKERS.items():
        piece = piece.replace(k, v)
    piece = re.sub(r"[ \t]+", " ", piece)
    return piece if piece.strip() == "" else piece.strip()


def build_offsets_from_ids(tokenizer, ids):
    pieces = tokenizer.convert_ids_to_tokens(ids)
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
        if attn_tensor is None or attn_tensor.numel() == 0 or attn_tensor.shape[1] == 0:
            return []
        gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
        gen_len = len(generated_tokens_ids)
        attn_tensor = attn_tensor.float().cpu()
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
    model,
    tokenizer,
    prompt_tokens_ids,
    generated_tokens_ids,
    theta=1.2,
    solver='max',
    debug=False,
):
    """Return a list of (token_idx, reward) for each sentence in the sequence."""

    resp_text, offsets = build_offsets_from_ids(tokenizer, generated_tokens_ids)
    doc = rind_calc.nlp(resp_text)

    raw_sents = [
        (span.text, span.start_char, span.end_char)
        for span in doc.sents
        if span.text.strip()
    ]

    close_tag_re = re.compile(
        r"^\s*(?:</(?:think|answer|search|information)>)\s*$",
        re.IGNORECASE,
    )
    search_tag_re = re.compile(r"<\s*search\s*>", re.IGNORECASE)
    answer_tag_re = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
    close_tag_prefix_re = re.compile(
        r"^\s*(?:</(?:think|answer|search|information)>)\s*",
        re.IGNORECASE,
    )
    sentences = []
    for text, s, e in raw_sents:
        # peel off any leading closing tags and merge them into the previous sentence
        while True:
            m = close_tag_prefix_re.match(text)
            if not m:
                break
            tag_text = m.group(0)
            tag_end = s + m.end()
            if sentences:
                prev_text, prev_s, prev_e = sentences[-1]
                sentences[-1] = (prev_text + tag_text, prev_s, tag_end)
            else:
                sentences.append((tag_text, s, tag_end))
            text = text[m.end():]
            s = tag_end
        if not text.strip():
            continue
        if close_tag_re.fullmatch(text):
            if sentences:
                prev_text, prev_s, prev_e = sentences[-1]
                sentences[-1] = (prev_text + text, prev_s, e)
            else:
                sentences.append((text, s, e))
        else:
            sentences.append((text, s, e))

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
    L = len(generated_tokens_ids)

    # Step 1: compute attention on the generated tokens only
    token_tensor = torch.tensor(generated_tokens_ids, dtype=torch.long, device=model.device)
    _prev_training_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                attn_out = model(
                    input_ids=token_tensor.unsqueeze(0),
                    attention_mask=torch.ones_like(token_tensor).unsqueeze(0),
                    use_cache=False,
                    output_attentions=True,
                )
    finally:
        model.train(_prev_training_mode)

    attn_full = attn_out.attentions[-1][0].float()
    attn_full = attn_full / attn_full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attn_full = attn_full.cpu()

    del attn_out
    torch.cuda.empty_cache()
    gc.collect()

    # Step 2: teacher-forced forward on full context to compute token entropies
    full_ids = torch.tensor(prompt_tokens_ids + generated_tokens_ids, device=model.device, dtype=torch.long)[None, :]
    attn_mask = torch.ones_like(full_ids, device=model.device)
    resp_len = len(generated_tokens_ids)

    _prev_training_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                out = model(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                    output_attentions=False,
                    num_logits_to_keep=resp_len,
                )
    finally:
        model.train(_prev_training_mode)

    step_logits = out.logits[0].float()
    entropies_full = entropy_from_logits(step_logits).cpu()

    del out, step_logits
    torch.cuda.empty_cache()
    gc.collect()

    for i, (sent_text, start_pos, end_pos) in enumerate(sentences):
        sent = sent_text
        if any(f"<{tag}" in sent for tag in ("search", "information", "answer")) or any(
            s <= start_pos < e for s, e in skip_spans
        ):
            if debug:
                print(f"Skip tagged sentence:\n {sent} -> reward 0")
            continue

        token_idxs = [idx for idx, (s, e) in enumerate(offsets) if 0 <= idx < L and s >= start_pos and e <= end_pos]
        if not token_idxs:
            continue
        start_idx, end_idx = token_idxs[0], token_idxs[-1]
        start_idx = max(0, min(start_idx, L - 1))
        end_idx = max(0, min(end_idx, L - 1))
        if end_idx < start_idx:
            continue
        sub_attn = attn_full[:, start_idx: end_idx + 1, start_idx: end_idx + 1]
        if sub_attn.shape[1] == 0:
            continue
        sub_ents = entropies_full[start_idx: end_idx + 1].tolist()
        sent_ids = generated_tokens_ids[start_idx: end_idx + 1]
        rind_list = rind_calc.compute_rind_from_attn_entropy(sub_attn, sent_ids, sub_ents, solver=solver)
        M = max((r for _, r, *_ in rind_list), default=0.0)

        j = i + 1
        while j < len(sentences) and close_tag_re.fullmatch(sentences[j][0]):
            j += 1

        if j < len(sentences) and search_tag_re.search(sentences[j][0]):
            action = 'SEARCH'
        elif j < len(sentences) and answer_tag_re.search(sentences[j][0]):
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
    return rewards
