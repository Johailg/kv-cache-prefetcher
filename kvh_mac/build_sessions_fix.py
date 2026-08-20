"""Drop-in replacement for sessions.build_sessions.

WHY: the old version templated the conversation once per turn and required each
rendering to be a strict extension of the previous one. That assumption is false
for Qwen3: msgs[:2] renders the assistant header followed by an empty
`<think>\\n\\n</think>\\n\\n` block, but re-rendering that same turn inside
msgs[:4] drops the block. The monotone guard then correctly refuses every
session and build_sessions returns [].

FIX: render the whole conversation ONCE, then locate each assistant reply by
character offset inside that single rendering. No additivity assumption, and
every token is still produced exactly once, in order, by the model's own
template -- which is what the original docstring claimed.

Paste build_sessions() over the old one in sessions.py. Run this file directly
to self-check first:

    python build_sessions_fix.py --model Qwen/Qwen3-0.6B \\
        --sessions /projects/bcjw/jgerard1/kvcache/smoke_sessions.jsonl
"""

import json

import torch


# Set to True to keep Qwen3's empty <think></think> block in assistant turns.
# Qwen3 non-thinking mode really does serve that block at inference, so False
# scores on a token sequence slightly different from production. Whichever you
# pick, keep it fixed across every run and write the choice down.
ENABLE_THINKING = False


def _render(tok, messages, add_generation_prompt=False):
    """Template to TEXT (not ids). Returns the rendered string."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=ENABLE_THINKING)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt)


def _char_to_token_spans(offsets, char_start, char_end):
    """Half-open token span [start, end) covering [char_start, char_end)."""
    start = end = None
    for i, (a, b) in enumerate(offsets):
        if a == b:                      # special tokens carry an empty offset
            continue
        if start is None and b > char_start:
            start = i
        if a < char_end:
            end = i + 1
    return start, end


def build_sessions(path, tok, max_len, min_turns, max_sessions, device):
    """jsonl of {"messages": [{"role": ..., "content": ...}, ...]} -> [Session].

    The conversation is rendered once. Each assistant reply is located by
    searching the rendered text forward from a cursor, so repeated content
    cannot match an earlier turn. Character spans are converted to token spans
    via the tokenizer's offset mapping.
    """
    from sessions import Session          # same class the rest of kvh expects

    sessions = []
    with open(path) as fh:
        for line in fh:
            if len(sessions) >= max_sessions:
                break

            msgs = [m for m in (json.loads(line).get("messages") or [])
                    if m.get("role") in ("user", "assistant")]

            # keep only complete user/assistant pairs, in order
            pairs = []
            for i in range(0, len(msgs) - 1, 2):
                if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                    break
                pairs.append((msgs[i], msgs[i + 1]))
            if not pairs:
                continue

            flat = [m for p in pairs for m in p]
            text = _render(tok, flat, add_generation_prompt=False)

            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]

            if len(ids) > max_len:
                # trim whole turns from the end until it fits
                while len(pairs) > 1:
                    pairs = pairs[:-1]
                    flat = [m for p in pairs for m in p]
                    text = _render(tok, flat, add_generation_prompt=False)
                    enc = tok(text, add_special_tokens=False,
                              return_offsets_mapping=True)
                    ids, offsets = enc["input_ids"], enc["offset_mapping"]
                    if len(ids) <= max_len:
                        break
                if len(ids) > max_len:
                    continue

            spans, cursor, turn, ok = [], 0, 0, True
            for _, assistant in pairs:
                content = assistant["content"]
                cs = text.find(content, cursor)
                if cs < 0:
                    ok = False           # template altered the content itself
                    break
                ce = cs + len(content)
                cursor = ce

                start, end = _char_to_token_spans(offsets, cs, ce)
                if start is None or end is None or end <= start or start < 1:
                    ok = False
                    break
                spans.append((turn, start, end))
                turn += 1

            if ok and turn >= min_turns and spans:
                sessions.append(
                    Session(torch.tensor([ids], device=device), spans))

    return sessions


# ---------------------------------------------------------------- self-check

def _self_check(model, path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    with open(path) as fh:
        line = fh.readline()
    msgs = [m for m in json.loads(line)["messages"]
            if m["role"] in ("user", "assistant")]

    text = _render(tok, msgs, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    # 1. re-tokenizing the rendered text must match tokenize=True exactly,
    #    otherwise the offset trick is measuring a different sequence
    try:
        direct = tok.apply_chat_template(msgs, tokenize=True,
                                         add_generation_prompt=False,
                                         enable_thinking=ENABLE_THINKING)
    except TypeError:
        direct = tok.apply_chat_template(msgs, tokenize=True,
                                         add_generation_prompt=False)
    print(f"tokenize=True len {len(direct)}   re-tokenized len {len(ids)}   "
          f"identical: {list(direct) == list(ids)}")

    # 2. every assistant span must decode back to its content
    cursor, n = 0, 0
    for i in range(1, len(msgs), 2):
        content = msgs[i]["content"]
        cs = text.find(content, cursor)
        ce = cs + len(content)
        cursor = ce
        s, e = _char_to_token_spans(offsets, cs, ce)
        got = tok.decode(ids[s:e]).strip()
        match = got == content.strip()
        print(f"  turn {n}: tokens [{s},{e})  roundtrip {'OK' if match else 'MISMATCH'}")
        if not match:
            print(f"    want: {content[:70]!r}")
            print(f"    got : {got[:70]!r}")
        n += 1

    # 3. spans must not overlap and must be increasing
    print("spans non-overlapping and ordered: checked above by construction")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--sessions", required=True)
    a = p.parse_args()
    _self_check(a.model, a.sessions)
