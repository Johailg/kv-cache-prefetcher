"""
kvh.sessions -- turn chat transcripts into token ids plus scoring spans.

WHY THIS IS ITS OWN FILE
------------------------
The obvious implementation is wrong and the wrongness is invisible. Templating
each turn independently

    tok.apply_chat_template([msgs[i]], add_generation_prompt=True)

re-emits the BOS token and the system header on EVERY turn, so a 5-turn session
carries five system headers buried in the middle of it. The model still runs,
the perplexity still comes out finite, and every multi-turn conclusion is drawn
from a transcript no chat model has ever seen.

The fix is to template the GROWING conversation and take the delta each time,
which is what a real serving stack does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Session:
    """ids   [1, T] the whole conversation, in order
       spans list of (turn_index, start, end) -- assistant tokens to score.
             Scoring token j needs the logits produced by feeding token j-1,
             so every span must satisfy start >= 1."""
    ids: torch.Tensor
    spans: List[Tuple[int, int, int]]


def _template(tok, messages, add_generation_prompt):
    """apply_chat_template, with Qwen's thinking block turned off when the
    tokenizer supports it. Returns a list of token ids."""
    try:
        return tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=add_generation_prompt,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=add_generation_prompt)


def build_sessions(path, tok, max_len, min_turns, max_sessions, device):
    """jsonl of {"messages": [{"role": ..., "content": ...}, ...]} -> [Session].

    For turn i the conversation is templated twice:
        prefix = template(msgs[:2i+1], add_generation_prompt=True)
        full   = template(msgs[:2i+2], add_generation_prompt=False)
    The new user segment is prefix[len(previous_full):] and the assistant span
    is full[len(prefix):]. Every token is produced exactly once, in order, by
    the same template the model was trained with.
    """
    sessions = []
    with open(path) as fh:
        for line in fh:
            if len(sessions) >= max_sessions:
                break
            msgs = [m for m in (json.loads(line).get("messages") or [])
                    if m.get("role") in ("user", "assistant")]

            spans, prev_full, turn = [], [], 0
            for i in range(0, len(msgs) - 1, 2):
                if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                    break
                prefix = _template(tok, msgs[:i + 1], True)
                full = _template(tok, msgs[:i + 2], False)

                # sanity: the template must be a strict extension of the last one
                if len(prefix) <= len(prev_full) or len(full) <= len(prefix):
                    break
                if full[:len(prev_full)] != prev_full:
                    break                       # non-monotone template, skip session

                if len(full) > max_len:
                    break
                start, end = len(prefix), len(full)
                if start >= 1:
                    spans.append((turn, start, end))
                prev_full, turn = full, turn + 1

            if turn >= min_turns and spans:
                sessions.append(Session(torch.tensor([prev_full], device=device), spans))
    return sessions

# === Aug 10: Qwen3 chat template is not additive; overrides the above ===
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


