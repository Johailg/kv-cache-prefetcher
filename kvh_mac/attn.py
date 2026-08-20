"""
kvh.attn -- policy-masked attention.

Read kvh/shapes.py first.

WHAT THIS DOES
--------------
Nothing is ever evicted. We compute the full attention scores, set the columns
the policy calls non-resident to -inf, and renormalise. The logits are
identical to what a genuinely evicted cache would produce, and the true
distribution is available in the same pass -- which is what lets one run report
end-to-end quality AND the escape-conditioned numbers from the offline ladder,
on the same rows.

The cost: no memory saved, no time saved. Right trade for a quality study,
wrong one for a latency claim. Keep those as separate artifacts.

THE MASK IS CONSTANT ACROSS QUERY ROWS WITHIN ONE CALL
------------------------------------------------------
This is the semantics, not a shortcut. A request arrives; the cache is in
whatever compressed state the previous step left it in; the whole prefill sees
that one state. Decode is the Q == 1 case, where selection therefore runs once
per token.

Consequence worth knowing: during a long prefill the policy cannot re-select
mid-prompt. That matches how H2O and SnapKV actually behave (compress after
prefill, not during) but it does mean a cluster policy gets one decision for
the whole user turn.

THE OBSERVATION WINDOW
----------------------
A policy that sets `obs_rows = N` gets an extra field on Obs:
col_mass_resident_lastw, the masked column mass summed over only the LAST N
query rows of the call. Only SnapKV asks for it, and it is the difference
between implementing SnapKV and implementing a position ranking: summed over
all Q causal rows, column j receives Q-j contributions, so the score inherits
a monotone recency penalty. Every one of the last N rows sees the entire
prefix, so restricting to them removes the bias exactly.

Cost is one [Hkv, N, K] float32 buffer, kept as a rolling window across query
chunks so it never depends on chunk size.

ORDER OF OPERATIONS PER CALL
----------------------------
    1. policy.select(...)        <- must see only keys up to c[t-1]
    2. policy.note_keys(...)     <- now the new keys exist
    3. compute masked attention
    4. escape readout
    5. policy.observe(...)

Swapping 1 and 2 shifts the state variable by one step and silently changes
what the flow graph is being asked to predict.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Works both as a package (`python -m kvh.attn`) and as a plain script
# (`python attn.py` from inside the directory). The relative form is tried
# first so the package layout stays canonical.
try:
    from .policies import ClusterIndex, Obs
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from policies import ClusterIndex, Obs

ESCAPE_THRESHOLD = 0.25          # mt_common.THRESH


# ==========================================================================
# accounting
# ==========================================================================

@dataclass
class Accounting:
    """Running totals for one sequence. Every field is a sum; the properties
    do the division, so partial results are always well defined.

    Residency and recovery are accumulated per (layer, kv_head, call).
    Escape statistics are accumulated per (layer, kv_head, DECODE step) only --
    prefill calls have Q > 1 and no single mass row to test.
    """
    resident_sum: float = 0.0        # sum of resident column counts
    column_sum: float = 0.0          # sum of total column counts
    call_count: int = 0

    # per-KV-head totals. Token policies keep the SAME COUNT on every head
    # (one scalar budget, topk per row) so this is flat for them. Cluster
    # policies do not: each head has its own centroids, its own hub and its own
    # assignment of the same physical positions, so k=4 covers a different
    # number of tokens on every head. Observed spreads of 3x are normal.
    # The mean is the right number for total bytes; the spread is what tells
    # you whether one head is carrying the compression.
    head_resident: Optional[torch.Tensor] = None     # [Hkv] float64
    head_columns: Optional[torch.Tensor] = None      # [Hkv] float64

    recovered_sum: float = 0.0       # sum over query rows of raw mass kept
    recovered_rows: float = 0.0

    step_count: float = 0.0          # QUERY-head-steps examined for escapes
    escape_count: float = 0.0        # QUERY-head-steps that escaped
    escape_recovered_sum: float = 0.0
    escape_hits: float = 0.0         # cstar landed in the chosen cluster set
    escape_scored: float = 0.0       # head-steps where a cluster set existed

    per_layer: Dict[int, List[float]] = field(default_factory=dict)

    def add_call(self, layer, n_resident, n_columns, recovered_sum, recovered_rows,
                 per_head_resident=None, per_head_columns=None):
        if per_head_resident is not None:
            if self.head_resident is None:
                self.head_resident = torch.zeros_like(per_head_resident,
                                                      dtype=torch.float64)
                self.head_columns = torch.zeros_like(per_head_resident,
                                                     dtype=torch.float64)
            self.head_resident += per_head_resident.to(torch.float64)
            self.head_columns += per_head_columns.to(torch.float64)
        self.resident_sum += n_resident
        self.column_sum += n_columns
        self.recovered_sum += recovered_sum
        self.recovered_rows += recovered_rows
        self.call_count += 1
        slot = self.per_layer.setdefault(layer, [0.0, 0.0])
        slot[0] += n_resident
        slot[1] += n_columns

    def add_escape_step(self, n_head_steps, n_escapes, recovered, hits, scored):
        self.step_count += n_head_steps
        self.escape_count += n_escapes
        self.escape_recovered_sum += recovered
        self.escape_hits += hits
        self.escape_scored += scored

    # -- derived ----------------------------------------------------------

    @property
    def residency(self) -> float:
        """Fraction of KV entries resident, averaged over calls and heads."""
        return self.resident_sum / max(self.column_sum, 1e-9)

    @property
    def residency_per_head(self) -> Optional[torch.Tensor]:
        """[Hkv] resident fraction for each KV head, or None if untracked."""
        if self.head_resident is None:
            return None
        return self.head_resident / self.head_columns.clamp_min(1e-9)

    @property
    def residency_spread(self) -> float:
        """max - min residency across KV heads. 0 for token policies by
        construction; nonzero for cluster policies always."""
        per = self.residency_per_head
        if per is None:
            return float("nan")
        return float(per.max() - per.min())

    @property
    def recovery(self) -> float:
        """Raw softmax mass landing on the resident set, all steps.
        NOTE: computed on p_true, column 0 INCLUDED. Different from
        recovery_esc below. See shapes.py."""
        return self.recovered_sum / max(self.recovered_rows, 1e-9)

    @property
    def escape_rate(self) -> float:
        return self.escape_count / max(self.step_count, 1e-9)

    @property
    def recovery_esc(self) -> float:
        """mass_nosink landing on the resident set, ESCAPE steps only.
        The discriminating column in mt_masscurve. Computed on the
        column-0-zeroed renormalised row, matching the escape event."""
        return self.escape_recovered_sum / max(self.escape_count, 1e-9)

    @property
    def h_esc(self) -> float:
        """Binary escape hit rate. nan for token policies, which have no
        cluster set to test cstar against."""
        if self.escape_scored == 0:
            return float("nan")
        return self.escape_hits / self.escape_scored

    def reset(self):
        self.__init__()


# ==========================================================================
# escape readout -- deliberately policy-independent
# ==========================================================================

class EscapeReadout:
    """Applies mt_common.escape_event to every decode step, for ANY policy.

    Maintains its own cluster assignment of the cache under a MEASUREMENT
    lineage, separate from whatever lineage a cluster policy is using. That
    separation is the point: it lets h2o and snapkv be scored on the offline
    ladder's own axis, which no run has ever done.

    ONE EVENT PER QUERY HEAD, NOT PER KV HEAD.
    Under GQA, G query heads read one KV head. Each query head has its own
    attention row and therefore its own answer to "did my attention escape".
    Averaging the group into a single row before testing the threshold hides
    the heterogeneous case -- three local heads plus one escaping head dilute
    the residual below 0.25 and the escape never registers, which biases every
    policy upward. Canon is MHA so G=1 and the question never arose.

    The event is per query head; the RESIDENCY it is tested against is per KV
    head, because that is what the cache physically is. That asymmetry is real:
    under GQA one resident set has to serve G possibly-divergent queries. Note
    it is NOT a differentiator between cluster and token policies -- H2O and
    SnapKV inherit exactly the same constraint through group_sum, so do not
    quote it as an excuse for the prefetcher.

    Per query head q (belonging to KV head h), at decode step t:
        mass[c]  = q's nosink row aggregated by h's cluster assignment
        cur      = c[t-1], h's most recent key cluster
        residual = mass with h's hubs and cur zeroed
        escape   = residual.sum() > threshold
        cstar    = residual.argmax()
        hit      = cstar in h's chosen cluster set

    cstar can never be a hub: hubs are zeroed before the argmax.
    """

    def __init__(self, index: ClusterIndex, threshold: float = ESCAPE_THRESHOLD):
        self.index = index
        self.threshold = threshold
        self._assign: Dict[int, torch.Tensor] = {}

    def reset(self, device):
        self.index = self.index.to(device)
        self._assign = {}

    def note_keys(self, layer: int, keys_new: torch.Tensor) -> None:
        ids = self.index.assign(layer, keys_new)              # [Hkv, n_new]
        seen = self._assign.get(layer)
        self._assign[layer] = ids if seen is None else torch.cat([seen, ids], dim=1)

    def score_step(self, layer: int, mass_rows: torch.Tensor, groups: int,
                   resident: Optional[torch.Tensor],
                   chosen: Optional[torch.Tensor]
                   ) -> Tuple[float, float, float, float]:
        """
        mass_rows  [Hq, K] float32, one nosink row per QUERY head; each sums to 1
        groups     G = Hq // Hkv
        resident   [Hkv, K] bool, or None for the full cache
        chosen     [Hkv, C] bool cluster set, or None for token policies

        -> (n_escapes, recovered_on_escapes, n_hits, n_scored), counted in
           query-head-steps
        """
        assign = self._assign.get(layer)
        if assign is None or assign.shape[1] < 2:
            return 0.0, 0.0, 0.0, 0.0

        n_c = self.index.n_clusters
        n_q, k_len = mass_rows.shape
        device = mass_rows.device

        # every per-KV-head object is expanded to query heads. repeat_interleave
        # matches repeat_kv's ordering: [kv0 x G, kv1 x G, ...]
        ids_kv = assign[:, :k_len]                            # [Hkv, K]
        ids_q = ids_kv.repeat_interleave(groups, dim=0)       # [Hq, K]

        by_cluster = torch.zeros(n_q, n_c, dtype=torch.float32, device=device)
        by_cluster.scatter_add_(1, ids_q, mass_rows)          # [Hq, C]

        # ids[:, -1] is position t (just written), so c[t-1] is ids[:, -2]
        cur_q = ids_kv[:, -2].repeat_interleave(groups)       # [Hq]
        hub_q = self.index.hub_mask(layer).repeat_interleave(groups, dim=0)

        residual = by_cluster.masked_fill(hub_q, 0.0)
        residual = residual.scatter(1, cur_q.unsqueeze(1), 0.0)   # [Hq, C]

        escaped = residual.sum(dim=1) > self.threshold        # [Hq] bool
        n_escapes = float(escaped.sum().item())
        if n_escapes == 0:
            return 0.0, 0.0, 0.0, 0.0

        if resident is None:
            kept = torch.ones(n_q, device=device)
        else:
            kept = (mass_rows * resident.repeat_interleave(groups, dim=0)).sum(dim=1)
        recovered = float((kept * escaped).sum().item())

        hits = scored = 0.0
        if chosen is not None:
            cstar = residual.argmax(dim=1)                    # [Hq]
            chosen_q = chosen.repeat_interleave(groups, dim=0)
            hit = torch.gather(chosen_q, 1, cstar.unsqueeze(1)).squeeze(1)
            hits = float((hit & escaped).sum().item())
            scored = n_escapes
        return n_escapes, recovered, hits, scored


# ==========================================================================
# harness state
# ==========================================================================

class PolicyState:
    """Everything the patched attention function needs. One per run; the policy
    is swapped between sweeps and reset_sequence() is called per session."""

    def __init__(self, policy, n_sink: int = 4, query_chunk: int = 512,
                 track_recovery: bool = True,
                 escape_index: Optional[ClusterIndex] = None,
                 escape_threshold: float = ESCAPE_THRESHOLD,
                 row_capture=None):
        # row_capture(layer, first_row_index, rows) is called with the PER-QUERY-ROW
        # nosink mass, [Hkv, n_rows, K], for every row including prefill rows.
        # kvh.harvest uses it. canon captures every row of the [T, T] attention
        # matrix in one forward pass; skipping prefill rows would build the flow
        # graph on a different support than mt_common.build_flow, which loops
        # over ALL t >= 1. Different support means a possibly different hub,
        # which means a different lineage.
        self.row_capture = row_capture
        self.policy = policy
        self.n_sink = n_sink
        self.query_chunk = query_chunk         # caps the [Q, K] score matrix
        self.track_recovery = track_recovery
        self.enabled = True
        self.acct = Accounting()
        self.escape = (EscapeReadout(escape_index, escape_threshold)
                       if escape_index is not None else None)
        self._kv_len: Dict[int, int] = {}      # layer -> KV length before this call

    def reset_sequence(self, n_layers, n_kv_heads, head_dim, device, dtype):
        self._kv_len.clear()
        self.acct.reset()
        self.policy.n_sink = self.n_sink
        self.policy.reset(n_layers, n_kv_heads, head_dim, device, dtype)
        if self.escape is not None:
            self.escape.reset(device)


# ==========================================================================
# small helpers
# ==========================================================================

def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """[B, Hkv, K, D] -> [B, Hq, K, D] by repeating each KV head G times.
    Transcription of transformers.modeling_utils.repeat_kv."""
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


def group_sum(per_query_head: torch.Tensor, n_kv: int, groups: int) -> torch.Tensor:
    """[Hq, K] -> [Hkv, K] by summing the G query heads that share a KV head.
    Query head order after repeat_kv is [kv0 x G, kv1 x G, ...], so a plain
    reshape to [Hkv, G, K] groups them correctly."""
    return per_query_head.view(n_kv, groups, -1).sum(dim=1)


def group_mean(per_query_head: torch.Tensor, n_kv: int, groups: int) -> torch.Tensor:
    return per_query_head.view(n_kv, groups, -1).mean(dim=1)


def to_nosink(probs: torch.Tensor) -> torch.Tensor:
    """The mt_harvest mass convention, applied per query row.

    probs: [..., K] a softmax distribution.
    Zero column 0, then RENORMALISE so the row sums to 1 again.

    The clamp mirrors mt_harvest's `rs.clamp_min(1e-12)`: a row whose only
    mass was on column 0 (position 0 attending to itself) comes out all zeros
    rather than NaN.
    """
    out = probs.clone()
    out[..., 0] = 0.0
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def build_resident_mask(policy_mask: Optional[torch.Tensor], n_kv: int,
                        k_len: int, n_old: int, device) -> Optional[torch.Tensor]:
    """Policy mask over old columns -> full residency mask over all K columns.

    policy_mask: [Hkv, n_old] bool, or None
    -> [Hkv, K] bool, with columns [n_old, K) forced True (written this call),
       or None when the policy kept everything.
    """
    if policy_mask is None:
        return None
    full = torch.ones(n_kv, k_len, dtype=torch.bool, device=device)
    full[:, :n_old] = policy_mask
    return full


# ==========================================================================
# the patched attention function
# ==========================================================================

def make_attention_fn(state: PolicyState):
    """Returns a drop-in replacement for transformers' eager_attention_forward.

    Signature must match:
        (module, query, key, value, attention_mask, scaling=, dropout=, **kw)
        -> (attn_output [B, Q, Hq, D], attn_weights or None)

    We return None for attn_weights. That is fine as long as nothing calls the
    model with output_attentions=True; if something does, it will get None
    rather than a wrong tensor.
    """

    def attention_forward(module, query, key, value, attention_mask,
                          scaling=None, dropout=0.0, **kwargs):
        # ---- shapes -----------------------------------------------------
        b, n_q_heads, q_len, head_dim = query.shape        # [B, Hq, Q, D]
        n_kv_heads, k_len = key.shape[1], key.shape[2]     # [B, Hkv, K, D]
        groups = n_q_heads // n_kv_heads
        if scaling is None:
            scaling = head_dim ** -0.5
        layer = int(getattr(module, "layer_idx", 0))

        if b != 1:
            raise RuntimeError("kvh runs batch size 1: residency accounting and "
                               "cluster state are per-sequence")

        key_rep = repeat_kv(key, groups)                   # [B, Hq, K, D]
        value_rep = repeat_kv(value, groups)               # [B, Hq, K, D]

        n_old = state._kv_len.get(layer, 0)
        n_new = k_len - n_old
        state._kv_len[layer] = k_len

        if not state.enabled:
            return _plain_attention(query, key_rep, value_rep, attention_mask,
                                    scaling, state.query_chunk), None

        # ---- STEP 1: select, while the newest cluster is still c[t-1] -----
        if getattr(state.policy, "needs_true_attention", False) and n_old > 0:
            true_mass = true_column_mass(query, key_rep, attention_mask, scaling,
                                         n_kv_heads, groups, state.query_chunk)
            state.policy.set_peek(layer, true_mass[:, :n_old])

        policy_mask = state.policy.select(layer, n_old, n_new)   # [Hkv, n_old] | None

        # ---- STEP 2: only now do the new keys exist ----------------------
        if n_new > 0:
            new_keys = key[0, :, n_old:, :].detach()             # [Hkv, n_new, D]
            state.policy.note_keys(layer, new_keys)
            if state.escape is not None:
                state.escape.note_keys(layer, new_keys)

        resident = build_resident_mask(policy_mask, n_kv_heads, k_len, n_old,
                                       query.device)             # [Hkv, K] | None
        if resident is None:
            per_head_resident = torch.full((n_kv_heads,), float(k_len),
                                           device=query.device)
        else:
            per_head_resident = resident.sum(dim=1).float()       # [Hkv]
        per_head_columns = torch.full((n_kv_heads,), float(k_len), device=query.device)
        n_resident = float(per_head_resident.sum().item())

        # broadcastable to the score tensor [B, Hq, Q, K]
        score_mask = (None if resident is None
                      else resident.repeat_interleave(groups, dim=0)[None, :, None, :])

        # ---- STEP 3: masked attention ------------------------------------
        # a policy that wants an observation window says how many trailing
        # query rows it needs; 0 means the buffer is never allocated.
        obs_rows = int(getattr(state.policy, "obs_rows", 0) or 0)
        out, stats = _masked_attention(
            query, key_rep, value_rep, attention_mask, scaling, score_mask,
            n_kv_heads, groups, state.query_chunk, state.track_recovery,
            row_capture=state.row_capture, layer=layer, obs_rows=obs_rows)

        # ---- STEP 4: escape readout (decode steps only) -------------------
        if state.escape is not None and q_len == 1:
            n_esc, recovered, hits, scored = state.escape.score_step(
                layer, stats["nosink_query_rows"], groups, resident,
                state.policy.chosen_clusters(layer))
            # denominator is QUERY-head-steps, matching the event definition
            state.acct.add_escape_step(float(n_q_heads), n_esc, recovered,
                                       hits, scored)

        # ---- STEP 5: bookkeeping and the policy's own update --------------
        state.acct.add_call(layer, n_resident, float(k_len * n_kv_heads),
                            stats["recovered_sum"], stats["recovered_rows"],
                            per_head_resident.cpu(), per_head_columns.cpu())
        state.policy.observe(Obs(
            layer=layer,
            col_mass_resident=stats["mass_resident"],
            col_mass_true=stats["mass_true"],
            col_mass_nosink=stats["mass_nosink"],
            col_mass_resident_lastw=stats["mass_resident_lastw"],
            n_old=n_old, n_new=n_new, is_prefill=(q_len > 1)))

        return out, None

    return attention_forward


def _masked_attention(query, key_rep, value_rep, attention_mask, scaling,
                      score_mask, n_kv, groups, chunk, track_recovery,
                      row_capture=None, layer=0, obs_rows=0):
    """The actual computation, chunked over query rows so the [Q, K] score
    matrix never has to exist all at once.

    Returns (output [B, Q, Hq, D], stats dict). Every stat is [Hkv, K] except
    mass_resident_lastw (None unless obs_rows > 0) and the two recovery scalars.
    """
    b, n_q_heads, q_len, head_dim = query.shape
    k_len = key_rep.shape[2]
    device = query.device

    out = torch.empty(b, q_len, n_q_heads, head_dim, dtype=query.dtype, device=device)
    zeros = lambda: torch.zeros(n_kv, k_len, dtype=torch.float32, device=device)
    mass_true, mass_resident, mass_nosink = zeros(), zeros(), zeros()
    nosink_query_rows = None      # [Hq, K], the LAST query row, pre-group-mean
    recovered_sum = recovered_rows = 0.0

    # rolling buffer of the last `obs_rows` query rows of the MASKED mass.
    # Held as a list of chunk tails so it never depends on `chunk`: each chunk
    # contributes at most obs_rows rows and the front is dropped once the
    # remaining entries already cover the window.
    tail_buf: List[torch.Tensor] = []
    tail_len = 0

    for start in range(0, q_len, chunk):
        stop = min(q_len, start + chunk)
        rows = stop - start

        scores = torch.matmul(query[:, :, start:stop],
                              key_rep.transpose(2, 3)) * scaling   # [B, Hq, q, K]
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, start:stop, :k_len]

        probs_true = F.softmax(scores, dim=-1, dtype=torch.float32)
        mass_true += group_sum(probs_true.sum(dim=2)[0], n_kv, groups)

        # canon mass: per query row, drop column 0 and renormalise, then reduce
        # over the G query heads sharing this KV head.
        #
        # MEAN, not sum. Each renormalised query row sums to 1, so a mean keeps
        # the KV-head row summing to 1 and keeps THRESH=0.25 meaning what it
        # means in mt_common. A sum would scale everything by G and silently
        # recalibrate the escape threshold.
        #
        # This is the one convention canon could not have specified: Pythia is
        # MHA, so G=1 and the question never arose. See REVIEW.md -- the live
        # alternative is to score G escape events per KV head instead of
        # averaging them into one.
        per_query = to_nosink(probs_true)[0]                  # [Hq, q, K]
        if stop == q_len:
            nosink_query_rows = per_query[:, -1, :]           # [Hq, K], last row
        grouped = per_query.view(n_kv, groups, rows, k_len).mean(dim=1)
        if row_capture is not None:
            row_capture(layer, start, grouped)                # [Hkv, q, K]
        mass_nosink += grouped.sum(dim=1)                     # [Hkv, K]

        if score_mask is None:
            probs_used = probs_true
        else:
            probs_used = F.softmax(scores.masked_fill(~score_mask, float("-inf")),
                                   dim=-1, dtype=torch.float32)
            if track_recovery:
                kept = (probs_true * score_mask).sum(dim=-1)       # [B, Hq, q]
                recovered_sum += float(kept.sum().item())
                recovered_rows += float(kept.numel())

        mass_resident += group_sum(probs_used.sum(dim=2)[0], n_kv, groups)

        # SUM over the group, matching mass_resident above -- the observation
        # window has to live in the same units as the score it replaces.
        if obs_rows > 0:
            per_row = probs_used[0].view(n_kv, groups, rows, k_len).sum(dim=1)
            take = min(obs_rows, rows)
            tail_buf.append(per_row[:, rows - take:, :].float())
            tail_len += take
            while tail_buf and tail_len - tail_buf[0].shape[1] >= obs_rows:
                tail_len -= tail_buf.pop(0).shape[1]

        out[:, start:stop] = torch.matmul(probs_used.to(query.dtype),
                                          value_rep).transpose(1, 2)

    mass_resident_lastw = None
    if obs_rows > 0 and tail_buf:
        cat = tail_buf[0] if len(tail_buf) == 1 else torch.cat(tail_buf, dim=1)
        mass_resident_lastw = cat[:, -obs_rows:, :].sum(dim=1)     # [Hkv, K]

    if score_mask is None:
        # full cache: recovery is 1.0 by definition, recorded explicitly rather
        # than left as 0/0
        recovered_sum = recovered_rows = float(n_q_heads * q_len)

    return out.contiguous(), dict(
        mass_true=mass_true, mass_resident=mass_resident, mass_nosink=mass_nosink,
        mass_resident_lastw=mass_resident_lastw,
        nosink_query_rows=nosink_query_rows,
        recovered_sum=recovered_sum, recovered_rows=recovered_rows)


def _plain_attention(query, key_rep, value_rep, attention_mask, scaling, chunk):
    """Unpatched behaviour, used when state.enabled is False."""
    b, n_q_heads, q_len, head_dim = query.shape
    k_len = key_rep.shape[2]
    out = torch.empty(b, q_len, n_q_heads, head_dim, dtype=query.dtype,
                      device=query.device)
    for start in range(0, q_len, chunk):
        stop = min(q_len, start + chunk)
        scores = torch.matmul(query[:, :, start:stop],
                              key_rep.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, start:stop, :k_len]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, start:stop] = torch.matmul(probs, value_rep).transpose(1, 2)
    return out.contiguous()


def true_column_mass(query, key_rep, attention_mask, scaling, n_kv, groups, chunk):
    """[Hkv, K] true column mass. Only for the oracle -- it costs a second full
    attention pass, which is why nothing else calls it."""
    b, n_q_heads, q_len, _ = query.shape
    k_len = key_rep.shape[2]
    total = torch.zeros(n_kv, k_len, dtype=torch.float32, device=query.device)
    for start in range(0, q_len, chunk):
        stop = min(q_len, start + chunk)
        scores = torch.matmul(query[:, :, start:stop],
                              key_rep.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, start:stop, :k_len]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        total += group_sum(probs.sum(dim=2)[0], n_kv, groups)
    return total


# ==========================================================================
# installation
# ==========================================================================

def install(model, state: PolicyState):
    """Swap the model family's module-level `eager_attention_forward`.

    Found by reflection off the first *Attention submodule rather than by
    importing a specific modeling file, so this works for any family whose
    attention resolves that symbol at call time. Returns restore().

    Requires attn_implementation="eager" because that is the branch where
    transformers materialises the additive causal mask this code adds to the
    scores. sdpa/flash paths may hand in None or a different form.
    """
    attention_block = next(
        (m for m in model.modules() if type(m).__name__.endswith("Attention")), None)
    if attention_block is None:
        raise RuntimeError("no *Attention submodule found on this model")

    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "eager":
        raise RuntimeError(
            f"load the model with attn_implementation='eager' (got {impl!r}); "
            "other backends do not materialise the additive mask this harness needs")

    modeling = sys.modules[type(attention_block).__module__]
    if not hasattr(modeling, "eager_attention_forward"):
        raise RuntimeError(f"{modeling.__name__} has no eager_attention_forward")

    original = modeling.eager_attention_forward
    modeling.eager_attention_forward = make_attention_fn(state)

    def restore():
        modeling.eager_attention_forward = original

    return restore


def load_causal_lm(model_id, dtype):
    """from_pretrained with attn_implementation='eager', handling the
    torch_dtype -> dtype rename in transformers 4.56."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager")
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager")


def model_shape(model) -> Tuple[int, int, int]:
    """-> (n_layers, n_kv_heads, head_dim)"""
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers, n_kv, head_dim
