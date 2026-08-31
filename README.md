# KV Cache Prefetching for Long-Context Inference

A KV cache prefetcher has to decide what to fetch before it knows what it needs. This repo is two phases of work on that.

**Phase 1** asks whether there is a learnable graph structure over the clusters formed by the KV vectors in their space, and whether you can use it to predict what to stage next.

**Phase 2** stops asking whether my method works and asks what the design space looks like instead. Two questions: **how far ahead can you decide before the model's loss on the next token starts climbing**, and **how much is good prediction worth at all**. That is what the `kvh` harness was built to measure.

Short version of the outcome: the Phase 1 method does not survive a fair comparison. The Phase 2 measurements do, and they are the contribution.

---

## Phase 1 — offline clustering on Pythia-410M

Full write-up in `memo.tex` / `memo_v2.tex`. This is the summary, including what did not survive.

KV-cache scaling introduces the problem of cache missing and cache thrashing in LLMs, leading to inefficient processing times for prompts. Phase 1 proposes an offline clustered prefetch architecture that uses the attention of a given head at a given layer as a signal for predicting the next cluster needed to evaluate the query. The hypothesis is that there is a learnable graph structure over the clusters formed by the KV vectors in their space, defined by the flow of attention through those clusters, and that it works like a Markov structure.

I chose Pythia-410M and focused on layer 13 head 2. The KV vectors were clustered with K-means over 500-token windows, 64 clusters, on Wikipedia text and coding prompts.

**Coverage** is the fraction of the mass that fell inside the resident set of clusters. The setup: you are a prefetcher, you get to keep *k* clusters resident at any moment, new tokens arrive and their queries attend over past keys. Attention at step *t* gives a distribution over past keys; group those keys by cluster and you get a distribution over 64 clusters. The predictors compared were:

- **Oracle** looks at the true attention mass and keeps the top-*k*. Unbeatable upper bound.
- **Flow graph** keeps the top-*k* clusters predicted by the average attention destination profile one step after each cluster.
- **Recency** keeps the last *k* distinct clusters seen. The null hypothesis that "attention is just local."
- **Static** keeps the globally most attended clusters, no dynamics at all. The null hypothesis that "one frequency table is enough."

What was found:

- **Temporal structure exists.** At order zero, basically the cluster frequencies, cross entropy was 5.98 bits, close to the ceiling. Conditioning on the previous cluster dropped it to 2.11 bits. That shows evidence of some temporal structure in the flow of attention. Self-loop rate 0.697.
- **A hub cluster.** One cluster, id 16, about 0.4% of all keys, carries roughly 0.575 of non-sink attention mass. Its contents on wiki are mostly space and newline and other whitespace and punctuation — a small, content-free vocabulary. On code, 78.8% of the hub keys are the newline token. The same cluster id captures a similar mass fraction on both domains. It is sink-like in function but distributed in position, unlike the real sink at position 0.
- **Coverage.** With the hub pinned and its flow zeroed, the flow graph still reached 0.951 mean captured mass at k=4 on held-out wiki, against 0.815 for static.
- **A break-even inequality.** For this entire prediction mechanism to work at all we need to beat the base case, a static cache. That gives `h · min(1, H/L) > h_static`. With realistic constants, L/t is about 0.008, so it collapses to `h > h_static`. Latency is not the constraint, prediction quality is.

**What did not survive.** The memo compares the flow graph against static, and never against a properly implemented LRU under the same escape-conditioned metric. When I added that baseline, the flow graph's margin over fair LRU was about **+0.015 at k=2** and went **negative by k=8**. The static collapse is real and it replicates on Qwen3, but "beats a frequency table" is a much weaker claim than the memo reads as making. Phase 2 exists because of this.

The Qwen3 port also killed the hub. Hub mass shares run 0.023 to 0.148 across all 28 layers, against Pythia's 0.62. There is no dominant hub in Qwen3 at any layer, measured at 409,970 keys per head. The whole hub framework is a Pythia-family result.

---

## Phase 2 — `kvh`, end to end on Qwen3

### What it measures

`kvh` runs a cache selection policy against the full cache and scores the damage in **perplexity**, the exponential of the cross entropy between the model's current output distribution and the indicator distribution of the actual next token. Non-resident KV columns are masked and the row renormalised, which gives logits identical to a real evicted cache. No memory or latency is saved, so this is a quality study only. A latency claim needs a separate artifact.

I profiled **H2O**, **StreamingLLM**, **SnapKV**, **LRU**, a **paging oracle**, and my own custom cluster method. I varied the lag time as well as the policies themselves. Selection is per `(layer, kv_head)`, because under GQA that is the granularity the cache physically has.

Everything is compared on **measured residency**, the fraction of KV entries actually resident, averaged over decode steps and heads. Not on the budget knob. Different policies land at different residencies for the same knob, so the knob is not a fair axis and the measured fraction is.

### How the lag actually works

This is the part that sounds wrong at first. You are not prefetching what you had ten steps ago. The lag is in **the information available at decision time**.

A real prefetcher has to issue the fetch early enough that the transfer lands before the token needs it. If the transfer takes L decode steps, then the decision about what to fetch is made L steps before the data gets used, which means it is made using only what was knowable L steps ago.

So the experiment is: at step *t*, pick the resident block set using the attention distribution from step *t−L*, then score that set against the attention that actually happens at *t*. Nothing is evicted. The cache contents are always exact. Only the **selection** is stale.

The selector is an **oracle**, so at *t−L* it makes the perfectly optimal choice for that moment. The curve is therefore a **ceiling on any predictor operating at lead time L**, not a description of a method.

Every arm carries a fixed 32-token recent window, which every system in this literature does. See the limitations, it matters.

### Headline numbers

10 sessions, Qwen3-0.6B, max-len 2048, teacher-forced, 14,394 scored tokens, full-cache ppl **5.2262**.

**Lead time is cheap, and the cost is a clean power law.**

| L (decode steps) | 0 | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| residency | 0.1354 | 0.1362 | 0.1384 | 0.1476 | 0.1596 | 0.1642 |
| ppl | 5.382 | 5.626 | 6.351 | 7.299 | 7.517 | 8.158 |

`ppl = 5.63 · L^0.089` fits L = 1 to 64 within 2%. Two parameters, six doublings. **Each doubling of lead time costs about 6.4% perplexity.** So a prefetcher can afford a predictor that takes tens of decode steps to run. Nobody currently assumes that design freedom exists.

**How much prediction is worth.** Against StreamingLLM, which is sinks plus a window and zero prediction, at matched bytes:

- At about 0.142 residency, perfect prediction is worth **3.14 ppl**.
- H2O's heuristic captures about **31%** of that.
- My learned cluster policy captures about **4.5%**.
- The value of prediction decays to zero between L=32 and L=64, crossing around L≈55. Past that a perfect block predictor is worth no more than doing nothing.

### Where my method fails

Two results invalidate it, and both are in this repo.

**H2O at matched bytes beats the lagged oracle from about L≈20 onwards.** H2O gets 7.109 at residency 0.1612 against the oracle's 7.517 at 0.1596, and H2O has slightly *more* cache there, so correcting makes it worse for me. Since the oracle is a ceiling, beating H2O would require a predictor that is both near-oracle and running 20 steps ahead.

**Equal-width position blocks beat my content clusters.** The paging oracle at k=9 gets 5.372 at residency 0.1416; the cluster oracle at k=4 gets 5.605 at 0.1433. Worse quality at slightly more cache. So my grouping added nothing either.

My prediction was not good, and the paging I adopted proved no extra value. But there clearly remains room for better prediction, as the lead-time result shows: the gap between doing nothing and doing it perfectly is 3.14 ppl and the best published heuristic only takes a third of it.

One note on the blocks-beat-clusters result. It runs against published consensus — ClusterKV, LouisKV and others argue semantic clusters beat paging. My read is that those comparisons pit real *methods* against each other, and the page-based arm is usually Quest, whose per-page score is a crude min-max upper bound, so the literature conflates *grouping* with *estimator*. Take the estimator out by putting an oracle on both sides and blocks win. That disambiguation is the interesting part, not the bare claim.

---

## Repo layout

```
kv-cache-prefetcher/     Phase 1, Pythia lineage
  mt_*.py                harvest / train_flow / eval / masscurve pipeline
  memo.tex, memo_v2.tex  the Phase 1 write-up
  *.npz                  flow graphs, centroids, harvest shards

harness/, kvh_mac/       Phase 2, the kvh harness
  attn.py                masking + renormalisation, true-attention peek
  policies.py            every policy, including PageOracle / LaggedPageOracle
  run_ppl.py             driver, writes the CSV after every job
  gate_*.py, gates.py    identity and equivalence gates
  results/               every run quoted above
```

Every Phase 2 number came from `kvh_mac`, on a MacBook Air over MPS.

---

## Results inventory

**Valid.**

| file | what it holds |
|---|---|
| `leadtime_w32.csv` | the lead-time curve, k=8 blocks + W=32 window, L = 0/1/4/16 |
| `leadtime_long.csv` | L = 32 and 64, plus a no-reset control at L=16 |
| `matched_baselines.csv` | streaming and h2o at residencies matched to the lag curve, 0.135 / 0.16 / 0.164 |
| `ppl_granularity.csv` | routed and cluster_oracle at k=4 |
| `ppl_gates.csv` | the token oracle at matched budget 0.1433, plus cluster_oracle at k=2 and k=8, the monotonicity gate |
| `e2e_page.csv` | page_oracle k=9 and k=10, the blocks-vs-clusters control |
| `ppl_tok_fixed.csv` | h2o and snapkv after the baseline-fidelity corrections, see the caveat below |
| `index_noop.csv`, `index_noop_off.csv` | the `--index` no-op check, identical to 4 dp, so runs with and without the escape readout are directly comparable |
| `gate_lag0`, `gate_w32`, `gate_noreset`, `gate_page` | identity gates, described below |

Every valid file carries the same checksum row: `full`, n_tok 14394, ppl 5.2262, and identical per-turn perplexities. If a run does not reproduce that row it is void.

The monotonicity gate in `ppl_gates.csv` passes on all three measures. cluster_oracle at k=2/4/8 gives residency 0.0883 / 0.1433 / 0.2545, ppl 7.0598 / 5.605 / 5.2834, h_esc 0.3186 / 0.8470 / 0.9361. A non-monotone oracle would be a bug, not a finding.

**Void, do not quote any row from these.**

`leadtime.csv` — the first lead-time run, k=9 with **no local window**. It reports 6.894 / 19.222 / 444.909 at L = 1/4/16 and all three are meaningless.

Why: when the cache is cold and we page it, the pages are really narrow. A new block opens every `ceil(n_old/64)` positions, which at small `n_old` is one new block every two tokens. The stale peek does not cover the newest columns at all, so the page holding the just-generated token scores zero and gets dropped from the resident set even though it contains the most important tokens in the row. Attention from token *t* to token *t−1* is one of the largest single entries there, so dropping it deletes local context, and that alone caused the cross entropy to increase dramatically.

The per-turn diagnostic confirmed it. Excess NLL at L=1 runs 1.044 at turn 0 down to 0.081 at turn 7, strictly monotone, a 12.9x ratio, which is almost exactly the predicted block-width ratio. So essentially the whole L=1 penalty was the dropped live page, not attention churn. Adding a fixed 32-token window brought L=1 from 6.894 to 5.626.

`ppl_tok_fixed.csv`, the h2o rows at budgets 0.03 and 0.06 — **void, these are not H2O.** Under `CHARGE_SINKS` the content budget is `keep − n_sink`, which is ≤ 0 while `round(budget · n_old) ≤ n_sink`, so the mask is empty and the monotone eviction ledger marks every non-sink column permanently evicted. The dead zone runs from token 4 to token 4/budget. Real H2O leaves the cache untouched until it outgrows the budget. The h2o row at 0.125 (0.1263 / 7.866) is fine, and all the snapkv rows are fine.

**Superseded.** `runs/e2e_token.csv` on Delta predates the Aug 13 baseline-fidelity corrections. Use `ppl_tok_fixed.csv`.

### Gates

Nothing here is trusted without a mechanical identity check. Every bug in this project was found by a gate firing or by ground truth arriving from outside, never by re-reading code that looked internally consistent.

- **G1** — the patched full cache reproduces stock logits, `max|Δlogit| = 0.000e+00` on a real model. Any downstream delta is caused by the policy and nothing else.
- **`gate_canon`** — the cluster selection layer diffed against the Phase 1 reference implementation, 0.000% mismatch on every policy over thousands of draws.
- **`gate_pythia`** — Phase 1 sessions re-run through `kvh` and diffed against the stored Phase 1 attention rows, 0 of 10,235 escape-event disagreements.
- **`gate_lag0`** — the lagged oracle at L=0 has to exactly reproduce the unlagged one. **This gate caught a real failure.** A misindented `select` meant `LaggedPageOracle` silently inherited the parent's method, which returned `None`, and in this harness `None` means *keep everything*. The job ran to completion and printed a plausible-looking row at residency 1.0000. A policy that returns nothing reads as success, and residency is the column that exposes it.
- **`gate_w32` / `gate_noreset` / `gate_page`** — the same pattern for the window, the prefill-reset flag, and the paging policy. At k = C the paging oracle has to give residency 1.0 and full-cache perplexity.

---

## Limitations

**One model, one context length.** Everything is Qwen3-0.6B at 2048 tokens, on a model trained for 32k. The field's operating point is 7 to 8B at 32k to 128k. The granularity result in particular is a claim about the geometry of key space, and one architecture cannot establish it.

**No confidence intervals.** There are none anywhere. The `±` printed on residency is `resid_spread`, the dispersion across KV heads within a run, not an interval on the mean. The only robustness evidence is an 8/8 per-turn sign test, p = 0.004, on cluster_oracle vs routed. `run_ppl.aggregate()` computes per-session records and then discards them, so nothing here can currently be bootstrapped.

**The baselines are my implementations, not the ones given by the creators.** H2O, SnapKV and StreamingLLM were written from the papers, then audited line by line against the released reference implementations, with the deviations documented and the corrections applied. Both corrections made the *baselines* stronger, which widened the gap against my own method. They pass mechanical gates against the reference selections, but they are still not the authors' code, and SnapKV was the most deviant before the fix.

**Teacher forcing.** Errors never compound, so every number is a lower bound on divergence. Free-running perplexity is not a fix — a policy that degenerates into repetition scores *low* perplexity on its own output, and under free generation each policy writes a different token sequence, so the metric is computed on different data. The right fix is a downstream task with checkable answers, needle-in-a-haystack or LongBench, which the KV-compression literature uses and this repo does not have yet.

**Fractional budgets starve turn 0 for token policies, and it contaminates the granularity tax.** A token policy at a fractional budget keeps `round(budget · n_old) − n_sink` content tokens, which is tiny while `n_old` is small, so it is badly starved on the first prompt. You can see it in `ppl_gates.csv`: the token oracle's turn-0 ppl is 13.179 against the full cache's 8.488, while on turns 2 through 7 it sits at or below full. A cluster or block policy has no such shrinkage. So any token-vs-cluster granularity number quoted against that reference is contaminated at turn 0, and the clean comparison is **block minus cluster**, 0.233 ppl at 0.0017 less residency, since both sides share the same reference. Re-running the token policies at absolute budgets is the fix.

**"Lead time is cheap" is conditional on the local window.** With W=32 the curve is logarithmically cheap. Without it the same measurement blows up, see `leadtime.csv`. Every real system keeps a local window so the claim stands, but the window is part of the architecture and has to be stated that way.

**The oracle is a ceiling, not a method.** Nothing here is actually prefetched. No transfer counts, no measured speedup, and the contiguity advantage is unmeasured — which is the entire systems rationale for block granularity in the first place.
