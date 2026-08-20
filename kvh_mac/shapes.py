"""
kvh.shapes -- the shape and index vocabulary used in every other module.

This file has no code. It exists so that every `# [Hkv, K]` annotation in the
rest of the package means one fixed thing, and so a reader can check an
annotation without reconstructing it from context.

TENSOR AXES
-----------
    B     batch. Always 1 in this harness. Enforced, not assumed.
    Hq    query heads          (config.num_attention_heads)
    Hkv   key/value heads      (config.num_key_value_heads)
    G     Hq // Hkv            query heads served by one KV head, GQA group size
    Q     query positions in THIS forward call
    K     total KV positions AFTER this call = n_old + n_new
    D     head dim
    C     number of clusters (64, mt_common.K -- renamed here to avoid colliding
          with the KV length, which the modeling code also calls K)
    V     vocab

POSITION VOCABULARY
-------------------
    n_old   KV positions already in the cache when this call began.
            Columns [0, n_old) are the ones a policy may evict.
    n_new   positions written by this call. Columns [n_old, K) are ALWAYS
            resident: they are being produced now.
    t       absolute position index into the sequence.
            At a decode step, K = t+1, n_old = t, n_new = 1.

THE ONE INDEX THAT MATTERS
--------------------------
    cur = c[t-1]     the "live cluster"

    Not c[t]. mt_eval.py line: `cur = int(c[t - 1])`, and mt_common.topk is
    called as topk(P[cur], cur, k, hubs) -- the row index and the seed are the
    same thing. mt_eval also scores recency against `recency_set(c[:t], k)`,
    a history that stops one short of t.

    Consequence for this harness: Policy.select() must run BEFORE
    Policy.note_keys() for the same step, so that the newest cluster the policy
    has seen is c[t-1]. attn.py enforces the ordering; every cluster policy
    then just reads `assign[:, -1]`.

TWO DIFFERENT "SINKS"
---------------------
    n_sink = 4    residency pinning. Positions 0..3 stay resident for every
                  policy so the comparison is not dominated by the attention
                  sink. StreamingLLM's convention.

    column 0      the mass convention. mt_harvest zeroes attention column 0 and
                  RENORMALISES the row before aggregating by cluster. That is
                  what mass_nosink means.

    These are unrelated. Changing one must not change the other.

TWO DIFFERENT "RECOVERY"
------------------------
    Accounting.recovery      mean over all query rows of the raw softmax mass
                             landing on the resident set. Includes column 0.

    Accounting.recovery_esc  mean over ESCAPE steps of the mass_nosink mass
                             landing on the resident set. Column 0 removed and
                             the row renormalised first, because that is the
                             convention the escape event itself is defined in.

    They are close but not equal and must not be quoted interchangeably.
"""