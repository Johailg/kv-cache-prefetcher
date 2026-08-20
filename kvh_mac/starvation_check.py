"""
kvh.starvation_check -- read an existing run_ppl CSV and say how much of each
headline number is early-turn starvation rather than policy quality.

    python starvation_check.py results/ppl.csv
    python starvation_check.py results/ppl.csv --n-sink 4

No model, no GPU, no torch. Runs on the CSV you already have.

THE PROBLEM IT MEASURES
-----------------------
Under a fractional budget, _budget_to_count floors at n_sink:

    keep = clamp(round(budget * n_old), n_sink, n_old)

so a token policy is LITERALLY sink_only until round(budget * n_old) > n_sink,
i.e. until n_old > n_sink / budget. At budget 0.03 with n_sink 4 that is
n_old > 133, and it stays near-sink_only for a good while after. sink_only is
ppl ~3654 in the Aug 12 run, so a handful of starved early steps can dominate
the average for a whole turn.

No paper does this. H2O returns the cache untouched while seq_len <=
hh_size + recent_size; SnapKV skips compression entirely while q_len <
max_capacity_prompt. Both are absolute-cache-size formulations, and neither
ever presents the model with a four-token cache. So the low-budget rows of a
fractional sweep carry a penalty the published methods would not pay.

WHAT TO DO WITH THE ANSWER
--------------------------
If turn 0 is an order of magnitude worse than turn 7 at the low budgets, the
0.03 and 0.06 columns are measuring the knob's floor, not the policy, and
should either be re-run with absolute budgets (policies now accept budget >= 1
as an absolute count) or reported with the caveat attached. If turn 0 is
merely somewhat worse, this is second-order and one sentence covers it.
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Dict, List, Optional


def read_rows(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def turn_columns(rows: List[dict]) -> List[str]:
    keys = {k for r in rows for k in r if k.startswith("ppl_t")}
    return sorted(keys, key=lambda k: int(k[5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="the run_ppl output, e.g. results/ppl.csv")
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--flag-ratio", type=float, default=3.0,
                    help="turn0/last ratio above which a row is flagged")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    tcols = turn_columns(rows)
    if not tcols:
        raise SystemExit("no ppl_t* columns in this CSV -- nothing to check")

    sink_only = next((r for r in rows if r.get("policy") == "sink_only"), None)
    if sink_only is not None:
        print(f"sink_only floor: ppl {num(sink_only, 'ppl'):.1f} at residency "
              f"{num(sink_only, 'residency'):.4f}\n")

    print(f"{'policy':14s} {'knob':>7s} {'kind':>6s} {'resid':>7s} {'ppl':>9s} "
          f"{'t0':>9s} {'tlast':>9s} {'t0/tlast':>9s}  starved-until")
    print("-" * 96)

    flagged = []
    for r in rows:
        pol, kind = r.get("policy", "?"), r.get("knob_kind", "")
        knob = num(r, "knob")
        first = num(r, tcols[0])
        last = None
        for c in reversed(tcols):
            last = num(r, c)
            if last is not None:
                last_name = c
                break
        if first is None or last is None or last <= 0:
            continue
        ratio = first / last

        # the position at which the budget stops being the n_sink floor
        if kind == "budget" and knob and 0 < knob < 1:
            n_star = args.n_sink / knob
            starved = f"n_old>{n_star:.0f}"
        elif kind == "budget" and knob and knob >= 1:
            starved = "absolute"
        else:
            starved = "-"

        mark = "  <-- flag" if ratio >= args.flag_ratio else ""
        resid = num(r, "residency")
        print(f"{pol:14s} {knob if knob is not None else float('nan'):7.4g} "
              f"{kind:>6s} {resid if resid is not None else float('nan'):7.4f} "
              f"{num(r, 'ppl'):9.3f} {first:9.3f} {last:9.3f} {ratio:9.2f}"
              f"  {starved}{mark}")
        if ratio >= args.flag_ratio:
            flagged.append((pol, knob, ratio, first, last))

    print()
    if not flagged:
        print(f"No row has turn-0 ppl >= {args.flag_ratio}x its last turn. "
              f"Early-turn starvation is not driving these numbers; one "
              f"sentence in the methods section covers the n_sink floor.")
        return

    print(f"{len(flagged)} row(s) with turn-0 ppl >= {args.flag_ratio}x the "
          f"last turn ({last_name}):")
    for pol, knob, ratio, first, last in flagged:
        print(f"  {pol} @ {knob:g}: {first:.1f} -> {last:.1f} ({ratio:.1f}x)")
    print("\nThese rows are partly measuring the fractional knob's n_sink "
          "floor rather than the policy. Re-run them with an ABSOLUTE budget "
          "(pass --budgets as integers, e.g. 64,128,256) to match the papers' "
          "formulation, or state the caveat and keep the residency axis.")


if __name__ == "__main__":
    main()
