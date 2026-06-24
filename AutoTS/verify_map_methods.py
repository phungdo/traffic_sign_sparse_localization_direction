"""verify_map_methods.py — ensure every sign has all per-algorithm results
==========================================================================
Audits ``results/map/map_sign_results.json`` and confirms that every sign with
at least one location point carries a complete set of aggregator results
(``est_lat/est_lon`` for all methods in ``map_methods``). Signs with zero
location points cannot have any aggregation result and are reported separately
(expected, not a failure).

    python verify_map_methods.py

Exit code is non-zero if any sign-with-points is missing a method, so this can
gate a rebuild in CI / a Makefile.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="./results/map/map_sign_results.json")
    args = parser.parse_args()

    with open(args.map) as f:
        d = json.load(f)
    signs = d["signs"]
    order = d.get("map_methods", [])
    want = set(order)

    complete = no_points = broken = no_gt = 0
    k_ge7 = k_lt7 = 0
    broken_signs = []

    for key, s in signs.items():
        n = s.get("n_points", 0)
        methods = s.get("methods", {})

        if n <= 0:
            no_points += 1
            continue
        k_ge7 += int(n >= 7)
        k_lt7 += int(0 < n < 7)

        have = set(methods.keys())
        null_est = any(methods.get(m, {}).get("est_lat") is None for m in order)
        if have != want or null_est:
            broken += 1
            reason = "missing " + ",".join(want - have) if have != want else "null est"
            broken_signs.append((key, reason, n))
            continue
        if any(methods[m].get("error_m") is None for m in order):
            no_gt += 1
        complete += 1

    print(f"methods expected per sign : {len(order)}  {order}")
    print(f"total signs               : {len(signs)}")
    print(f"complete (all methods)    : {complete}")
    print(f"  with k>=7 (sparse runs) : {k_ge7}")
    print(f"  with k<7  (k-guard)     : {k_lt7}")
    print(f"complete but no GT error  : {no_gt}")
    print(f"no location points (skip) : {no_points}")
    print(f"BROKEN (points but gaps)  : {broken}")
    for key, reason, n in broken_signs[:20]:
        print(f"    - {key}  ({reason}, n_points={n})")

    if broken:
        print("\nFAIL: some signs with points are missing per-method results.")
        return 1
    print("\nOK: every sign with location points has all per-method results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
