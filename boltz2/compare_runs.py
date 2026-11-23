#!/usr/bin/env python3
"""Compare the confidence scores of two or more Boltz-2 runs over the same inputs.

Used to decide whether the image reproduces the environment the reference folds ran in. It cannot
be answered by asking for identical numbers: Boltz-2 on a GPU is not deterministic even with a
fixed seed and identical inputs, so two runs of the SAME software disagree a little. The question
is therefore whether the image-against-venv disagreement is any larger than the disagreement of
one of them with itself.

    compare_runs.py --run venv=<dir> --run image_a=<dir> --run image_b=<dir> \
                    --baseline image_a:image_b --out comparison.csv

Each <dir> is a `boltz predict --out_dir` directory, or the predictions/ directory inside it.
Pairs present in every run are compared on the top-ranked sample (model_0).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

FIELDS = ("confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
          "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")


def predictions_dir(path: Path) -> Path:
    if path.name == "predictions":
        return path
    hits = sorted(path.glob("boltz_results_*/predictions"))
    if hits:
        return hits[0]
    if (path / "predictions").is_dir():
        return path / "predictions"
    sys.exit(f"FATAL: no predictions directory under {path}")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    root = predictions_dir(path)
    out: dict[str, dict[str, float]] = {}
    for pred in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pred.name
        f = pred / f"confidence_{pid}_model_0.json"
        if f.is_file():
            payload = json.loads(f.read_text())
            out[pid] = {k: payload[k] for k in FIELDS if isinstance(payload.get(k), (int, float))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="a named run directory; give it at least twice")
    ap.add_argument("--baseline", required=True, metavar="A:B",
                    help="the two runs whose disagreement is the yardstick (same software twice)")
    ap.add_argument("--out", required=True, help="per-pair comparison table")
    a = ap.parse_args(argv)

    runs: dict[str, dict] = {}
    for spec in a.run:
        name, _, d = spec.partition("=")
        if not d:
            sys.exit(f"FATAL: --run takes NAME=DIR, got {spec!r}")
        runs[name] = read_run(Path(d).expanduser().resolve())
        print(f"{name}: {len(runs[name])} prediction(s)")
    if len(runs) < 2:
        sys.exit("FATAL: give at least two runs")
    base_a, _, base_b = a.baseline.partition(":")
    for n in (base_a, base_b):
        if n not in runs:
            sys.exit(f"FATAL: --baseline names {n!r}, which is not one of {sorted(runs)}")

    shared = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not shared:
        sys.exit("FATAL: the runs share no pairs")
    missing = {n: sorted(set(shared) ^ set(r)) for n, r in runs.items() if set(r) != set(shared)}
    for n, m in missing.items():
        print(f"WARNING: {n} differs on {len(m)} pair(s), which are left out: {m[:5]}")

    names = list(runs)
    rows = []
    for pid in shared:
        for field in FIELDS:
            if any(field not in runs[n][pid] for n in names):
                continue
            row = {"pair_id": pid, "field": field}
            row.update({n: runs[n][pid][field] for n in names})
            rows.append(row)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "field"] + names, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(shared)} pair(s) compared -> {a.out}")

    # --- the verdict ---------------------------------------------------------------------------
    # The yardstick is how far the same software lands from itself on the same GPU, pair by pair.
    # The comparison is then paired: for each (pair, field), the cross-environment difference sits
    # beside the same-software difference for that very pair, and a Wilcoxon signed-rank test asks
    # whether one is systematically larger. Pairing matters because the differences vary enormously
    # between pairs -- a hard target is noisy in both runs -- and an unpaired test would drown in
    # that spread.
    #
    # An earlier version of this script compared the two MAXIMA instead. That was a mistake worth
    # recording: the maximum of a dozen values is the noisiest summary available, it moves with a
    # single observation, and on the first real run it declared 3 of 9 fields "larger" purely by
    # chance while the medians and the paired test showed nothing at all.
    def spread(x: str, y: str, field: str) -> list[float]:
        return [abs(runs[x][p][field] - runs[y][p][field])
                for p in shared if field in runs[x][p] and field in runs[y][p]]

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
        print("\n(scipy not available: reporting the spreads without a test)")

    others = [n for n in names if n not in (base_a, base_b)]
    verdict_ok = True
    for other in others:
        print(f"\n=== {other} against {base_a}, yardstick {base_a} against {base_b} "
              f"({len(shared)} pairs) ===")
        print(f"{'field':<17}{'median cross':>14}{'median same':>13}{'max cross':>11}"
              f"{'max same':>10}{'p':>8}")
        print("-" * 73)
        pooled_cross: list[float] = []
        pooled_base: list[float] = []
        for field in FIELDS:
            base = spread(base_a, base_b, field)
            cross = spread(other, base_a, field)
            if not base or not cross:
                continue
            pooled_cross += cross
            pooled_base += base
            p = float("nan")
            if wilcoxon is not None and any(c != b for c, b in zip(cross, base)):
                try:
                    p = wilcoxon(cross, base).pvalue
                except ValueError:                      # all differences zero
                    p = float("nan")
            print(f"{field:<17}{statistics.median(cross):>14.5f}"
                  f"{statistics.median(base):>13.5f}{max(cross):>11.4f}"
                  f"{max(base):>10.4f}{p:>8.3f}")
        print("-" * 73)
        if not pooled_cross:
            continue
        p_all = float("nan")
        if wilcoxon is not None and any(c != b for c, b in zip(pooled_cross, pooled_base)):
            try:
                p_all = wilcoxon(pooled_cross, pooled_base).pvalue
            except ValueError:
                p_all = float("nan")
        print(f"{'ALL FIELDS':<17}{statistics.median(pooled_cross):>14.5f}"
              f"{statistics.median(pooled_base):>13.5f}{max(pooled_cross):>11.4f}"
              f"{max(pooled_base):>10.4f}{p_all:>8.3f}")
        print(f"n = {len(pooled_cross)} paired comparisons")

        # The pooled test is the decision; the per-field p-values above are descriptive, and nine
        # of them uncorrected would throw up a "significant" one often enough on their own.
        larger = statistics.median(pooled_cross) > statistics.median(pooled_base)
        if wilcoxon is not None and p_all == p_all and p_all < 0.05 and larger:
            verdict_ok = False
            print(f"\n{other} differs from {base_a} by MORE than {base_a} differs from {base_b} "
                  f"(p = {p_all:.4f}). That must be explained before the two are treated as one "
                  f"tool.")
        else:
            print(f"\nNo difference shown: {other} sits as close to {base_a} as {base_a} sits to "
                  f"{base_b} (p = {p_all:.3f}). Absence of evidence over {len(shared)} pairs is "
                  f"not proof the environments are identical, but nothing here argues otherwise.")

    print("\nNote: a field whose difference is 0.0000 everywhere is computed identically by both,\n"
          "which for protein-only terms is expected -- the sampling that varies between runs is\n"
          "the ligand placement.")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
