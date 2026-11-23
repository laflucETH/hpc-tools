#!/usr/bin/env python3
"""Compare the affinity scores of two or more Nesso-1 runs over the same inputs.

Used to decide whether the image scores the same as the virtual environment the reference scores
were produced in. It cannot be answered by asking for identical numbers: Nesso-1 on a GPU is not
deterministic even with a fixed seed and byte-identical inputs. Three runs of the SAME venv over
the same 24 pairs disagreed by up to 0.254 in affinity_pred_value (README, "Scores are not
reproducible between runs"). Demanding an exact match would therefore fail for a reason that has
nothing to do with the image.

The question is instead: is the image-against-venv disagreement any larger than the disagreement
of the same software with itself, measured on the same GPU in the same job?

    compare_runs.py --run venv=<dir> --run image_a=<dir> --run image_b=<dir> \
                    --baseline image_a:image_b --out comparison.csv

Each <dir> is a `nesso predict --out_dir` directory, or the predictions/ directory inside it.

The statistical core here is deliberately the same as boltz2/compare_runs.py beside it, including
the paired design and the reason for it; only the fields and the output layout differ.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# The scalars nesso writes per pair. affinity_pred_value is the one the scores table is read on:
# the ensemble mean of log10(IC50 / uM), where LOWER means stronger predicted binding.
FIELDS = ("affinity_pred_value", "affinity_pred_value1", "affinity_pred_value2",
          "affinity_probability_binary", "affinity_logits_binary",
          "entropy_crop_pl", "entropy_crop_pp", "entropy_crop_ll",
          "entropy_pp", "entropy_pl", "entropy_ll")

# The field the verdict is decided on. The others are reported because a disagreement confined to
# one of them would say something about where the two environments part company, but the scores
# table is read on this one and it is what a shortlist gets ranked by.
HEADLINE = "affinity_pred_value"


def predictions_dir(path: Path) -> Path:
    if path.name == "predictions":
        return path
    if (path / "predictions").is_dir():
        return path / "predictions"
    sys.exit(f"FATAL: no predictions directory under {path}")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    root = predictions_dir(path)
    out: dict[str, dict[str, float]] = {}
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        aff = rec / "affinity.json"
        if not aff.is_file():
            continue
        payload = json.loads(aff.read_text())
        out[rec.name] = {k: payload[k] for k in FIELDS
                         if isinstance(payload.get(k), (int, float))}
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
    # between pairs -- a long protein is noisy in both runs -- and an unpaired test would drown in
    # that spread. (On the 24-pair example, HSP90 at 732 aa produced 5 of the 6 largest run-to-run
    # differences, so the spread across pairs is roughly an order of magnitude.)
    def spread(x: str, y: str, field: str) -> list[float]:
        return [abs(runs[x][p][field] - runs[y][p][field])
                for p in shared if field in runs[x][p] and field in runs[y][p]]

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
        print("\n(scipy not available: reporting the spreads without a test)")

    def test(cross: list[float], base: list[float]) -> float:
        if wilcoxon is None or not any(c != b for c, b in zip(cross, base)):
            return float("nan")
        try:
            return float(wilcoxon(cross, base).pvalue)
        except ValueError:                              # all differences zero
            return float("nan")

    others = [n for n in names if n not in (base_a, base_b)]
    verdict_ok = True
    for other in others:
        print(f"\n=== {other} against {base_a}, yardstick {base_a} against {base_b} "
              f"({len(shared)} pairs) ===")
        print(f"{'field':<28}{'median cross':>14}{'median same':>13}{'max cross':>11}"
              f"{'max same':>10}{'p':>8}")
        print("-" * 84)
        head_cross: list[float] = []
        head_base: list[float] = []
        for field in FIELDS:
            base = spread(base_a, base_b, field)
            cross = spread(other, base_a, field)
            if not base or not cross:
                continue
            if field == HEADLINE:
                head_cross, head_base = cross, base
            print(f"{field:<28}{statistics.median(cross):>14.5f}"
                  f"{statistics.median(base):>13.5f}{max(cross):>11.4f}"
                  f"{max(base):>10.4f}{test(cross, base):>8.3f}")
        print("-" * 84)
        if not head_cross:
            print(f"WARNING: no {HEADLINE} in common, so there is no verdict to give")
            continue

        # The decision rests on affinity_pred_value alone. The eleven per-field p-values above are
        # descriptive: eleven uncorrected tests would throw up a "significant" one by chance often
        # enough on their own, and the entropies are not what anyone ranks a shortlist by.
        p_head = test(head_cross, head_base)
        print(f"n = {len(head_cross)} paired comparisons on {HEADLINE}")

        # A second, different question. Everything above is built on ABSOLUTE differences, so it
        # asks whether the two environments land further apart. It cannot see a systematic shift:
        # if one of them scored every pair 0.05 lower, each |difference| would be a tidy 0.05 and
        # nothing above would stir -- yet a shortlist built from a mixture of the two would be
        # biased. The sign test asks that question directly, and the same-software row is again
        # the yardstick: it should sit near half.
        def signed(x: str, y: str) -> list[float]:
            return [runs[x][p][HEADLINE] - runs[y][p][HEADLINE]
                    for p in shared if HEADLINE in runs[x][p] and HEADLINE in runs[y][p]]

        try:
            from scipy.stats import binomtest
        except ImportError:
            binomtest = None
        p_sign = {}
        print(f"\nsigned differences on {HEADLINE} -- is one systematically higher?")
        for label, key, d in ((f"{other} - {base_a}", "cross", signed(other, base_a)),
                              (f"{base_b} - {base_a}", "same", signed(base_b, base_a))):
            pos = sum(v > 0 for v in d)
            p_sign[key] = (float(binomtest(pos, len(d)).pvalue)
                           if binomtest is not None and d else float("nan"))
            print(f"  {label:<26}{pos:>3} of {len(d)} above, mean {sum(d) / len(d):+.5f}, "
                  f"sign test p = {p_sign[key]:.3f}")

        shifted = (p_sign.get("cross", float("nan")) < 0.05
                   and not p_sign.get("same", float("nan")) < 0.05)
        if shifted:
            verdict_ok = False
            print(f"\n{other} is systematically offset from {base_a} (sign test "
                  f"p = {p_sign['cross']:.4f}) while {base_b} is not. A shift in one direction "
                  f"biases anything ranked from a mixture of the two, even where the spreads "
                  f"match, so this must be explained before they are treated as one tool.")

        larger = statistics.median(head_cross) > statistics.median(head_base)
        if p_head == p_head and p_head < 0.05 and larger:
            verdict_ok = False
            print(f"\n{other} differs from {base_a} by MORE than {base_a} differs from {base_b} "
                  f"(p = {p_head:.4f}). That must be explained before scores from the two are "
                  f"pooled or compared.")
        elif not shifted:
            print(f"\nNo difference shown: {other} sits as close to {base_a} as {base_a} sits to "
                  f"{base_b} (p = {p_head:.3f}), and neither is offset from the other. Absence of "
                  f"evidence over {len(shared)} pairs is not proof the two are identical -- a "
                  f"difference smaller than the run-to-run noise cannot be resolved by this "
                  f"many pairs -- but nothing here argues otherwise.")

    print(f"\nWhat the numbers mean for a scores table: {HEADLINE} is log10(IC50 / uM), so a\n"
          f"difference of 0.3 is a factor of two in predicted IC50, and a run-to-run difference\n"
          f"of 0.25 on one pair is the same order as the gap between neighbouring pairs in a\n"
          f"within-protein ranking. That is why the yardstick column, not zero, is what\n"
          f"'cross' is read against.")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
