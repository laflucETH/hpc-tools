#!/usr/bin/env python3
"""Drive a Nesso-1 protein-ligand affinity run on a SLURM cluster.

Nesso-1 scores one (protein sequence, ligand SMILES) pair at a time and writes, per pair, an
input YAML, a structure .npz, a record .json and a predictions/<id>/affinity.json -- about five
filesystem entries. At a few thousand pairs that is untidy; at the few hundred thousand pairs an
all-against-all screen needs it exhausts a file-count (inode) quota long before it fills any
disk. This driver therefore keeps every per-pair file on scratch, and collapses each shard into
one table plus one zip as soon as that shard succeeds, so the peak file count scales with the
number of shards running at once rather than with the size of the run.

Three subcommands, in the order a run uses them:

    plan      a pairs table -> per-pair input YAMLs on scratch, grouped into shards
    collect   one finished shard -> one CSV + one zip of raw outputs; then clear its scratch tree
    finalize  all shard CSVs -> one scores table in project storage, with manifest and provenance

The input contract is deliberately small so any project can use it. `--pairs` is a CSV with

    pair_id             unique, and usable as a filename: [A-Za-z0-9][A-Za-z0-9._-]*
    protein_sequence    one-letter amino acids
    ligand_smiles       SMILES of the ligand whose affinity is predicted

Any other column is carried through to the output untouched. One optional column is understood:
`extra_ligands`, a ';'-separated list of further ligands placed in the complex as their own
chains. Each entry is either a SMILES or `ccd:<CODE>` for a Chemical Component Dictionary code,
which is how a metal ion is given beside the macrocycle it sits in (`ccd:FE2`). Affinity is
always predicted against the ligand in `ligand_smiles`.

No row is ever dropped. A pair that cannot be built, or that Nesso-1 fails to score, keeps its row
in the final table with a `nesso:status` saying why. A run whose output table has fewer rows than
its input has lost something silently, which is the failure this rule exists to make impossible.

Deleting a shard's scratch tree is guarded twice: it happens only when the prediction step exited
cleanly (`--allow-cleanup`, which the job script passes only on success), and only inside a run
directory that resolves under the scratch root. A failed shard keeps every file, because a failure
is exactly when they need to be readable.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:                                     # pragma: no cover
    sys.exit("PyYAML is required; run this with the nesso virtual environment's python")

# --- contract ---------------------------------------------------------------------------------
REQUIRED_COLUMNS = ("pair_id", "protein_sequence", "ligand_smiles")
EXTRA_LIGAND_COLUMN = "extra_ligands"
CCD_PREFIX = "ccd:"
PAIR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_CHAIN = "A"
LIGAND_CHAIN = "B"                                      # the binder; extras get C, D, ...
CHAIN_ALPHABET = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

# Scalar fields of nesso's affinity.json, in the order the output table carries them. Written as
# `nesso:<field>` so a downstream join can tell them from another tool's scores at a glance.
AFFINITY_FIELDS = (
    "affinity_pred_value",          # ensemble mean, log10(IC50 / uM); LOWER is stronger binding
    "affinity_pred_value1",         # ensemble member 1
    "affinity_pred_value2",         # ensemble member 2
    "affinity_probability_binary",  # P(binder), 0-1; HIGHER is more likely a binder
    "affinity_logits_binary",       # the logit behind the probability
    "entropy_crop_pl",              # protein-ligand distogram entropy on the cropped interface;
                                    # the model card calls this the structural confidence to read,
                                    # and 0.0 means the ligand could not be placed confidently
    "entropy_crop_pp",
    "entropy_crop_ll",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
)
STATUS_COLUMN = "nesso:status"
SHARD_COLUMN = "nesso:shard"

SHARD_FMT = "shard_%04d"
DEFAULT_PAIRS_PER_SHARD = 2000


# --- small helpers ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def fail(msg: str) -> None:
    sys.exit(f"FATAL: {msg}")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scratch_roots() -> list[Path]:
    """Directories a run may live in, and so the only ones cleanup may delete inside.

    Defaults to the account's personal scratch. NESSO_SCRATCH_ROOT overrides it for another site
    or for the tests, which must be able to run on a temporary directory.
    """
    env = os.environ.get("NESSO_SCRATCH_ROOT")
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p]
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    return [Path(f"/cluster/scratch/{user}").resolve()]


def under_any(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def shard_name(i: int) -> str:
    return SHARD_FMT % i


def run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "plan": run_dir / "plan",
        "inputs": run_dir / "inputs",
        "work": run_dir / "work",
        "shards": run_dir / "shards",
        "logs": run_dir / "logs",
    }


def read_csv_rows(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        return rows, list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)                                   # atomic: a reader never sees half a table


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + (" (dirty)" if dirty else "")
    except Exception:
        pass
    return "unknown"


# --- plan -------------------------------------------------------------------------------------
def validate_pair(row: dict, seen: set[str]) -> str | None:
    """Return None if the row can be folded, else the reason it cannot."""
    pid = (row.get("pair_id") or "").strip()
    if not pid:
        return "invalid_pair_id: empty"
    if not PAIR_ID_RE.match(pid):
        return "invalid_pair_id: must match [A-Za-z0-9][A-Za-z0-9._-]*"
    if pid in seen:
        return "duplicate_pair_id"
    seq = (row.get("protein_sequence") or "").strip().upper()
    if not seq:
        return "no_protein_sequence"
    bad = sorted(set(seq) - STANDARD_AA)
    if bad:
        # ESM-2 would map these to <unk> and nesso's tokenizer may reject them; rather than let a
        # silently degraded embedding through, the pair is kept out and the reason recorded.
        return f"non_standard_residue: {''.join(bad)}"
    if not (row.get("ligand_smiles") or "").strip():
        return "no_ligand_smiles"

    # A complex holding the same molecule twice cannot be scored by Nesso-1 1.0.0, so the pair is
    # kept out here rather than failing a GPU job hours later. The cause is upstream, in
    # nesso/data/yaml_input.py: identical ligand entities are grouped, and the group's RDKit
    # conformer is written to one pickle named after its FIRST chain id only (line 622,
    # "use the first chain id as pkl name"), while inference asks for a pickle per ligand chain
    # (nesso/data/inference.py, _setup_molecules) and raises "No pkl for ligand chain '<id>'" for
    # the second copy. Both the two-entry form and the documented `id: [B, C]` form group the
    # same way, so no YAML spelling avoids it. Measured on ferric dicitrate, two citrates around
    # one iron, 2026-11-17.
    entities = [(row.get("ligand_smiles") or "").strip()]
    entities += [s.strip() for s in (row.get(EXTRA_LIGAND_COLUMN) or "").split(";") if s.strip()]
    if len(entities) != len(set(entities)):
        return "repeated_ligand_entity: nesso 1.0.0 writes one conformer per distinct molecule"
    return None


def yaml_for_pair(row: dict) -> dict:
    seq = row["protein_sequence"].strip().upper()
    sequences: list[dict] = [{"protein": {"id": PROTEIN_CHAIN, "sequence": seq}}]
    ligands = [row["ligand_smiles"].strip()]
    extra = (row.get(EXTRA_LIGAND_COLUMN) or "").strip()
    if extra:
        ligands += [s.strip() for s in extra.split(";") if s.strip()]
    if len(ligands) > len(CHAIN_ALPHABET):
        raise ValueError(f"too many ligand entities ({len(ligands)})")

    for chain, entry in zip(CHAIN_ALPHABET, ligands):
        # A ccd: entry names a Chemical Component Dictionary code, which is how a bare metal ion
        # is placed beside the macrocycle it belongs to; anything else is read as SMILES.
        if entry.lower().startswith(CCD_PREFIX):
            sequences.append({"ligand": {"id": chain, "ccd": entry[len(CCD_PREFIX):].strip().upper()}})
        else:
            sequences.append({"ligand": {"id": chain, "smiles": entry}})
    return {"sequences": sequences,
            "properties": [{"affinity": {"binder": LIGAND_CHAIN}}]}


def cmd_plan(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    pairs_csv = Path(args.pairs).expanduser().resolve()
    if not pairs_csv.is_file():
        fail(f"no such pairs table: {pairs_csv}")
    if not args.allow_any_path and not under_any(run_dir, scratch_roots()):
        fail(f"run directory is outside the scratch root(s) {[str(r) for r in scratch_roots()]}: "
             f"{run_dir}\n       per-pair files must not be written to project storage; pass "
             f"--allow-any-path only for a test")

    rows, fieldnames = read_csv_rows(pairs_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        fail(f"pairs table is missing required column(s): {', '.join(missing)}")
    if not rows:
        fail("pairs table has no rows")

    paths = run_paths(run_dir)
    if paths["plan"].exists() and not args.override:
        fail(f"a plan already exists at {paths['plan']}; pass --override to replace it")
    if args.override and paths["plan"].exists():
        # Replacing a plan has to take the previous run's OUTPUT with it. A shard whose table is
        # already present is skipped by the submit script, so leaving stale tables behind means
        # the next run quietly finalizes the old scores against the new plan -- and if the pair
        # ids happen to match, the mismatch is invisible in the output.
        stale = [p for p in (paths["inputs"], paths["work"], paths["shards"]) if p.is_dir()
                 and any(p.iterdir())]
        if stale and not under_any(run_dir, scratch_roots()):
            fail(f"--override would have to delete {[str(p) for p in stale]}, which is not under "
                 f"{[str(r) for r in scratch_roots()]}; clear it yourself, or plan into a new "
                 f"run directory")
        for victim in stale:
            shutil.rmtree(victim)
            log(f"--override: cleared the previous run's {victim.name}/")
    for p in (paths["plan"], paths["inputs"], paths["work"], paths["shards"], paths["logs"]):
        p.mkdir(parents=True, exist_ok=True)

    # Validate first, so a bad table is rejected before any file is written.
    seen: set[str] = set()
    foldable: list[dict] = []
    for row in rows:
        reason = validate_pair(row, seen)
        row[STATUS_COLUMN] = "planned" if reason is None else f"skipped: {reason}"
        if reason is None:
            seen.add(row["pair_id"].strip())
            foldable.append(row)

    # Shard by protein. Pairs are sorted by sequence so each protein's pairs stay together, which
    # means its ESM-2 embedding is computed once per shard instead of once per shard it appears
    # in. At most one protein straddles any boundary. Sorting is by sequence rather than by any
    # name column so the grouping does not depend on a caller's naming.
    foldable.sort(key=lambda r: (r["protein_sequence"].strip().upper(), r["pair_id"].strip()))
    per_shard = max(1, int(args.pairs_per_shard))
    for i, row in enumerate(foldable):
        row[SHARD_COLUMN] = shard_name(i // per_shard)

    n_shards = (len(foldable) + per_shard - 1) // per_shard if foldable else 0
    log(f"{len(rows)} input rows: {len(foldable)} foldable, {len(rows) - len(foldable)} skipped")
    log(f"{n_shards} shard(s) of at most {per_shard} pairs")

    # Write the input YAMLs.
    written = 0
    for row in foldable:
        shard_dir = paths["inputs"] / row[SHARD_COLUMN]
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            doc = yaml_for_pair(row)
        except ValueError as e:
            row[STATUS_COLUMN] = f"skipped: {e}"
            row.pop(SHARD_COLUMN, None)
            continue
        with open(shard_dir / f"{row['pair_id'].strip()}.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        written += 1
    log(f"wrote {written} input YAML(s) under {paths['inputs']}")

    out_fields = list(fieldnames)
    for c in (STATUS_COLUMN, SHARD_COLUMN):
        if c not in out_fields:
            out_fields.append(c)
    write_csv_rows(paths["plan"] / "pairs.csv", rows, out_fields)

    counts: dict[str, dict] = {}
    for row in foldable:
        s = row.get(SHARD_COLUMN)
        if not s:
            continue
        c = counts.setdefault(s, {"shard": s, "n_pairs": 0, "proteins": set()})
        c["n_pairs"] += 1
        c["proteins"].add(row["protein_sequence"].strip().upper())
    shard_rows = [{"shard": c["shard"], "n_pairs": c["n_pairs"], "n_proteins": len(c["proteins"])}
                  for c in sorted(counts.values(), key=lambda c: c["shard"])]
    write_csv_rows(paths["plan"] / "shards.csv", shard_rows, ["shard", "n_pairs", "n_proteins"])

    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "pairs_csv": str(pairs_csv), "pairs_csv_md5": md5_file(pairs_csv),
            "n_input_rows": len(rows), "n_foldable": len(foldable), "n_shards": len(shard_rows),
            "pairs_per_shard": per_shard, "run_dir": str(run_dir),
            "driver_commit": git_commit(Path(__file__).resolve().parent)}
    (paths["plan"] / "plan.json").write_text(json.dumps(meta, indent=2) + "\n")

    if shard_rows:
        lo, hi = shard_rows[0]["shard"], shard_rows[-1]["shard"]
        log(f"submit with --array=0-{len(shard_rows) - 1}  (shards {lo}..{hi})")
    return 0


# --- collect ----------------------------------------------------------------------------------
def cmd_collect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    paths = run_paths(run_dir)
    shard = args.shard if args.shard.startswith("shard_") else shard_name(int(args.shard))

    plan_csv = paths["plan"] / "pairs.csv"
    if not plan_csv.is_file():
        fail(f"no plan at {plan_csv}; run `plan` first")
    rows, fieldnames = read_csv_rows(plan_csv)
    planned = [r for r in rows if r.get(SHARD_COLUMN) == shard]
    if not planned:
        fail(f"the plan has no pairs for {shard}")

    pred_dir = paths["work"] / shard / "predictions"
    out_csv = paths["shards"] / f"{shard}.csv"
    out_zip = paths["shards"] / f"{shard}.zip"
    paths["shards"].mkdir(parents=True, exist_ok=True)

    # Gather every affinity.json this shard produced.
    found: dict[str, dict] = {}
    if pred_dir.is_dir():
        for rec in sorted(pred_dir.iterdir()):
            aff = rec / "affinity.json"
            if not aff.is_file():
                continue
            try:
                found[rec.name] = json.loads(aff.read_text())
            except Exception as e:
                log(f"  unreadable {aff}: {type(e).__name__}: {e}")

    out_fields = list(fieldnames)
    for c in [f"nesso:{f}" for f in AFFINITY_FIELDS]:
        if c not in out_fields:
            out_fields.append(c)

    n_scored = 0
    for row in planned:
        pid = row["pair_id"].strip()
        payload = found.get(pid)
        if payload is None:
            row[STATUS_COLUMN] = "no_prediction"
            continue
        for f in AFFINITY_FIELDS:
            row[f"nesso:{f}"] = payload.get(f, "")
        row[STATUS_COLUMN] = "scored" if payload.get("affinity_pred_value") is not None \
            else "no_affinity_value"
        n_scored += 1

    write_csv_rows(out_csv, planned, out_fields)
    log(f"{shard}: {n_scored} of {len(planned)} pair(s) scored -> {out_csv}")

    # Zip the raw affinity.json files. The CSV already carries every scalar; the zip keeps the
    # untouched model output so a later question about a field that was not extracted can still
    # be answered. Zip rather than a solid tar so one member stays randomly accessible.
    n_zipped = 0
    if args.keep_raw and found:
        tmp_zip = out_zip.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pid in sorted(found):
                zf.write(pred_dir / pid / "affinity.json", arcname=f"{pid}/affinity.json")
                n_zipped += 1
        tmp_zip.replace(out_zip)

    # Verify before anything is deleted: member count must match what was found, and the archive
    # must pass its own integrity check. Deleting originals on an unverified archive is how an
    # irrecoverable loss happens.
    verified = True
    if args.keep_raw and found:
        with zipfile.ZipFile(out_zip) as zf:
            members = zf.namelist()
            if len(members) != len(found):
                log(f"  VERIFY FAILED: zip holds {len(members)}, expected {len(found)}")
                verified = False
            elif zf.testzip() is not None:
                log(f"  VERIFY FAILED: corrupt member {zf.testzip()}")
                verified = False
        if verified:
            log(f"  zip verified: {n_zipped} member(s), {out_zip.stat().st_size / 1e6:.1f} MB")

    if not args.allow_cleanup:
        log("  scratch tree kept (cleanup not requested)")
        return 0 if n_scored else 1
    if not verified:
        log("  scratch tree KEPT because the archive did not verify")
        return 1
    if n_scored < len(planned):
        missing = len(planned) - n_scored
        if not args.accept_unscored:
            log(f"  scratch tree KEPT: {missing} pair(s) unscored, so the shard is "
                f"not finished and its files are needed to debug it")
            return 1
        # Accepting is a deliberate act for a failure that has already been diagnosed: the rows
        # keep their no_prediction status in the table, so the gap stays visible in the output
        # even though the files behind it are cleared.
        log(f"  {missing} pair(s) unscored, accepted on request; they stay in the table as "
            f"no_prediction")
    if not under_any(run_dir, scratch_roots()):
        log(f"  scratch tree KEPT: {run_dir} is not under {[str(r) for r in scratch_roots()]}")
        return 1

    for victim in (paths["work"] / shard, paths["inputs"] / shard):
        if victim.is_dir():
            shutil.rmtree(victim)
            log(f"  cleared {victim}")
    return 0


# --- finalize ---------------------------------------------------------------------------------
def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    paths = run_paths(run_dir)

    plan_csv = paths["plan"] / "pairs.csv"
    if not plan_csv.is_file():
        fail(f"no plan at {plan_csv}")
    plan_rows, plan_fields = read_csv_rows(plan_csv)
    by_id = {r["pair_id"].strip(): r for r in plan_rows}

    shard_csvs = sorted(paths["shards"].glob("shard_*.csv"))
    if not shard_csvs:
        fail(f"no shard tables under {paths['shards']}; nothing to finalize")

    out_fields = list(plan_fields)
    merged: dict[str, dict] = {}
    for sc in shard_csvs:
        rows, fields = read_csv_rows(sc)
        for c in fields:
            if c not in out_fields:
                out_fields.append(c)
        for r in rows:
            merged[r["pair_id"].strip()] = r

    # Every planned pair must come back. One that does not is reported as unscored rather than
    # quietly left out, so the output table always has exactly as many rows as the input.
    final_rows: list[dict] = []
    for pid, prow in by_id.items():
        row = merged.get(pid)
        if row is None:
            row = dict(prow)
            if row.get(STATUS_COLUMN) == "planned":
                row[STATUS_COLUMN] = "shard_not_collected"
        final_rows.append(row)

    status_counts: dict[str, int] = {}
    for r in final_rows:
        status_counts[r.get(STATUS_COLUMN, "")] = status_counts.get(r.get(STATUS_COLUMN, ""), 0) + 1

    dest.mkdir(parents=True, exist_ok=True)
    label = args.label or run_dir.name
    scores_csv = dest / f"{label}_nesso_scores.csv"
    write_csv_rows(scores_csv, final_rows, out_fields)

    # One zip for the whole run, built from the verified per-shard zips.
    raw_zip = dest / f"{label}_nesso_raw.zip"
    manifest_rows: list[dict] = []
    shard_zips = sorted(paths["shards"].glob("shard_*.zip"))
    if shard_zips:
        tmp = raw_zip.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for sz in shard_zips:
                with zipfile.ZipFile(sz) as src:
                    for m in src.namelist():
                        out.writestr(m, src.read(m))
                        manifest_rows.append({"member": m, "pair_id": m.split("/")[0],
                                              "from_shard": sz.stem})
        tmp.replace(raw_zip)
        with zipfile.ZipFile(raw_zip) as zf:
            if len(zf.namelist()) != len(manifest_rows) or zf.testzip() is not None:
                fail(f"the merged archive {raw_zip} did not verify")
        log(f"raw archive: {len(manifest_rows)} member(s), "
            f"{raw_zip.stat().st_size / 1e6:.1f} MB -> {raw_zip}")
        write_csv_rows(dest / f"{label}_nesso_raw_manifest.csv", manifest_rows,
                       ["member", "pair_id", "from_shard"])

    plan_meta = {}
    pj = paths["plan"] / "plan.json"
    if pj.is_file():
        plan_meta = json.loads(pj.read_text())

    prov = [
        f"Nesso-1 run: {label}",
        f"finalized      : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"run directory  : {run_dir}",
        f"pairs table    : {plan_meta.get('pairs_csv', 'unknown')}",
        f"pairs md5      : {plan_meta.get('pairs_csv_md5', 'unknown')}",
        f"planned        : {plan_meta.get('n_foldable', 'unknown')} of "
        f"{plan_meta.get('n_input_rows', 'unknown')} input rows, in "
        f"{plan_meta.get('n_shards', 'unknown')} shard(s)",
        f"driver commit  : {plan_meta.get('driver_commit', 'unknown')}",
        f"nesso version  : {args.nesso_version or os.environ.get('NESSO_VERSION', 'unknown')}",
        f"model revision : {args.model_revision or os.environ.get('NESSO_MODEL_REVISION', 'unknown')}",
        f"seed           : {args.seed if args.seed is not None else os.environ.get('NESSO_SEED', 'unknown')}",
        f"GPU type       : {args.gpu or os.environ.get('NESSO_GPU', 'unknown')}",
        "",
        f"output table   : {scores_csv.name} ({len(final_rows)} rows)",
        f"table md5      : {md5_file(scores_csv)}",
        "",
        "status counts:",
    ]
    prov += [f"  {k or '(blank)'}: {v}" for k, v in sorted(status_counts.items())]
    prov += [
        "",
        "Reading the scores:",
        "  nesso:affinity_pred_value        log10(IC50 / uM); LOWER means stronger predicted binding",
        "  nesso:affinity_probability_binary P(binder) in 0-1; HIGHER means more likely a binder",
        "  nesso:entropy_crop_pl            protein-ligand interface distogram entropy; 0.0 means",
        "                                   the ligand could not be placed confidently and the",
        "                                   prediction for that pair should not be trusted",
    ]
    (dest / f"{label}_provenance.txt").write_text("\n".join(prov) + "\n")

    log(f"{len(final_rows)} row(s) -> {scores_csv}")
    for k, v in sorted(status_counts.items()):
        log(f"  {k or '(blank)'}: {v}")
    if len(final_rows) != len(plan_rows):
        log(f"WARNING: {len(final_rows)} output rows but {len(plan_rows)} planned rows")
        return 1
    return 0


# --- cli --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="pairs table -> per-pair YAMLs on scratch, grouped in shards")
    p.add_argument("--pairs", required=True, help="CSV with pair_id, protein_sequence, ligand_smiles")
    p.add_argument("--run-dir", required=True, help="run directory on scratch")
    p.add_argument("--pairs-per-shard", type=int, default=DEFAULT_PAIRS_PER_SHARD)
    p.add_argument("--override", action="store_true",
                   help="replace an existing plan, deleting the previous run's inputs, work and "
                        "shard tables with it")
    p.add_argument("--allow-any-path", action="store_true",
                   help="permit a run directory outside scratch (tests only)")
    p.set_defaults(func=cmd_plan)

    c = sub.add_parser("collect", help="one finished shard -> CSV + zip, then clear its scratch tree")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--shard", required=True, help="shard_0000 or 0")
    c.add_argument("--allow-cleanup", action="store_true",
                   help="delete the shard's scratch tree once its archive verifies")
    c.add_argument("--no-keep-raw", dest="keep_raw", action="store_false",
                   help="do not zip the raw affinity.json files")
    c.add_argument("--accept-unscored", action="store_true",
                   help="clear the scratch tree even though some pairs went unscored; for a "
                        "failure already diagnosed, whose rows stay no_prediction in the table")
    c.set_defaults(func=cmd_collect, keep_raw=True)

    f = sub.add_parser("finalize", help="shard tables -> one scores table in project storage")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--dest", required=True, help="destination directory, in project storage")
    f.add_argument("--label", default=None, help="prefix for the output files (default: run name)")
    f.add_argument("--nesso-version", default=None)
    f.add_argument("--model-revision", default=None)
    f.add_argument("--seed", default=None)
    f.add_argument("--gpu", default=None)
    f.set_defaults(func=cmd_finalize)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
