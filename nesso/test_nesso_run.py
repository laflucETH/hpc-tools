#!/usr/bin/env python3
"""Tests for nesso_run.py, run without a GPU or a nesso install.

    python3 -m pytest nesso/test_nesso_run.py

The prediction step is the only part that needs a GPU, so it is simulated: the test writes the
affinity.json files nesso would have written and then runs `collect` and `finalize` over them.
What is actually checked is the behaviour the pipeline depends on -- that no input row is lost,
that a shard's scratch tree survives a failure and is cleared only on success, and that cleanup
refuses to delete outside the scratch root.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nesso_run  # noqa: E402

SEQ_A = "MKTAYIAKQRQISFVKSHFSRQ"
SEQ_B = "MVLSPADKTNVKAAWGKVGAHAGEY"


def write_pairs(path: Path, rows: list[dict]) -> None:
    fields = ["pair_id", "protein_sequence", "ligand_smiles", "tf_name", "extra_ligands"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A planned run on a temporary directory that counts as the scratch root."""
    monkeypatch.setenv("NESSO_SCRATCH_ROOT", str(tmp_path))
    pairs = tmp_path / "pairs.csv"
    rows = [
        {"pair_id": "p1", "protein_sequence": SEQ_A, "ligand_smiles": "CCO", "tf_name": "TfA"},
        {"pair_id": "p2", "protein_sequence": SEQ_B, "ligand_smiles": "CCC", "tf_name": "TfB"},
        {"pair_id": "p3", "protein_sequence": SEQ_A, "ligand_smiles": "CCN", "tf_name": "TfA"},
        # kept out of folding, each for a different reason, but still expected in the output
        {"pair_id": "bad id", "protein_sequence": SEQ_A, "ligand_smiles": "C", "tf_name": "TfA"},
        {"pair_id": "p5", "protein_sequence": "MKTXZ", "ligand_smiles": "C", "tf_name": "TfC"},
        {"pair_id": "p6", "protein_sequence": SEQ_A, "ligand_smiles": "", "tf_name": "TfA"},
    ]
    write_pairs(pairs, rows)
    run_dir = tmp_path / "run1"
    rc = nesso_run.main(["plan", "--pairs", str(pairs), "--run-dir", str(run_dir),
                         "--pairs-per-shard", "10"])
    assert rc == 0
    return run_dir, pairs, rows


def fake_predictions(run_dir: Path, shard: str, pair_ids: list[str]) -> None:
    for pid in pair_ids:
        d = run_dir / "work" / shard / "predictions" / pid
        d.mkdir(parents=True, exist_ok=True)
        (d / "affinity.json").write_text(json.dumps({
            "affinity_pred_value": -1.5, "affinity_pred_value1": -1.4,
            "affinity_pred_value2": -1.6, "affinity_probability_binary": 0.77,
            "affinity_logits_binary": 1.2, "entropy_crop_pl": 0.42,
            "entropy_crop_pp": 0.5, "entropy_crop_ll": 0.2,
            "entropy_pp": 0.5, "entropy_pl": 0.4, "entropy_ll": 0.2,
        }))


def test_plan_writes_one_yaml_per_foldable_pair(run):
    run_dir, _, _ = run
    yamls = sorted((run_dir / "inputs").rglob("*.yaml"))
    assert [y.stem for y in yamls] == ["p1", "p2", "p3"]
    plan = read_csv(run_dir / "plan" / "pairs.csv")
    assert len(plan) == 6, "every input row must survive planning"
    status = {r["pair_id"]: r[nesso_run.STATUS_COLUMN] for r in plan}
    assert status["p1"] == "planned"
    assert status["bad id"].startswith("skipped: invalid_pair_id")
    assert status["p5"].startswith("skipped: non_standard_residue")
    assert status["p6"] == "skipped: no_ligand_smiles"


def test_plan_groups_pairs_by_protein(run):
    """Pairs of one protein land in one shard, so its ESM-2 embedding is computed once."""
    run_dir, _, _ = run
    plan = {r["pair_id"]: r for r in read_csv(run_dir / "plan" / "pairs.csv")}
    assert plan["p1"][nesso_run.SHARD_COLUMN] == plan["p3"][nesso_run.SHARD_COLUMN]


def test_yaml_names_the_binder_and_extra_ligands(tmp_path):
    """Extras become their own chains; a ccd: entry becomes a CCD code, not a SMILES."""
    doc = nesso_run.yaml_for_pair({"protein_sequence": SEQ_A, "ligand_smiles": "CCO",
                                   "extra_ligands": "ccd:FE2;O"})
    ligands = [s["ligand"] for s in doc["sequences"] if "ligand" in s]
    assert [lg["id"] for lg in ligands] == ["B", "C", "D"]
    assert ligands[0] == {"id": "B", "smiles": "CCO"}
    assert ligands[1] == {"id": "C", "ccd": "FE2"}, "a metal ion is given as its CCD code"
    assert ligands[2] == {"id": "D", "smiles": "O"}
    assert doc["properties"][0]["affinity"]["binder"] == "B"


def test_a_repeated_ligand_entity_is_kept_out(tmp_path, monkeypatch):
    """A complex holding the same molecule twice is skipped at plan time, not sent to a GPU.

    Nesso-1 1.0.0 writes one RDKit conformer per DISTINCT ligand entity, named after that
    entity's first chain id, but asks for a conformer per ligand CHAIN at inference. The second
    copy therefore has no conformer and the record dies with "No pkl for ligand chain '<id>'".
    Measured on ferric dicitrate, two citrates around one iron.
    """
    citrate = "OC(=O)CC(O)(CC(O)=O)C(O)=O"
    reason = nesso_run.validate_pair(
        {"pair_id": "fe3dcit", "protein_sequence": SEQ_A, "ligand_smiles": citrate,
         "extra_ligands": f"{citrate};ccd:FE"}, set())
    assert reason is not None and reason.startswith("repeated_ligand_entity")


def test_distinct_ligands_are_foldable(tmp_path):
    doc = nesso_run.yaml_for_pair({"protein_sequence": SEQ_A, "ligand_smiles": "CCO",
                                   "extra_ligands": "CCC;ccd:FE2"})
    ligands = [s["ligand"] for s in doc["sequences"] if "ligand" in s]
    assert [lg["id"] for lg in ligands] == ["B", "C", "D"]
    assert nesso_run.validate_pair(
        {"pair_id": "ok", "protein_sequence": SEQ_A, "ligand_smiles": "CCO",
         "extra_ligands": "CCC;ccd:FE2"}, set()) is None


def test_collect_can_accept_a_diagnosed_failure(run):
    """--accept-unscored clears the tree but leaves the gap visible in the table."""
    run_dir, _, _ = run
    shard = "shard_0000"
    fake_predictions(run_dir, shard, ["p1", "p2"])          # p3 never scored
    rc = nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", shard,
                         "--allow-cleanup", "--accept-unscored"])
    assert rc == 0
    assert not (run_dir / "work" / shard).exists()
    rows = {r["pair_id"]: r[nesso_run.STATUS_COLUMN]
            for r in read_csv(run_dir / "shards" / f"{shard}.csv")}
    assert rows["p3"] == "no_prediction", "the unscored pair is still reported, not hidden"


def test_collect_scores_and_clears_only_on_success(run):
    run_dir, _, _ = run
    shard = "shard_0000"
    fake_predictions(run_dir, shard, ["p1", "p2", "p3"])
    rc = nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", shard,
                         "--allow-cleanup"])
    assert rc == 0
    rows = read_csv(run_dir / "shards" / f"{shard}.csv")
    assert {r["pair_id"] for r in rows} == {"p1", "p2", "p3"}
    assert all(r[nesso_run.STATUS_COLUMN] == "scored" for r in rows)
    assert rows[0]["nesso:affinity_pred_value"] == "-1.5"
    assert rows[0]["tf_name"] in {"TfA", "TfB"}, "passthrough columns survive"
    with zipfile.ZipFile(run_dir / "shards" / f"{shard}.zip") as zf:
        assert sorted(zf.namelist()) == ["p1/affinity.json", "p2/affinity.json", "p3/affinity.json"]
    assert not (run_dir / "work" / shard).exists(), "a finished shard's tree is cleared"
    assert not (run_dir / "inputs" / shard).exists()


def test_collect_keeps_the_tree_when_a_pair_is_unscored(run):
    """An incomplete shard is exactly when the per-pair files must stay readable."""
    run_dir, _, _ = run
    shard = "shard_0000"
    fake_predictions(run_dir, shard, ["p1", "p2"])          # p3 missing
    rc = nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", shard,
                         "--allow-cleanup"])
    assert rc == 1
    assert (run_dir / "work" / shard).exists()
    rows = {r["pair_id"]: r[nesso_run.STATUS_COLUMN]
            for r in read_csv(run_dir / "shards" / f"{shard}.csv")}
    assert rows["p3"] == "no_prediction"


def test_cleanup_refuses_outside_the_scratch_root(run, monkeypatch, tmp_path):
    run_dir, _, _ = run
    fake_predictions(run_dir, "shard_0000", ["p1", "p2", "p3"])
    monkeypatch.setenv("NESSO_SCRATCH_ROOT", str(tmp_path / "somewhere_else"))
    rc = nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", "shard_0000",
                         "--allow-cleanup"])
    assert rc == 1
    assert (run_dir / "work" / "shard_0000").exists()


def test_plan_refuses_a_run_dir_outside_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("NESSO_SCRATCH_ROOT", str(tmp_path / "scratch"))
    pairs = tmp_path / "pairs.csv"
    write_pairs(pairs, [{"pair_id": "p1", "protein_sequence": SEQ_A, "ligand_smiles": "CCO"}])
    with pytest.raises(SystemExit):
        nesso_run.main(["plan", "--pairs", str(pairs), "--run-dir", str(tmp_path / "project")])


def test_finalize_returns_every_input_row(run):
    run_dir, pairs, rows = run
    fake_predictions(run_dir, "shard_0000", ["p1", "p2", "p3"])
    nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", "shard_0000",
                    "--allow-cleanup"])
    dest = run_dir.parent / "delivery"
    rc = nesso_run.main(["finalize", "--run-dir", str(run_dir), "--dest", str(dest),
                         "--label", "t1"])
    assert rc == 0
    out = read_csv(dest / "t1_nesso_scores.csv")
    assert len(out) == len(rows), "the output table has one row per input row"
    assert {r["pair_id"] for r in out} == {r["pair_id"] for r in rows}
    by_id = {r["pair_id"]: r for r in out}
    assert by_id["p1"][nesso_run.STATUS_COLUMN] == "scored"
    assert by_id["p5"][nesso_run.STATUS_COLUMN].startswith("skipped:")
    prov = (dest / "t1_provenance.txt").read_text()
    assert "pairs md5" in prov and "status counts" in prov
    with zipfile.ZipFile(dest / "t1_nesso_raw.zip") as zf:
        assert len(zf.namelist()) == 3


def test_override_clears_the_previous_runs_output(run):
    """A replanned run must not inherit the last run's shard tables.

    The submit script skips a shard whose table already exists, so a stale table left behind by
    `--override` would be finalized as if it were this run's result -- silently, since the pair
    ids match.
    """
    run_dir, pairs, _ = run
    fake_predictions(run_dir, "shard_0000", ["p1", "p3"])
    nesso_run.main(["collect", "--run-dir", str(run_dir), "--shard", "0"])
    stale = run_dir / "shards" / "shard_0000.csv"
    assert stale.is_file()

    rc = nesso_run.main(["plan", "--pairs", str(pairs), "--run-dir", str(run_dir),
                         "--pairs-per-shard", "10", "--override"])
    assert rc == 0
    assert not stale.exists(), "the previous run's shard table survived --override"
    assert not (run_dir / "shards" / "shard_0000.zip").exists()
    # and the new plan is in place
    assert (run_dir / "plan" / "pairs.csv").is_file()
    assert sorted(p.name for p in (run_dir / "inputs" / "shard_0000").glob("*.yaml"))
