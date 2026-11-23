#!/usr/bin/env python3
"""Tests for boltz2_run.py.

These exercise the driver's own logic -- parsing inputs, resolving alignments, sharding,
collecting scores, the cleanup guards -- against simulated Boltz-2 output. They never run
Boltz-2, so they say nothing about whether a prediction is any good; they say that the machinery
around it does what it claims.

    python3 -m pytest test_boltz2_run.py -q

They need pytest and PyYAML, but not Boltz, so any Python with those two will do. The image does
not carry pytest on purpose: it holds the reference environment's package list exactly, and
adding a test dependency would make it no longer that environment.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boltz2_run as br                                                       # noqa: E402


# --- fixtures ---------------------------------------------------------------------------------
SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SEQ_B = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSG"
DNA_1 = "ATGCGCATGCGC"


def write_yaml(d: Path, name: str, doc: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def protein_ligand(seq: str = SEQ_A, smiles: str = "CC(=O)O", msa: str = "auto") -> dict:
    return {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": seq, "msa": msa}},
        {"ligand": {"id": "B", "smiles": smiles}},
    ]}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A fake scratch root, so the cleanup guards allow deletion inside tmp_path."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("BOLTZ2_SCRATCH_ROOT", str(root))
    return root


@pytest.fixture
def store(tmp_path):
    """An MSA store holding alignments for both test sequences."""
    s = tmp_path / "store"
    s.mkdir()
    for seq in (SEQ_A, SEQ_B):
        (s / f"{br.seq_key(seq)}.csv").write_text(f"key,sequence\n-1,{seq}\n")
    return s


def simulate_predictions(run_dir: Path, shard: str, job_ids, samples: int = 2,
                         affinity: bool = False, extra_field: bool = False) -> None:
    """Write what Boltz-2 would have written for a shard, so collect has something to read."""
    root = run_dir / "work" / shard / f"boltz_results_{shard}" / "predictions"
    for i, jid in enumerate(job_ids):
        d = root / jid
        d.mkdir(parents=True, exist_ok=True)
        for k in range(samples):
            rec = {"confidence_score": 0.9 - 0.01 * k - 0.001 * i, "ptm": 0.8, "iptm": 0.7,
                   "ligand_iptm": 0.6, "protein_iptm": 0.85, "complex_plddt": 0.88,
                   "complex_iplddt": 0.75, "complex_pde": 1.2, "complex_ipde": 2.3}
            if extra_field:
                rec["chains_ptm"] = 0.5                      # a field the old fixed list missed
            (d / f"confidence_{jid}_model_{k}.json").write_text(json.dumps(rec))
            (d / f"{jid}_model_{k}.cif").write_text(f"# simulated structure {jid} {k}\n")
        if affinity:
            (d / f"affinity_{jid}.json").write_text(json.dumps(
                {"affinity_pred_value": 1.5, "affinity_probability_binary": 0.4}))


def plan_args(**kw):
    """A plan namespace with the parser's defaults, overridden by kw."""
    base = dict(inputs=None, run_dir=None, msa_store=None, allow_missing_msa=False,
                per_shard=br.DEFAULT_PER_SHARD, samples=2, seed=1, no_potentials=False,
                recycling_steps=3, sampling_steps=200, step_scale=1.638,
                affinity_mw_correction=False, output_format="mmcif", boltz_arg=[],
                time_safety=1.5, override=False, allow_any_path=False)
    base.update(kw)
    return type("NS", (), base)()


# --- parsing the input ------------------------------------------------------------------------
def test_parses_protein_ligand(tmp_path):
    p = write_yaml(tmp_path, "job1", protein_ligand())
    spec = br.parse_input(p)
    assert spec.job_id == "job1"
    assert [e.kind for e in spec.entities] == ["protein", "ligand"]
    assert spec.distinct_protein_sequences == [SEQ_A]
    assert not spec.has_affinity


def test_parses_homomultimer_id_list(tmp_path):
    doc = {"version": 1, "sequences": [{"protein": {"id": ["A", "B"], "sequence": SEQ_A}}]}
    spec = br.parse_input(write_yaml(tmp_path, "dimer", doc))
    assert spec.entities[0].n_copies == 2
    # a homodimer is two copies of the same chain: twice the tokens, one distinct sequence
    assert br.token_count(spec)[0] == 2 * len(SEQ_A)
    assert spec.distinct_protein_sequences == [SEQ_A]


def test_parses_protein_dna_complex(tmp_path):
    doc = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": SEQ_A}},
        {"dna": {"id": "B", "sequence": DNA_1}},
    ]}
    spec = br.parse_input(write_yaml(tmp_path, "pd", doc))
    assert {e.kind for e in spec.entities} == {"protein", "dna"}
    assert br.token_count(spec)[0] == len(SEQ_A) + len(DNA_1)


def test_heteromultimer_has_two_distinct_sequences(tmp_path):
    doc = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": SEQ_A}},
        {"protein": {"id": "B", "sequence": SEQ_B}},
    ]}
    spec = br.parse_input(write_yaml(tmp_path, "het", doc))
    assert spec.distinct_protein_sequences == [SEQ_A, SEQ_B]


@pytest.mark.parametrize("doc,fragment", [
    ({"sequences": [{"protein": {"id": "A"}}]}, "no sequence"),
    ({"sequences": [{"wombat": {"id": "A", "sequence": "MK"}}]}, "unknown entity type"),
    ({"sequences": [{"ligand": {"id": "B"}}]}, "neither smiles nor ccd"),
    ({"sequences": [{"ligand": {"id": "B", "smiles": "C", "ccd": "ATP"}}]}, "exclusive"),
    ({"sequences": [{"protein": {"sequence": "MK"}}]}, "no id"),
    ({"sequences": []}, "no `sequences:` list"),
])
def test_rejects_malformed_input(tmp_path, doc, fragment):
    p = write_yaml(tmp_path, "bad", doc)
    with pytest.raises(ValueError) as e:
        br.parse_input(p)
    assert fragment in str(e.value)


def test_rejects_unusable_file_name(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    p = tmp_path / "-leading-dash.yaml"
    p.write_text(yaml.safe_dump(protein_ligand()))
    with pytest.raises(ValueError) as e:
        br.parse_input(p)
    assert "not usable as a job id" in str(e.value)


def test_duplicate_stems_are_reported_not_merged(tmp_path):
    d = tmp_path / "in"
    write_yaml(d, "same", protein_ligand())
    (d / "same.fasta").write_text(">A|protein\n" + SEQ_A + "\n")
    specs, bad = br.load_all_inputs(d)
    assert len(specs) == 1 and len(bad) == 1
    assert "duplicate job id" in bad[0]["reason"]


def test_parses_fasta(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "f1.fasta").write_text(f">A|protein|empty\n{SEQ_A}\n>B|smiles\nCC(=O)O\n")
    spec = br.parse_input(d / "f1.fasta")
    assert [e.kind for e in spec.entities] == ["protein", "ligand"]
    assert spec.entities[0].msa == "empty"


# --- token counting ---------------------------------------------------------------------------
def test_token_count_is_residues_plus_ligand_atoms(tmp_path):
    spec = br.parse_input(write_yaml(tmp_path, "j", protein_ligand(smiles="CC(=O)O")))
    n, quality = br.token_count(spec)
    # acetic acid is 4 heavy atoms; without RDKit the fallback counts element symbols
    assert n > len(SEQ_A)
    assert n - len(SEQ_A) <= 8
    assert quality in ("exact", "approx")


def test_ccd_ligand_is_counted(tmp_path):
    doc = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": SEQ_A}},
        {"ligand": {"id": "B", "ccd": "ATP"}}]}
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    n, quality = br.token_count(spec)
    assert n > len(SEQ_A)
    assert quality in ("exact", "guess")


# --- the sizing table -------------------------------------------------------------------------
def test_sizing_table_loads_and_is_ordered():
    table = br.load_sizing_table()
    assert table, "sizing.tsv should ship with the tool"
    assert table == sorted(table, key=lambda r: r["max_tokens"])
    for row in table:
        assert row["gpumem_gb"] > 0
        assert row["basis"] in ("measured", "extrapolated")
        assert row["source"], "every band must say where its numbers came from"


def test_band_for_picks_the_first_band_that_fits():
    table = br.load_sizing_table()
    assert br.band_for(1, table)["max_tokens"] == table[0]["max_tokens"]
    assert br.band_for(10 ** 9, table) is table[-1]       # bigger than any band: the largest


def test_walltime_never_below_ten_minutes():
    assert br.hhmmss(1) == "00:10:00"
    assert br.hhmmss(3661) == "01:01:01"


def test_timing_constants_load_from_the_table():
    t = br.load_timing()
    for k in br.TIMING_DEFAULTS:
        assert k in t and t[k] > 0


def test_shard_time_is_fixed_cost_plus_per_token():
    """At these sizes the fixed cost dominates, which a per-input model gets badly wrong: on
    Euler 2026-11-18 a 2-input shard took 160 s and a 1-input shard took 155 s."""
    t = {"startup_s": 120.0, "seconds_per_token_sample": 0.067,
         "affinity_extra_s_per_input": 59.0, "steering_factor": 2.4}
    # the three structure-only shards that were measured
    assert br.shard_seconds(301, 2, 2, False, True, t) == pytest.approx(160, abs=3)
    assert br.shard_seconds(247, 2, 1, False, True, t) == pytest.approx(155, abs=3)
    assert br.shard_seconds(1122, 2, 2, False, True, t) == pytest.approx(270, abs=3)
    # and the affinity shard, whose extra cost was read off the same run
    assert br.shard_seconds(171, 2, 1, True, True, t) == pytest.approx(202, abs=5)


def test_turning_steering_off_shrinks_only_the_variable_part():
    t = br.TIMING_DEFAULTS
    on = br.shard_seconds(1000, 2, 2, False, True, t)
    off = br.shard_seconds(1000, 2, 2, False, False, t)
    assert off < on
    assert off > t["startup_s"]                   # the fixed cost does not go away
    assert (on - t["startup_s"]) / (off - t["startup_s"]) == pytest.approx(
        t["steering_factor"], rel=1e-6)


# --- prep -------------------------------------------------------------------------------------
def test_prep_builds_yaml_from_a_table(tmp_path):
    table = tmp_path / "t.csv"
    table.write_text("id,entity,sequence,msa,copies,affinity\n"
                     f"c1,protein,{SEQ_A},auto,1,\n"
                     "c1,ligand,CC(=O)O,,1,yes\n"
                     f"c2,protein,{SEQ_B},auto,2,\n"
                     "c2,ccd,ATP,,1,\n")
    out = tmp_path / "inputs"
    rc = br.cmd_prep(type("NS", (), dict(table=str(table), out=str(out), override=False))())
    assert rc == 0
    c1 = yaml.safe_load((out / "c1.yaml").read_text())
    assert c1["sequences"][0]["protein"]["sequence"] == SEQ_A
    assert c1["sequences"][1]["ligand"]["smiles"] == "CC(=O)O"
    assert c1["properties"] == [{"affinity": {"binder": "B"}}]
    c2 = yaml.safe_load((out / "c2.yaml").read_text())
    assert c2["sequences"][0]["protein"]["id"] == ["A", "B"]      # copies: 2
    assert c2["sequences"][1]["ligand"]["ccd"] == "ATP"
    assert "properties" not in c2
    # and what prep writes must parse back
    for name in ("c1", "c2"):
        br.parse_input(out / f"{name}.yaml")


def test_prep_refuses_two_affinity_binders(tmp_path):
    table = tmp_path / "t.csv"
    table.write_text("id,entity,sequence,affinity\n"
                     f"c1,protein,{SEQ_A},\n"
                     "c1,ligand,CC(=O)O,yes\n"
                     "c1,ligand,CCO,yes\n")
    out = tmp_path / "inputs"
    rc = br.cmd_prep(type("NS", (), dict(table=str(table), out=str(out), override=False))())
    assert rc == 1
    assert not (out / "c1.yaml").exists()


# --- alignments -------------------------------------------------------------------------------
def test_combine_paired_unpaired_follows_boltz(tmp_path):
    """Paired rows keyed by row index, all-gap rows dropped, unpaired keyed -1 with the query
    dropped. Transcribed from boltz.main.compute_msa; this pins the transcription."""
    paired = ">q\nAAAA\n>h1\n----\n>h2\nACAA\n"
    unpaired = ">q\nAAAA\n>u1\nAGAA\n"
    out = br.combine_paired_unpaired(paired, unpaired, max_paired=10, max_seqs=100)
    lines = out.splitlines()
    assert lines[0] == "key,sequence"
    # row 1 of the paired block is all gaps and is dropped; the surviving rows keep their ORIGINAL
    # indices 0 and 2, which is what makes the same key mean the same organism across chains
    assert lines[1] == "0,AAAA"
    assert lines[2] == "2,ACAA"
    # the unpaired query is dropped because the paired block already carries it
    assert lines[3] == "-1,AGAA"
    assert len(lines) == 4


def test_unpaired_conversion_keys_everything_minus_one():
    out = br.a3m_to_boltz_csv(">q\nAAAA\n>h\nACAA\n", max_seqs=10)
    assert out.splitlines() == ["key,sequence", "-1,AAAA", "-1,ACAA"]


def test_store_lookup_is_by_sequence_not_label(store):
    assert br.store_lookup(store, SEQ_A) is not None
    assert br.store_lookup(store, SEQ_A.lower()) is not None      # case-insensitive
    assert br.store_lookup(store, "MKV") is None


def test_paired_lookup_needs_every_member(tmp_path, store):
    seqs = [SEQ_A, SEQ_B]
    d = store / "paired" / br.group_key(seqs)
    d.mkdir(parents=True)
    (d / f"{br.seq_key(SEQ_A)}.csv").write_text("key,sequence\n0,X\n")
    assert br.paired_lookup(store, seqs) is None                  # SEQ_B missing
    (d / f"{br.seq_key(SEQ_B)}.csv").write_text("key,sequence\n0,X\n")
    assert br.paired_lookup(store, seqs) is not None


def test_group_key_is_order_independent():
    assert br.group_key([SEQ_A, SEQ_B]) == br.group_key([SEQ_B, SEQ_A])


def test_explicit_msa_path_is_kept(tmp_path, store):
    own = tmp_path / "mine.csv"
    own.write_text("key,sequence\n-1,X\n")
    spec = br.parse_input(write_yaml(tmp_path / "in", "j", protein_ligand(msa=str(own))))
    mapping, how, used = br.resolve_msas(spec, store, allow_missing=False)
    assert mapping["A"] == str(own.resolve())
    assert used and used[0]["how"] == "given in the input"


def test_msa_empty_is_passed_through(tmp_path, store):
    spec = br.parse_input(write_yaml(tmp_path / "in", "j", protein_ligand(msa="empty")))
    mapping, _, used = br.resolve_msas(spec, store, allow_missing=False)
    assert mapping["A"] == "empty"
    assert used == []


def test_missing_alignment_blocks_the_plan_unless_allowed(tmp_path, store):
    spec = br.parse_input(write_yaml(tmp_path / "in", "j", protein_ligand(seq="MKVFFF")))
    mapping, how, _ = br.resolve_msas(spec, store, allow_missing=False)
    assert mapping is None and how == "no alignment in the store"
    mapping, how, _ = br.resolve_msas(spec, store, allow_missing=True)
    assert mapping["A"] == "empty" and how == "missing -> empty"


def test_apply_msas_preserves_constraints_and_templates(tmp_path, store):
    doc = protein_ligand()
    doc["constraints"] = [{"pocket": {"binder": "B", "contacts": [["A", 10]], "max_distance": 6}}]
    doc["templates"] = [{"cif": "/cluster/project/x/ref.cif"}]
    spec = br.parse_input(write_yaml(tmp_path / "in", "j", doc))
    mapping, _, _ = br.resolve_msas(spec, store, allow_missing=False)
    out = br.apply_msas(spec, mapping)
    assert out["constraints"] == doc["constraints"]
    assert out["templates"] == doc["templates"]
    assert out["sequences"][0]["protein"]["msa"].endswith(".csv")
    assert out["sequences"][0]["protein"]["msa"] != "auto"


def test_apply_msas_does_not_mutate_the_original(tmp_path, store):
    spec = br.parse_input(write_yaml(tmp_path / "in", "j", protein_ligand()))
    mapping, _, _ = br.resolve_msas(spec, store, allow_missing=False)
    br.apply_msas(spec, mapping)
    assert spec.doc["sequences"][0]["protein"]["msa"] == "auto"


# --- affinity validation ----------------------------------------------------------------------
def test_affinity_on_a_protein_chain_is_refused(tmp_path):
    doc = protein_ligand()
    doc["properties"] = [{"affinity": {"binder": "A"}}]           # A is the protein
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    assert "only scores a ligand" in br.check_affinity(spec)


def test_affinity_on_an_unknown_chain_is_refused(tmp_path):
    doc = protein_ligand()
    doc["properties"] = [{"affinity": {"binder": "Z"}}]
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    assert "not a chain of this input" in br.check_affinity(spec)


def test_large_affinity_binder_warns_but_is_not_refused(tmp_path):
    """Boltz-2's hard limit is 128 atoms but it was trained to about 56. Between the two it
    answers, and the answer is outside what it has seen — so warn, do not refuse."""
    big = "C" * 80                                       # 80 heavy atoms, under the hard limit
    doc = protein_ligand(smiles=big)
    doc["properties"] = [{"affinity": {"binder": "B"}}]
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    assert br.check_affinity(spec) is None               # not refused
    note = br.affinity_advisory(spec)
    assert note and "caution" in note


def test_small_affinity_binder_gets_no_warning(tmp_path):
    doc = protein_ligand(smiles="CC(=O)O")
    doc["properties"] = [{"affinity": {"binder": "B"}}]
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    assert br.affinity_advisory(spec) is None


def test_affinity_on_a_ligand_is_accepted(tmp_path):
    doc = protein_ligand()
    doc["properties"] = [{"affinity": {"binder": "B"}}]
    spec = br.parse_input(write_yaml(tmp_path, "j", doc))
    assert br.check_affinity(spec) is None


# --- plan -------------------------------------------------------------------------------------
def test_plan_shards_and_records_every_input(tmp_path, scratch, store):
    d = tmp_path / "in"
    for i in range(5):
        write_yaml(d, f"j{i}", protein_ligand())
    write_yaml(d, "nomsa", protein_ligand(seq="MKVFFF"))          # not in the store
    run = scratch / "run1"
    rc = br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                               per_shard=2, allow_missing_msa=True))
    assert rc == 0
    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    assert len(rows) == 6                                          # no input is ever dropped
    assert {r["job_id"] for r in rows} == {f"j{i}" for i in range(5)} | {"nomsa"}
    shards = {r[br.SHARD_COLUMN] for r in rows if r[br.SHARD_COLUMN]}
    assert len(shards) == 3                                        # 6 inputs, 2 per shard
    for s in shards:
        assert (run / "inputs" / s).is_dir()


def test_plan_refuses_a_run_outside_scratch(tmp_path, store):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    with pytest.raises(SystemExit) as e:
        br.cmd_plan(plan_args(inputs=str(d), run_dir=str(tmp_path / "elsewhere"),
                              msa_store=str(store)))
    assert "outside the scratch root" in str(e.value)


def test_plan_stops_when_alignments_are_missing(tmp_path, scratch, store):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand(seq="MKVFFF"))
    with pytest.raises(SystemExit) as e:
        br.cmd_plan(plan_args(inputs=str(d), run_dir=str(scratch / "r"), msa_store=str(store)))
    assert "no alignment" in str(e.value)


def test_plan_groups_similar_sizes_into_a_shard(tmp_path, scratch, store):
    d = tmp_path / "in"
    for i in range(3):
        write_yaml(d, f"small{i}", protein_ligand(seq=SEQ_A))
    for i in range(3):
        write_yaml(d, f"big{i}", protein_ligand(seq=SEQ_B * 6))
    (store / f"{br.seq_key(SEQ_B * 6)}.csv").write_text("key,sequence\n-1,X\n")
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store), per_shard=3))
    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    by_shard: dict[str, set] = {}
    for r in rows:
        by_shard.setdefault(r[br.SHARD_COLUMN], set()).add(r["job_id"][:3])
    # each shard holds one size class, not a mixture, so one walltime fits the whole shard
    assert all(len(v) == 1 for v in by_shard.values())


def test_size_and_plan_agree_on_which_shard_is_which(tmp_path, scratch, store):
    """The array ranges `size` prints must name the shards `plan` actually writes.

    They are worked out by one function for exactly this reason. When they were computed
    separately, a band boundary falling inside a shard made them disagree, and the sbatch line
    for one band then carried its memory request to another band's shards.
    """
    d = tmp_path / "in"
    sizes = {"tiny": SEQ_A, "small": SEQ_A * 2, "mid": SEQ_B * 10, "big": SEQ_B * 14}
    for name, seq in sizes.items():
        for i in range(3):
            write_yaml(d, f"{name}{i}", protein_ligand(seq=seq))
        (store / f"{br.seq_key(seq)}.csv").write_text("key,sequence\n-1,X\n")
    run = scratch / "r"
    per_shard = 2
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                          per_shard=per_shard))

    specs, _ = br.load_all_inputs(d)
    sized = [(s.job_id, br.token_count(s)[0], br.protein_signature(s), s.has_affinity)
             for s in specs]
    assigned, records = br.band_plan(sized, per_shard, br.load_sizing_table())

    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    from_plan = {r["job_id"]: r[br.SHARD_COLUMN] for r in rows}
    assert from_plan == assigned

    # and every shard named by a band's array range really exists in the plan
    on_disk = {p.name for p in (run / "inputs").iterdir()}
    for rec in records:
        for i in range(rec["array_from"], rec["array_to"] + 1):
            assert br.shard_name(i) in on_disk


def test_no_shard_mixes_two_size_bands(tmp_path, scratch, store):
    """A shard is one SLURM task with one memory request, so it must not straddle a band."""
    d = tmp_path / "in"
    for name, seq in {"s": SEQ_A, "m": SEQ_B * 10, "l": SEQ_B * 14}.items():
        for i in range(3):
            write_yaml(d, f"{name}{i}", protein_ligand(seq=seq))
        (store / f"{br.seq_key(seq)}.csv").write_text("key,sequence\n-1,X\n")
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store), per_shard=2))

    table = br.load_sizing_table()
    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    by_shard: dict[str, set] = {}
    for r in rows:
        band = br.band_for(int(r[br.TOKENS_COLUMN]), table)["max_tokens"]
        by_shard.setdefault(r[br.SHARD_COLUMN], set()).add(band)
    assert all(len(v) == 1 for v in by_shard.values()), by_shard


def test_affinity_inputs_never_share_a_shard_with_others(tmp_path, scratch, store):
    """Boltz-2 2.2.1 runs its affinity stage over EVERY input in the call, not only those that
    asked for it, and then dies on the missing pre_affinity_<id>.npz of one that did not — after
    the structures are already predicted, so the GPU time is lost. Seen on Euler job
    21396655_1 (2026-11-18). The two kinds must therefore be sharded apart."""
    d = tmp_path / "in"
    for i in range(3):
        doc = protein_ligand()
        doc["properties"] = [{"affinity": {"binder": "B"}}]
        write_yaml(d, f"aff{i}", doc)
        write_yaml(d, f"plain{i}", protein_ligand())    # identical size, so only affinity differs
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store), per_shard=4))

    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    by_shard: dict[str, set] = {}
    for r in rows:
        by_shard.setdefault(r[br.SHARD_COLUMN], set()).add(r["job_id"][:3])
    assert all(len(v) == 1 for v in by_shard.values()), by_shard
    # and per_shard=4 would otherwise have fitted all six into two shards of mixed kind
    assert len(by_shard) == 2


def mixed_plan(tmp_path, scratch, store, per_shard=2):
    """A plan spanning two size bands and an affinity group: three sbatch calls."""
    d = tmp_path / "in"
    for i in range(3):
        write_yaml(d, f"s{i}", protein_ligand(seq=SEQ_A))
    big = SEQ_B * 10
    (store / f"{br.seq_key(big)}.csv").write_text("key,sequence\n-1,X\n")
    for i in range(2):
        write_yaml(d, f"m{i}", protein_ligand(seq=big))
    doc = protein_ligand()
    doc["properties"] = [{"affinity": {"binder": "B"}}]
    write_yaml(d, "aff", doc)
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                          per_shard=per_shard))
    return run


def test_plan_writes_one_band_per_sbatch_call(tmp_path, scratch, store):
    run = mixed_plan(tmp_path, scratch, store)
    bands = json.loads((run / "plan" / "bands.json").read_text())
    assert len(bands) == 3
    assert sum(b["affinity"] for b in bands) == 1
    for b in bands:
        assert b["gpumem_gb"] > 0 and b["mem_mb_per_cpu"] > 0
        assert b["time"].count(":") == 2


def test_band_ranges_cover_every_shard_exactly_once(tmp_path, scratch, store):
    run = mixed_plan(tmp_path, scratch, store)
    bands = json.loads((run / "plan" / "bands.json").read_text())
    covered = [i for b in bands for i in range(b["array_from"], b["array_to"] + 1)]
    on_disk = sorted(int(p.name.split("_")[1]) for p in (run / "inputs").iterdir())
    assert sorted(covered) == on_disk


def test_submit_script_is_valid_bash_with_one_sbatch_per_band(tmp_path, scratch, store):
    import subprocess
    run = mixed_plan(tmp_path, scratch, store)
    script = run / "plan" / "submit.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    bands = json.loads((run / "plan" / "bands.json").read_text())
    assert text.count("sbatch --parsable") == len(bands)
    for b in bands:
        assert f"--array={b['array_from']}-{b['array_to']}" in text
        assert f"--gres=gpumem:{b['gpumem_gb']}g" in text
    # the job scripts write logs there and SLURM cannot create it, so the script must
    assert 'mkdir -p "/cluster/scratch/$USER/boltz2/slurm"' in text


def test_submit_script_points_at_this_checkout(tmp_path, scratch, store):
    run = mixed_plan(tmp_path, scratch, store)
    text = (run / "plan" / "submit.sh").read_text()
    here = str(Path(br.__file__).resolve().parent)
    assert f'TOOL="{here}"' in text and f'RUN="{run.resolve()}"' in text


def test_shards_csv_records_the_suggested_memory(tmp_path, scratch, store):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store)))
    rows, fields = br.read_csv_rows(run / "plan" / "shards.csv")
    assert "suggested_gpumem_gb" in fields and "sizing_basis" in fields
    assert int(rows[0]["suggested_gpumem_gb"]) > 0


def test_plan_writes_resolved_inputs_not_auto(tmp_path, scratch, store):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store)))
    written = yaml.safe_load((run / "inputs" / "shard_0000" / "j.yaml").read_text())
    assert written["sequences"][0]["protein"]["msa"].endswith(".csv")


def test_plan_will_not_silently_replace_itself(tmp_path, scratch, store):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store)))
    with pytest.raises(SystemExit) as e:
        br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store)))
    assert "already exists" in str(e.value)


# --- protocol ---------------------------------------------------------------------------------
def test_protocol_round_trips_through_the_plan(tmp_path, scratch, store, capsys):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store), samples=7,
                          seed=42, no_potentials=True, recycling_steps=5,
                          boltz_arg=["--subsample_msa"]))
    br.cmd_protocol(type("NS", (), dict(run_dir=str(run)))())
    printed = capsys.readouterr().out.split()
    assert printed[printed.index("--diffusion_samples") + 1] == "7"
    assert printed[printed.index("--seed") + 1] == "42"
    assert printed[printed.index("--recycling_steps") + 1] == "5"
    # steering is the wrapper's default, so OFF has to be said out loud
    assert "--no_potentials" in printed and "--use_potentials" not in printed
    assert "--subsample_msa" in printed                            # passthrough survives


def test_protocol_default_turns_steering_on(tmp_path, scratch, store, capsys):
    d = tmp_path / "in"
    write_yaml(d, "j", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store)))
    br.cmd_protocol(type("NS", (), dict(run_dir=str(run)))())
    assert "--use_potentials" in capsys.readouterr().out


# --- collect ----------------------------------------------------------------------------------
def collect_ns(run, shard, cleanup=False, incomplete=False):
    return type("NS", (), dict(run_dir=str(run), shard=shard, allow_cleanup=cleanup,
                               accept_incomplete=incomplete))()


def planned_run(tmp_path, scratch, store, n=3, **kw):
    d = tmp_path / "in"
    for i in range(n):
        write_yaml(d, f"j{i}", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                          per_shard=n, **kw))
    return run, [f"j{i}" for i in range(n)]


def test_collect_reads_every_scalar_field(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store)
    simulate_predictions(run, "shard_0000", ids, samples=2, extra_field=True)
    assert br.cmd_collect(collect_ns(run, "shard_0000")) == 0
    rows, fields = br.read_csv_rows(run / "shards" / "shard_0000.csv")
    assert all(r[br.STATUS_COLUMN] == "scored" for r in rows)
    # a field not on any hard-coded list still reaches the table
    assert "boltz2:chains_ptm" in fields and "boltz2:chains_ptm_mean" in fields
    assert "boltz2:iptm" in fields and "boltz2:iptm_mean" in fields


def test_collect_reports_model_0_and_the_mean(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=1)
    simulate_predictions(run, "shard_0000", ids, samples=2)
    br.cmd_collect(collect_ns(run, "shard_0000"))
    row = br.read_csv_rows(run / "shards" / "shard_0000.csv")[0][0]
    assert float(row["boltz2:confidence_score"]) == pytest.approx(0.9)
    assert float(row["boltz2:confidence_score_mean"]) == pytest.approx((0.9 + 0.89) / 2)


def test_collect_marks_a_missing_prediction(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=3)
    simulate_predictions(run, "shard_0000", ids[:2])
    assert br.cmd_collect(collect_ns(run, "shard_0000")) == 1
    rows, _ = br.read_csv_rows(run / "shards" / "shard_0000.csv")
    assert sorted(r[br.STATUS_COLUMN] for r in rows) == ["no_prediction", "scored", "scored"]


def test_collect_flags_too_few_samples(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=1, samples=5)
    simulate_predictions(run, "shard_0000", ids, samples=2)
    br.cmd_collect(collect_ns(run, "shard_0000"))
    row = br.read_csv_rows(run / "shards" / "shard_0000.csv")[0][0]
    assert row[br.STATUS_COLUMN] == "incomplete: 2 of 5 samples"


def test_collect_carries_affinity_through(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=1)
    simulate_predictions(run, "shard_0000", ids, samples=2, affinity=True)
    br.cmd_collect(collect_ns(run, "shard_0000"))
    row = br.read_csv_rows(run / "shards" / "shard_0000.csv")[0][0]
    assert float(row["boltz2:affinity_pred_value"]) == pytest.approx(1.5)


def test_collect_zips_and_verifies(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=2)
    simulate_predictions(run, "shard_0000", ids, samples=2)
    br.cmd_collect(collect_ns(run, "shard_0000"))
    z = run / "shards" / "shard_0000.zip"
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
    assert f"{ids[0]}/confidence_{ids[0]}_model_0.json" in names
    assert any(n.endswith(".cif") for n in names)


def test_cleanup_only_after_a_complete_shard(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=3)
    simulate_predictions(run, "shard_0000", ids[:2])                # one missing
    assert br.cmd_collect(collect_ns(run, "shard_0000", cleanup=True)) == 1
    assert (run / "work" / "shard_0000").is_dir(), "a failed shard keeps its files"
    assert not (run / "shards" / "shard_0000.done").exists()

    simulate_predictions(run, "shard_0000", ids)                     # now complete
    assert br.cmd_collect(collect_ns(run, "shard_0000", cleanup=True)) == 0
    assert not (run / "work" / "shard_0000").exists()
    assert (run / "shards" / "shard_0000.done").exists()


def test_incomplete_shard_can_be_closed_deliberately(tmp_path, scratch, store):
    run, ids = planned_run(tmp_path, scratch, store, n=3)
    simulate_predictions(run, "shard_0000", ids[:2])
    assert br.cmd_collect(collect_ns(run, "shard_0000", cleanup=True, incomplete=True)) == 0
    assert not (run / "work" / "shard_0000").exists()


def test_cleanup_refuses_outside_scratch(tmp_path, store, monkeypatch):
    monkeypatch.setenv("BOLTZ2_SCRATCH_ROOT", str(tmp_path / "somewhere-else"))
    (tmp_path / "somewhere-else").mkdir()
    d = tmp_path / "in"
    write_yaml(d, "j0", protein_ligand())
    run = tmp_path / "run-not-on-scratch"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                          allow_any_path=True))
    simulate_predictions(run, "shard_0000", ["j0"])
    assert br.cmd_collect(collect_ns(run, "shard_0000", cleanup=True)) == 1
    assert (run / "work" / "shard_0000").is_dir(), "nothing outside scratch is ever deleted"


# --- finalize ---------------------------------------------------------------------------------
def finalize_ns(run, dest, label=None, allow_open=False):
    return type("NS", (), dict(run_dir=str(run), dest=str(dest), label=label,
                               allow_open=allow_open))()


def finished_run(tmp_path, scratch, store, n=4, per_shard=2, image="sha-one"):
    d = tmp_path / "in"
    for i in range(n):
        write_yaml(d, f"j{i}", protein_ligand())
    run = scratch / "r"
    br.cmd_plan(plan_args(inputs=str(d), run_dir=str(run), msa_store=str(store),
                          per_shard=per_shard))
    rows, _ = br.read_csv_rows(run / "plan" / "jobs.csv")
    by_shard: dict[str, list[str]] = {}
    for r in rows:
        by_shard.setdefault(r[br.SHARD_COLUMN], []).append(r["job_id"])
    for shard, ids in by_shard.items():
        simulate_predictions(run, shard, ids, samples=2)
        (run / "shards" / f"{shard}.job.json").write_text(json.dumps(
            {"shard": shard, "gpu": "NVIDIA GeForce RTX 3090", "image_sha256": image,
             "image": "/x/boltz2.sif", "array_job_id": "1", "task_id": shard[-1],
             "n_inputs": len(ids), "elapsed_s": 120}))
        br.cmd_collect(collect_ns(run, shard, cleanup=True))
    return run


def test_finalize_merges_every_shard(tmp_path, scratch, store):
    run = finished_run(tmp_path, scratch, store, n=4, per_shard=2)
    dest = tmp_path / "out"
    assert br.cmd_finalize(finalize_ns(run, dest, label="demo")) == 0
    rows, _ = br.read_csv_rows(dest / "demo_boltz2_scores.csv")
    assert len(rows) == 4
    assert all(r[br.STATUS_COLUMN] == "scored" for r in rows)
    manifest, _ = br.read_csv_rows(dest / "demo_boltz2_manifest.csv")
    assert {m["job_id"] for m in manifest} == {f"j{i}" for i in range(4)}
    prov = (dest / "demo_boltz2_provenance.txt").read_text()
    assert "RTX 3090" in prov and "sha-one" in prov


def test_finalize_refuses_while_shards_are_open(tmp_path, scratch, store):
    run = finished_run(tmp_path, scratch, store, n=2, per_shard=2)
    (run / "shards" / "shard_0000.done").unlink()
    with pytest.raises(SystemExit) as e:
        br.cmd_finalize(finalize_ns(run, tmp_path / "out"))
    assert "not finished" in str(e.value)


def test_finalize_reports_a_mixture_of_images(tmp_path, scratch, store):
    run = finished_run(tmp_path, scratch, store, n=4, per_shard=2)
    p = run / "shards" / "shard_0001.job.json"
    j = json.loads(p.read_text())
    j["image_sha256"] = "sha-two"
    p.write_text(json.dumps(j))
    dest = tmp_path / "out"
    assert br.cmd_finalize(finalize_ns(run, dest, label="mix")) == 1
    assert "MORE THAN ONE IMAGE" in (dest / "mix_boltz2_provenance.txt").read_text()


def test_finalize_never_overwrites_a_different_archive(tmp_path, scratch, store):
    run = finished_run(tmp_path, scratch, store, n=2, per_shard=2)
    dest = tmp_path / "out"
    br.cmd_finalize(finalize_ns(run, dest, label="x"))
    target = next(dest.glob("x_boltz2_shard_*.zip"))
    target.write_bytes(b"something else entirely")
    with pytest.raises(SystemExit) as e:
        br.cmd_finalize(finalize_ns(run, dest, label="x"))
    assert "different contents" in str(e.value)


# --- file hygiene -----------------------------------------------------------------------------
def test_tables_are_written_with_unix_line_endings(tmp_path, scratch, store):
    run = finished_run(tmp_path, scratch, store, n=2, per_shard=2)
    dest = tmp_path / "out"
    br.cmd_finalize(finalize_ns(run, dest, label="x"))
    for p in list(dest.glob("*.csv")) + [run / "plan" / "jobs.csv"]:
        assert b"\r\n" not in p.read_bytes(), f"{p.name} has CRLF line endings"


def fake_repo(root: Path, packed: bool = False) -> str:
    """A minimal .git directory: HEAD pointing at a branch, the branch at a fixed hash."""
    sha = "0123456789abcdef0123456789abcdef01234567"
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    if packed:
        (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{sha} refs/heads/main\n")
    else:
        (git / "refs" / "heads" / "main").write_text(sha + "\n")
    return sha


@pytest.mark.parametrize("packed", [False, True])
def test_commit_is_read_from_files_when_there_is_no_git(tmp_path, monkeypatch, packed):
    """The image has no git binary, and plan runs inside it, so provenance used to say
    'unknown'. The commit must still be recorded, and must not claim a clean tree."""
    sha = fake_repo(tmp_path, packed=packed)
    sub = tmp_path / "boltz2"
    sub.mkdir()

    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(br.subprocess, "run", no_git)
    got = br.git_commit(sub)
    assert got.startswith(sha)
    assert "dirty state not checked" in got


def test_commit_reading_follows_a_worktree_gitdir_file(tmp_path, monkeypatch):
    main = tmp_path / "main"
    sha = fake_repo(main)
    wt_git = main / ".git" / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/main\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert br.read_head_commit(wt) == sha


def test_no_repository_means_unknown(tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(br.subprocess, "run", no_git)
    assert br.git_commit(tmp_path) == "unknown"


def test_scratch_root_guard_matches_only_real_children(tmp_path):
    roots = [tmp_path / "scratch"]
    (tmp_path / "scratch").mkdir()
    assert br.under_any(tmp_path / "scratch" / "run", roots)
    assert not br.under_any(tmp_path / "scratch-other" / "run", roots)
    assert not br.under_any(tmp_path / "elsewhere", roots)
