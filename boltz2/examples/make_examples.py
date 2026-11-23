#!/usr/bin/env python3
"""Write the example Boltz-2 inputs used by the first-run notebook.

Every sequence is fetched from UniProt by accession at run time, so nothing here is a pasted
sequence that could silently drift from what the accession actually holds. The accessions are
the only data in this file, and they are visible in `EXAMPLES` below.

    python3 make_examples.py --out inputs

The proteins were chosen for being small, so the example finishes in minutes on one GPU. They
are not a biological claim: this script exists to show the input shapes, not to make a point
about these particular molecules.

Shapes covered:
    01_monomer              one protein, nothing else
    02_protein_ligand       one protein and one small molecule, with an affinity request
    03_homodimer            two identical copies of one protein
    04_protein_dna          one protein and a double-stranded DNA fragment
    05_heterodimer          two different proteins (needs a paired alignment)
    06_pocket_constraint    protein and ligand, with the ligand steered towards named residues
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required; run this with `boltz2 python` or in any env with PyYAML")

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
# A plain client name. Never put an email address or any other personal detail in a header.
USER_AGENT = "hpc-tools-boltz2-examples/1.0"

# accession -> what it is, for the log only. Nothing downstream depends on these descriptions.
PROTEINS = {
    "P00698": "lysozyme C, Gallus gallus",
    "P61823": "ribonuclease pancreatic, Bos taurus",
    "P0A8V2": "DNA-directed RNA polymerase subunit beta, Escherichia coli (long, size example)",
    "P03023": "lac repressor, Escherichia coli",
    "P00648": "barnase, Bacillus amyloliquefaciens",
    "P11540": "barstar, Bacillus amyloliquefaciens",
}

# A short palindromic DNA fragment, written as its two strands. Generated here rather than
# pasted from a paper: the reverse complement is computed, so the two strands cannot disagree.
DNA_TOP = "AATTGTGAGCGGATAACAATT"
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

# A CCD code, i.e. a ligand named by its Chemical Component Dictionary entry rather than SMILES.
# C5P is cytidine-5'-monophosphate. The image carries the full dictionary, so no download is
# needed for this to resolve.
LIGAND_CCD = "C5P"
# Acetate, as SMILES, to show the other way of naming a ligand.
LIGAND_SMILES = "CC(=O)[O-]"


def fetch_sequence(acc: str) -> str:
    """The sequence of one UniProt accession, from its FASTA record."""
    req = urllib.request.Request(UNIPROT_FASTA.format(acc=acc),
                                 headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
    except urllib.error.URLError as e:
        sys.exit(f"FATAL: could not fetch {acc} from UniProt: {e}\n"
                 f"       On a compute node this will always fail -- run it on a login node, "
                 f"or in a job that has loaded eth_proxy.")
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    seq = "".join(lines)
    if not seq:
        sys.exit(f"FATAL: {acc} returned no sequence")
    return seq


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def build(seqs: dict[str, str]) -> dict[str, dict]:
    """The example documents, as plain data. One entry per input file."""
    out: dict[str, dict] = {}

    # 1. A protein on its own. `msa: auto` is this tool's marker for "fill this in from the
    #    store"; it is not a Boltz keyword. `boltz2_run.py plan` replaces it with a real path.
    out["01_monomer"] = {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": seqs["P00698"], "msa": "auto"}}],
    }

    # 2. A protein with a ligand, and an affinity request. Affinity is scored for exactly one
    #    ligand chain, named by `binder`.
    out["02_protein_ligand"] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seqs["P61823"], "msa": "auto"}},
            {"ligand": {"id": "B", "ccd": LIGAND_CCD}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }

    # 3. Two identical copies of one protein: one entity, two chain ids. Written this way rather
    #    than as two entries, so Boltz knows the chains are the same molecule.
    out["03_homodimer"] = {
        "version": 1,
        "sequences": [{"protein": {"id": ["A", "B"], "sequence": seqs["P03023"],
                                   "msa": "auto"}}],
    }

    # 4. A protein with double-stranded DNA: two dna entities, the second the reverse complement
    #    of the first. DNA and RNA take no msa.
    out["04_protein_dna"] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seqs["P03023"], "msa": "auto"}},
            {"dna": {"id": "B", "sequence": DNA_TOP}},
            {"dna": {"id": "C", "sequence": reverse_complement(DNA_TOP)}},
        ],
    }

    # 5. Two DIFFERENT proteins. This is the case that needs a *paired* alignment: the model has
    #    to see which row of chain A's alignment and which row of chain B's come from the same
    #    organism. A per-sequence alignment store cannot supply that, so `boltz2_run.py msa`
    #    fetches a paired set for this combination of chains and caches it separately.
    #    Barnase and barstar are a long-standing model of a small, tight protein-protein complex,
    #    chosen here because both chains are short.
    out["05_heterodimer"] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seqs["P00648"], "msa": "auto"}},
            {"protein": {"id": "B", "sequence": seqs["P11540"], "msa": "auto"}},
        ],
    }

    # 6. A pocket constraint: the ligand is told which residues of the protein it should sit
    #    near. Residue indices start at 1, counted along the sequence IN THIS FILE.
    #
    #    A trap worth knowing: a UniProt record is the PRECURSOR, signal peptide included. For
    #    P61823 that is 26 extra residues at the front, so position 12 here is not His12 of
    #    mature ribonuclease A -- the conventional active-site numbering is shifted by 26. The
    #    contacts below are therefore illustrative of the syntax and are NOT this protein's
    #    binding site. When the pilot ran this input it came back with a markedly lower ligand
    #    interface score than the unconstrained protein-ligand example, which is what a
    #    constraint pointing somewhere unhelpful should do.
    #
    #    For real work: decide whether you want the precursor or the mature chain, and number
    #    the constraint against the sequence you actually put in the file.
    out["06_pocket_constraint"] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seqs["P61823"], "msa": "auto"}},
            {"ligand": {"id": "B", "smiles": LIGAND_SMILES}},
        ],
        "constraints": [{"pocket": {"binder": "B",
                                    "contacts": [["A", 12], ["A", 41], ["A", 119]],
                                    "max_distance": 6,
                                    "force": False}}],
    }
    return out


def write_chain_table(path: Path, seqs: dict[str, str]) -> None:
    """The same two complexes as a table, to show what `boltz2_run.py prep` takes.

    One row per chain. `prep` turns this into YAML; it is a convenience for simple shapes, and
    anything it cannot express (constraints, templates) is written as YAML directly.
    """
    rows = [
        ("t1_monomer", "protein", seqs["P00698"], "auto", "1", ""),
        ("t2_complex", "protein", seqs["P61823"], "auto", "1", ""),
        ("t2_complex", "ccd", LIGAND_CCD, "", "1", "yes"),
        ("t3_dimer", "protein", seqs["P03023"], "auto", "2", ""),
    ]
    lines = ["id,entity,sequence,msa,copies,affinity"]
    lines += [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="inputs", help="directory for the example inputs")
    ap.add_argument("--large", default="inputs_large",
                    help="directory for the one large example, which is for sizing only")
    ap.add_argument("--table", default="chains.csv", help="where to write the example table")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    seqs = {}
    for acc, what in PROTEINS.items():
        seqs[acc] = fetch_sequence(acc)
        print(f"  {acc}  {len(seqs[acc]):>5} aa   {what}")

    docs = build(seqs)
    for name, doc in docs.items():
        p = out / f"{name}.yaml"
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)
        n_chains = sum(len(v["id"]) if isinstance(v["id"], list) else 1
                       for e in doc["sequences"] for v in e.values())
        print(f"  wrote {p}  ({n_chains} chain(s))")

    # The one large protein lives in a directory of its own, and nothing in the first-run
    # notebook predicts it. It is here so `size` has something that lands in a different band,
    # which is the only way to see what banding is for. Predicting it needs a bigger GPU and
    # considerably more time than a first run should spend.
    large = Path(args.large).expanduser()
    large.mkdir(parents=True, exist_ok=True)
    big = {"version": 1,
           "sequences": [{"protein": {"id": "A", "sequence": seqs["P0A8V2"], "msa": "auto"}}]}
    with open(large / "07_large_monomer.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(big, fh, default_flow_style=False, sort_keys=False)
    (large / "README.txt").write_text(
        f"07_large_monomer.yaml is UniProt P0A8V2, {len(seqs['P0A8V2'])} residues.\n"
        "It is here only so `boltz2_run.py size` has an input in a larger band, making the\n"
        "point of size banding visible. Nothing in the first-run notebook predicts it.\n")
    print(f"  wrote {large / '07_large_monomer.yaml'}  (sizing demonstration only)")

    table = Path(args.table).expanduser()
    write_chain_table(table, seqs)
    print(f"  wrote {table}")

    manifest = out / "PROVENANCE.txt"
    manifest.write_text(
        "Example Boltz-2 inputs, generated by make_examples.py.\n"
        "Sequences fetched from UniProt (rest.uniprot.org) at generation time:\n"
        + "".join(f"  {a}  {len(seqs[a])} aa  {w}\n" for a, w in PROTEINS.items())
        + f"\nDNA: {DNA_TOP} and its reverse complement, computed here.\n"
          f"Ligands: CCD {LIGAND_CCD}; SMILES {LIGAND_SMILES}.\n"
          "\nThese files are generated, not edited by hand. Rerun make_examples.py to rebuild\n"
          "them; do not patch them in place.\n")
    print(f"  wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
