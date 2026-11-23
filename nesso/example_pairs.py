#!/usr/bin/env python3
"""Build an example pairs table for the Nesso-1 runner, from public records.

A worked starting point for a new user, and a smoke test for an install: it fetches real protein
sequences from UniProt and real ligand structures from PubChem, writes them in the runner's input
contract, and leaves a table that `nesso_run.py plan` accepts unchanged.

Nothing is pasted in: each protein is named by its UniProt accession and each ligand by its
PubChem name, both visible below, and both are fetched at run time. Swap those lists for your own
and the same script makes your table.

Run it on a login node -- compute nodes have no external network.

    python nesso/example_pairs.py --out /cluster/scratch/$USER/nesso/example_pairs.csv

The default set is four well-studied drug targets against six ligands, a mixture of each target's
own inhibitor and other targets' inhibitors, so a score that works should rank each target's own
ligand above the others. Which ligand belongs to which target is written out as a `cognate`
column, so that check needs nothing but the output table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

# UniProt accession -> a short label. The accession is the record; the label is only for reading.
PROTEINS = {
    "P00918": "carbonic_anhydrase_2",
    "P00533": "EGFR",
    "P24941": "CDK2",
    "P07900": "HSP90_alpha",
}

# PubChem compound name -> the accession above that it is the established inhibitor of.
# This mapping is the experiment's *design*, not fetched data: it is what makes the table a
# control set, giving every target at least one known binder and at least four molecules with no
# reason to bind it. It travels into the output as a `cognate` column, so that check can be made
# from the table itself rather than from memory.
LIGANDS = {
    "acetazolamide": "P00918",   # carbonic anhydrase II
    "dorzolamide":   "P00918",   # carbonic anhydrase II
    "erlotinib":     "P00533",   # EGFR
    "gefitinib":     "P00533",   # EGFR
    "roscovitine":   "P24941",   # CDK2
    "geldanamycin":  "P07900",   # HSP90-alpha
}

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
# PubChem renamed its SMILES properties: a request for the old "CanonicalSMILES" now answers
# with "ConnectivitySMILES", which has no stereochemistry. Both current names are requested and
# the stereo-bearing one preferred, so a chiral drug does not quietly lose its centres.
PUBCHEM_SMILES = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
                  "/property/SMILES,ConnectivitySMILES/JSON")
SMILES_KEYS = ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES")
TIMEOUT = 30


def get(url: str) -> str:
    # A generic client name: the request carries nothing about who is running it.
    req = urllib.request.Request(url, headers={"User-Agent": "nesso-example-pairs/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8")


def protein_sequence(acc: str) -> str:
    fasta = get(UNIPROT_FASTA.format(acc=acc))
    lines = [ln.strip() for ln in fasta.splitlines() if ln and not ln.startswith(">")]
    seq = "".join(lines).upper()
    if not seq:
        raise RuntimeError(f"no sequence returned for {acc}")
    return seq


def ligand_smiles(name: str) -> str:
    payload = json.loads(get(PUBCHEM_SMILES.format(name=urllib.parse.quote(name))))
    props = payload["PropertyTable"]["Properties"]
    if not props:
        raise RuntimeError(f"no record returned for {name}")
    for key in SMILES_KEYS:
        if props[0].get(key):
            return str(props[0][key])
    raise RuntimeError(f"no SMILES field for {name}; got {sorted(props[0])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="pairs table to write")
    args = ap.parse_args()

    print(f"fetching {len(PROTEINS)} protein sequence(s) from UniProt")
    seqs = {}
    for acc, label in PROTEINS.items():
        seqs[acc] = protein_sequence(acc)
        print(f"  {acc} {label}: {len(seqs[acc])} residues")
        time.sleep(0.2)

    print(f"fetching {len(LIGANDS)} ligand structure(s) from PubChem")
    smiles = {}
    for name in LIGANDS:
        smiles[name] = ligand_smiles(name)
        print(f"  {name}: {smiles[name]}")
        time.sleep(0.2)

    rows = []
    for acc, label in PROTEINS.items():
        for name, target_acc in LIGANDS.items():
            rows.append({"pair_id": f"{label}__{name}",
                         "protein_sequence": seqs[acc],
                         "ligand_smiles": smiles[name],
                         "uniprot": acc,
                         "target": label,
                         "ligand_name": name,
                         "cognate": int(target_acc == acc)})

    fields = ["pair_id", "protein_sequence", "ligand_smiles", "uniprot", "target", "ligand_name",
              "cognate"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} pairs -> {args.out}")
    print("next:\n"
          f"  python nesso/nesso_run.py plan --pairs {args.out} "
          f"--run-dir /cluster/scratch/$USER/nesso/example --pairs-per-shard 100\n"
          "  sbatch --array=0 --time=00:30:00 --cpus-per-task=2 --mem-per-cpu=8192 "
          "--gpus=1 --gres=gpumem:20g nesso/submit_nesso.sh /cluster/scratch/$USER/nesso/example")
    return 0


if __name__ == "__main__":
    sys.exit(main())
