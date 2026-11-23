#!/bin/bash
# =============================================================================
# Does this installation work? Plans one tiny prediction and prints how to run it.
#
#   boltz2/check_install.sh [work_dir]
#
# Checks, in order: the wrapper resolves an image; the image's checksum matches what was
# recorded when it was built; the driver's own tests pass; Boltz's CLI starts; and a one-input
# run can be planned end to end. It does NOT predict anything -- that needs a GPU, and the last
# line tells you how to submit it.
#
# Safe to run on a login node: everything here is seconds of CPU.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-/cluster/scratch/$USER/boltz2/install_check}"
BOLTZ2="$HERE/boltz2"
ACCESSION="${ACCESSION:-P61823}"        # bovine ribonuclease A, 150 aa: small, and folds fast
LIGAND="${LIGAND:-C5P}"                 # a CCD code, so the check also proves mols/ is readable

ok() { printf '  ok    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; FAILED=1; }
FAILED=0

echo "=== 1. the image ==="
if "$BOLTZ2" info --verify; then ok "image resolves and its checksum matches"
else bad "the wrapper could not resolve or verify an image"; fi

echo
echo "=== 2. the driver's tests ==="
# The tests need pytest and PyYAML but not Boltz, so they run under any Python. The image
# deliberately does NOT carry pytest: it holds the reference environment's package list exactly,
# and adding a test dependency to it would make it no longer that environment.
if "$BOLTZ2" python -c 'import pytest' 2>/dev/null; then TESTPY="$BOLTZ2 python"
elif python3 -c 'import pytest, yaml' 2>/dev/null; then TESTPY="python3"
else TESTPY=""; fi
if [ -z "$TESTPY" ]; then
    echo "  skip  no Python here has pytest; run the tests from any environment that does:"
    echo "        python3 -m pytest $HERE/test_boltz2_run.py -q"
elif $TESTPY -m pytest "$HERE/test_boltz2_run.py" -q 2>&1 | tail -3; then
    ok "driver tests pass"
else bad "driver tests failed"; fi

echo
echo "=== 3. boltz starts ==="
if "$BOLTZ2" predict --help >/dev/null 2>&1; then ok "boltz predict --help runs"
else bad "boltz would not start (a numba cache problem looks like this)"; fi

echo
echo "=== 4. plan one prediction ==="
mkdir -p "$WORK/inputs" || { bad "cannot write to $WORK"; exit 1; }
# The sequence is fetched by accession, never pasted, so this file cannot drift from what the
# accession actually holds.
if ! "$BOLTZ2" python - "$ACCESSION" "$LIGAND" "$WORK/inputs/check.yaml" <<'PY'
import sys, urllib.request, yaml
acc, ccd, out = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
                             headers={"User-Agent": "hpc-tools-boltz2-check/1.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    seq = "".join(l.strip() for l in r.read().decode().splitlines() if not l.startswith(">"))
doc = {"version": 1, "sequences": [
    {"protein": {"id": "A", "sequence": seq, "msa": "empty"}},
    {"ligand": {"id": "B", "ccd": ccd}}]}
with open(out, "w") as fh:
    yaml.safe_dump(doc, fh, sort_keys=False)
print(f"  {acc}: {len(seq)} residues, plus CCD {ccd}")
PY
then
    bad "could not build the input (no network on this node? try a login node)"
else
    ok "input written to $WORK/inputs/check.yaml"
    # msa: empty keeps this offline. Single-sequence mode is much less accurate and is right
    # only for a check like this one, never for real work.
    if "$BOLTZ2" python "$HERE/boltz2_run.py" plan \
            --inputs "$WORK/inputs" --run-dir "$WORK/run" --per-shard 1 --samples 1 --override
    then ok "planned"
    else bad "planning failed"; fi
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "All checks passed. To actually predict it:"
    echo
    echo "    mkdir -p /cluster/scratch/\$USER/boltz2/slurm"
    echo "    sbatch --array=0-0 --time=00:20:00 $HERE/submit_boltz2.sh $WORK/run"
    echo
    echo "Note this check used msa: empty (single-sequence mode), which is fast but much less"
    echo "accurate. Real work fetches alignments first -- see README.md, step 2."
else
    echo "Something failed above. README.md, 'If something goes wrong', covers the usual causes."
fi
exit "$FAILED"
