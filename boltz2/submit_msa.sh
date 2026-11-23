#!/bin/bash -l
#SBATCH --job-name=boltz2_msa
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --output=/cluster/scratch/%u/boltz2/slurm/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/boltz2/slurm/%x-%j.err
# =============================================================================
# Fetch the alignments (MSAs) a set of inputs needs, into a store shared across runs.
#
#   mkdir -p /cluster/scratch/$USER/boltz2/slurm
#   sbatch boltz2/submit_msa.sh <inputs_dir> <msa_store>
#
# One alignment per distinct protein sequence, from the public ColabFold server, named by the
# sha1 of the sequence. The store is append-only and is the point of the whole step: a sequence
# fetched once is never fetched again, by this run or any later one, which is both much faster
# and much politer to a free shared service.
#
# An input with more than one DISTINCT protein chain also gets a paired alignment for that exact
# group of chains, cached under <store>/paired/<hash>/. Pairing is what lets the model see which
# rows of two chains' alignments come from the same organism, and it is the part of a multimer
# prediction that a per-sequence store cannot supply on its own.
#
# Why a separate step at all, rather than boltz's own --use_msa_server during the prediction:
# that would hold a GPU idle while it waits on a web service, and would re-fetch the same
# alignment for every input that protein appears in. Here the waiting happens once, on a CPU.
#
# This job needs network, so it loads eth_proxy. It is CPU-only: almost all of its time is spent
# waiting for the server.
#
# The store can live in project storage. Alignments are few (one per protein, not one per
# prediction) and are worth keeping; it is the per-prediction files that must stay on scratch.
# =============================================================================
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

INPUTS="${1:-}"
STORE="${2:-}"
[ -n "$INPUTS" ] && [ -n "$STORE" ] || fatal "usage: sbatch submit_msa.sh <inputs_dir> <msa_store>"
[ -d "$INPUTS" ] || fatal "no such input directory: $INPUTS"

# Where this tool lives. sbatch runs a COPY of this script from a spool directory, so
# BASH_SOURCE does not point into the checkout -- hence SLURM's own record of the submitted path
# is tried first. Without it, submitting by absolute path from anywhere else fails.
find_tool_dir() {
    local c d
    if [ -n "${HPC_TOOLS_BOLTZ2_DIR:-}" ]; then printf '%s\n' "$HPC_TOOLS_BOLTZ2_DIR"; return; fi
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        c=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
            | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
        d=$(dirname "${c:-/nonexistent}")
        [ -f "$d/boltz2" ] && { printf '%s\n' "$d"; return; }
    fi
    d=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)
    [ -n "$d" ] && [ -f "$d/boltz2" ] && { printf '%s\n' "$d"; return; }
    printf '%s\n' "${SLURM_SUBMIT_DIR:-$PWD}/boltz2"
}
HERE="$(find_tool_dir)"
[ -f "$HERE/boltz2" ] || fatal "cannot find the boltz2 wrapper (looked in '$HERE'); set HPC_TOOLS_BOLTZ2_DIR to the boltz2 directory of the hpc-tools checkout"

module load eth_proxy >/dev/null 2>&1 || fatal "module load eth_proxy failed; this job needs network"

echo "=== alignments for $INPUTS into $STORE ==="
echo "    started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
"$HERE/boltz2" python "$HERE/boltz2_run.py" msa \
    --inputs "$INPUTS" --msa-store "$STORE" \
    ${MSA_HOST:+--host "$MSA_HOST"} \
    ${MSA_BATCH:+--batch "$MSA_BATCH"} \
    ${MSA_PAUSE:+--pause "$MSA_PAUSE"} \
    ${MSA_PAIRING:+--pairing-strategy "$MSA_PAIRING"}
RC=$?
echo "=== finished $(date '+%Y-%m-%d %H:%M:%S'), exit $RC ==="
# Exit 1 means some sequences came back without an alignment. The store keeps what did arrive, so
# the job can simply be run again for the rest.
exit "$RC"
