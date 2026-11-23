#!/bin/bash -l
#SBATCH --job-name=nesso
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=/cluster/scratch/%u/nesso/slurm/%x-%A_%a.out
#SBATCH --error=/cluster/scratch/%u/nesso/slurm/%x-%A_%a.err
# =============================================================================
# Score one shard of a Nesso-1 run on one GPU.
#
#   sbatch --array=0-11 nesso/submit_nesso.sh /cluster/scratch/$USER/nesso/<run>
#
# One array task = one shard of the plan written by `nesso_run.py plan`. The task predicts the
# shard's pairs, then collapses its output into one CSV and one zip and clears the shard's
# scratch tree -- but only if every pair was scored and the archive verified. A failed shard
# keeps all of its files, because that is when they are needed.
#
# The resource directives above are deliberate starting points for a pilot, NOT measured values.
# Size a real run from `seff <jobid>` on one or two pilot tasks and pass the measured numbers on
# the sbatch command line; see nesso/README.md, "Sizing a run".
#
# A task that finds its shard's CSV already present exits 0 without recomputing, so a partly
# finished array can simply be resubmitted.
#
# Nesso-1 runs from the shared Apptainer image named in nesso/site.conf: one 5.5 GB file the
# whole group reads, rather than a 41,358-file virtual environment per person. Setting NESSO_VENV
# runs a virtual environment of your own instead.
#
# Environment (all optional):
#   NESSO_IMAGE    a particular image                   $NESSO_HOME/current/nesso.sif
#   NESSO_VENV     run this virtual environment instead (unset = the shared image)
#   NESSO_CACHE    weights -- the venv engine only      /cluster/work/beltrao/$USER/nesso/cache
#   NESSO_KERNELS  1 to use the cuEquivariance kernels  (default 0, i.e. --no_kernels)
#   NESSO_SEED     prediction seed                      42
#   EXPECT_GPU     refuse to run on another GPU type    (empty = no check)
#   NESSO_DRIVER   path to nesso_run.py                 (found automatically; see below)
# =============================================================================
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

RUN_DIR="${1:-${NESSO_RUN_DIR:-}}"
[ -n "$RUN_DIR" ] || fatal "name the run directory (first argument, or NESSO_RUN_DIR=...)"
[ -d "$RUN_DIR" ] || fatal "no such run directory: $RUN_DIR"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || fatal "submit with --array=<shard numbers>, e.g. --array=0-11"
case "$SLURM_ARRAY_TASK_ID" in ''|*[!0-9]*) fatal "the array index must be a whole number" ;; esac
SHARD=$(printf 'shard_%04d' "$((10#$SLURM_ARRAY_TASK_ID))")

CACHE="${NESSO_CACHE:-/cluster/work/beltrao/$USER/nesso/cache}"
SEED="${NESSO_SEED:-42}"
# sbatch COPIES the job script into the node's spool directory before running it, so
# ${BASH_SOURCE[0]} points at that copy and not at the checkout -- the driver is not beside it.
# The checkout is therefore found by trying, in order: an explicit override, the directory the
# job was submitted from (the normal case, since a run is submitted from the checkout), the
# script's own directory (when it is run directly rather than through sbatch), and the group's
# Euler checkout.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER=""
for cand in "${NESSO_DRIVER:-}"             "${SLURM_SUBMIT_DIR:-}/nesso/nesso_run.py"             "$HERE/nesso_run.py"             "/cluster/project/beltrao/$USER/repos/hpc-tools/nesso/nesso_run.py"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then DRIVER="$cand"; break; fi
done

[ -n "$DRIVER" ]          || fatal "cannot find nesso_run.py; submit from the hpc-tools checkout,
       or set NESSO_DRIVER=/path/to/nesso/nesso_run.py"

# --- which Nesso-1 runs -------------------------------------------------------------------------
# The shared image, unless a virtual environment is named. NESSO_VENV is deliberately NOT guessed
# and there is no fallback to one: if the image were missing and this quietly used whatever venv
# happened to be in the usual place, two people could score the same pairs against two different
# builds of Nesso-1 with nothing in either log to say so. A missing image is an error instead.
TOOLS="$(cd "$(dirname "$DRIVER")" && pwd)"
PREDICT_EXTRA=()
if [ -n "${NESSO_VENV:-}" ]; then
    [ -x "$NESSO_VENV/bin/nesso" ] || fatal "NESSO_VENV is set but has no nesso CLI at
       $NESSO_VENV/bin/nesso; run nesso/setup_nesso_euler.sh, or unset NESSO_VENV for the image"
    [ -d "$CACHE" ] || fatal "no cache at $CACHE; run nesso/setup_nesso_euler.sh first"
    PREDICT=("$NESSO_VENV/bin/nesso" predict)
    PYTHON=("$NESSO_VENV/bin/python")
    REVISION="$(cat "$CACHE/REVISION" 2>/dev/null || true)"
    ENGINE="venv $NESSO_VENV"
    # The image supplies these itself; a venv has to be given them by hand.
    PREDICT_EXTRA=(--cache "$CACHE")
    [ -n "$REVISION" ] && PREDICT_EXTRA+=(--model_revision "$REVISION")
    [ "${NESSO_KERNELS:-0}" = "1" ] || PREDICT_EXTRA+=(--no_kernels)
    # Compute nodes have no external network. Offline mode turns a missing cached asset into an
    # immediate, readable error instead of a long retry-and-hang on the Hugging Face endpoint.
    export NESSO_CACHE="$CACHE"
    export HF_HOME="$CACHE/huggingface"
    export HF_HUB_CACHE="$CACHE/huggingface"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    WRAPPER="$TOOLS/nesso"
    [ -x "$WRAPPER" ] || fatal "no nesso wrapper at $WRAPPER; submit from the hpc-tools checkout"
    INFO="$("$WRAPPER" info 2>&1)" || fatal "$INFO"
    REVISION="$(printf '%s\n' "$INFO" | sed -n 's/^weights_revision *: *//p')"
    ENGINE="image $(printf '%s\n' "$INFO" | sed -n 's/^resolves to: *//p')"
    PREDICT=("$WRAPPER" predict)
    PYTHON=("$WRAPPER" python)
    # PREDICT_EXTRA stays empty: the wrapper adds --cache, --model_revision and --no_kernels, and
    # the image sets the offline Hugging Face variables in its own %environment.
fi
export TOKENIZERS_PARALLELISM=false

IN_DIR="$RUN_DIR/inputs/$SHARD"
OUT_DIR="$RUN_DIR/work/$SHARD"
SHARD_CSV="$RUN_DIR/shards/$SHARD.csv"

if [ -f "$SHARD_CSV" ]; then
    echo "$SHARD already collected ($SHARD_CSV); nothing to do"
    exit 0
fi
[ -d "$IN_DIR" ] || fatal "no inputs for $SHARD at $IN_DIR (is the array range larger than the plan?)"

echo "=== $SHARD on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
if [ -n "${EXPECT_GPU:-}" ]; then
    GOT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$GOT" in
        *"$EXPECT_GPU"*) ;;
        *) fatal "expected GPU '$EXPECT_GPU' but got '$GOT'; scores from different GPU types are
       not necessarily bit-identical, so a run is kept to one type" ;;
    esac
fi

module load stack/2024-06 >/dev/null 2>&1
module load cuda/12.8.0   >/dev/null 2>&1 || true

N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.yaml' | wc -l)
echo "predicting $N_IN pair(s) from $IN_DIR"
echo "engine=$ENGINE"
echo "revision=${REVISION:-<unpinned>} seed=$SEED kernels=${NESSO_KERNELS:-0}"

START=$(date +%s)
"${PREDICT[@]}" "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    ${PREDICT_EXTRA[@]+"${PREDICT_EXTRA[@]}"} \
    --accelerator gpu \
    --devices 1 \
    --num_workers "${SLURM_CPUS_PER_TASK:-2}" \
    --seed "$SEED" \
    --require_affinity
RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "nesso predict exited $RC after ${ELAPSED}s ($(awk -v e=$ELAPSED -v n=$N_IN 'BEGIN{if(n)printf "%.2f", e/n; else print "NA"}') s/pair)"

if [ "$RC" -ne 0 ]; then
    echo "prediction failed; leaving $OUT_DIR and $IN_DIR on scratch for debugging" >&2
    exit 1
fi

# Collapse this shard: CSV + verified zip, then clear its scratch tree. --allow-cleanup is passed
# only here, after a clean exit, so a failed shard can never have its evidence deleted.
"${PYTHON[@]}" "$DRIVER" collect \
    --run-dir "$RUN_DIR" \
    --shard "$SHARD" \
    --allow-cleanup
CRC=$?
echo "=== $SHARD finished at $(date '+%Y-%m-%d %H:%M:%S'), collect exited $CRC ==="
exit "$CRC"
