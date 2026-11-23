#!/bin/bash -l
#SBATCH --job-name=boltz2
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=9216
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=/cluster/scratch/%u/boltz2/slurm/%x-%A_%a.out
#SBATCH --error=/cluster/scratch/%u/boltz2/slurm/%x-%A_%a.err
# =============================================================================
# Predict one shard of a Boltz-2 run on one GPU.
#
#   mkdir -p /cluster/scratch/$USER/boltz2/slurm
#   sbatch --array=0-11 boltz2/submit_boltz2.sh /cluster/scratch/$USER/boltz2/<run>
#
# One array task = one shard planned by `boltz2_run.py plan`. The task predicts the shard's
# inputs, then collapses its output into one CSV and one zip and clears the shard's scratch tree
# -- but only once every input is complete and the archive verifies. A failed shard keeps all of
# its files, because that is when they are needed.
#
# THE RESOURCE LINES ABOVE ARE ONLY A DEFAULT, for small complexes (a few hundred residues) at
# 50 inputs per shard. Do not reuse them for a different workload. Ask what this run needs:
#
#   boltz2 python boltz2_run.py size --inputs <your inputs>
#
# which prints one sbatch line per size band, and override on the command line, e.g.
#
#   sbatch --array=12-14 --gres=gpumem:40g --time=06:00:00 submit_boltz2.sh <run_dir>
#
# Command-line options beat the #SBATCH lines, so one plan can be submitted as several arrays
# with different resources -- which is the point of banding: a run of mostly small complexes
# should not be charged the memory its three biggest need.
#
# The protocol (samples, seed, steering, recycling, ...) is NOT set here. It is fixed by the plan
# and read back from it, so every shard of a run is predicted the same way and the scores can be
# compared with each other. To change it, plan a new run.
#
# A task whose shard is already collected exits 0 without recomputing, so a partly finished array
# can be resubmitted as it is. Boltz also skips predictions that already exist, so a resubmitted
# shard redoes only what is missing.
#
# Environment (all optional):
#   BOLTZ2_HOME / BOLTZ2_IMAGE   which image to use (default: site.conf)
#   EXPECT_GPU                   refuse to run on any other GPU type (e.g. EXPECT_GPU="RTX 4090")
# =============================================================================
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

RUN_DIR="${1:-${BOLTZ2_RUN_DIR:-}}"
[ -n "$RUN_DIR" ] || fatal "name the run directory (first argument, or BOLTZ2_RUN_DIR=...)"
[ -d "$RUN_DIR" ] || fatal "no such run directory: $RUN_DIR"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || fatal "submit with --array=<shard numbers>, e.g. --array=0-11"
case "$SLURM_ARRAY_TASK_ID" in ''|*[!0-9]*) fatal "the array index must be a whole number" ;; esac
SHARD=$(printf 'shard_%04d' "$((10#$SLURM_ARRAY_TASK_ID))")

# Where this tool lives. sbatch runs a COPY of this script from a spool directory, so
# BASH_SOURCE does not point into the checkout -- which is why SLURM's own record of the
# submitted path is tried first. Without that, submitting from anywhere other than the checkout
# itself fails, and an absolute path (what a notebook or a cron job would use) is exactly that
# case.
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
BOLTZ2="$HERE/boltz2"
DRIVER="$HERE/boltz2_run.py"

IN_DIR="$RUN_DIR/inputs/$SHARD"
OUT_DIR="$RUN_DIR/work/$SHARD"
RECEIPT="$RUN_DIR/shards/$SHARD.job.json"

if [ -f "$RUN_DIR/shards/$SHARD.done" ]; then
    echo "$SHARD already finished; nothing to do"
    exit 0
fi
[ -d "$IN_DIR" ] || fatal "no inputs for $SHARD at $IN_DIR (is the array range larger than the plan?)"

echo "=== $SHARD on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S') ==="
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU: ${GPU_NAME:-none} (${GPU_MEM:-unknown})"
if [ -n "${EXPECT_GPU:-}" ]; then
    case "$GPU_NAME" in
        *"$EXPECT_GPU"*) ;;
        # Boltz-2 is not bit-reproducible across GPU types: the same input and seed can shift the
        # scores, and runtime varies several-fold, so a run meant to be internally comparable is
        # kept to one card type.
        *) fatal "expected GPU '$EXPECT_GPU' but got '${GPU_NAME:-none}'" ;;
    esac
fi

# The protocol the plan fixed, as boltz options.
mapfile -t PROTO < <("$BOLTZ2" python "$DRIVER" protocol --run-dir "$RUN_DIR") \
    || fatal "could not read the protocol from $RUN_DIR/plan/plan.json"
[ "${#PROTO[@]}" -gt 0 ] || fatal "the plan named no protocol"

IMAGE_INFO="$("$BOLTZ2" info)"
IMAGE_PATH="$(printf '%s\n' "$IMAGE_INFO" | sed -n 's/^resolves to: //p')"
IMAGE_SHA="$(printf '%s\n' "$IMAGE_INFO" | sed -n 's/^image_sha256 *: *//p')"
N_IN=$(find "$IN_DIR" -maxdepth 1 \( -name '*.yaml' -o -name '*.yml' -o -name '*.fasta' -o -name '*.fa' \) | wc -l)
echo "predicting $N_IN input(s) from $IN_DIR"
echo "protocol: ${PROTO[*]}"

mkdir -p "$RUN_DIR/shards"
# Written before the prediction starts, so a job that is killed still leaves a record of where it
# ran; rewritten afterwards with the elapsed time, which `size --from-run` reads back.
write_receipt() {
    cat > "$RECEIPT" <<JSON
{"shard": "$SHARD", "job_id": "${SLURM_JOB_ID:-none}",
 "array_job_id": "${SLURM_ARRAY_JOB_ID:-none}", "task_id": "${SLURM_ARRAY_TASK_ID:-none}",
 "host": "$(hostname)", "gpu": "${GPU_NAME:-unknown}", "gpu_memory": "${GPU_MEM:-unknown}",
 "cpus": "${SLURM_CPUS_PER_TASK:-unknown}", "mem_per_cpu": "${SLURM_MEM_PER_CPU:-unknown}",
 "image": "${IMAGE_PATH:-unknown}", "image_sha256": "${IMAGE_SHA:-unknown}",
 "started": "$STARTED", "n_inputs": $N_IN, "elapsed_s": ${1:-null}, "exit_code": ${2:-null}}
JSON
}
STARTED="$(date '+%Y-%m-%dT%H:%M:%S')"
write_receipt

START=$(date +%s)
"$BOLTZ2" predict "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --accelerator gpu \
    --devices 1 \
    --num_workers "${SLURM_CPUS_PER_TASK:-2}" \
    "${PROTO[@]}"
RC=$?
ELAPSED=$(( $(date +%s) - START ))
write_receipt "$ELAPSED" "$RC"
echo "boltz predict exited $RC after ${ELAPSED}s ($(awk -v e=$ELAPSED -v n=$N_IN \
    'BEGIN{if(n)printf "%.1f", e/n; else print "NA"}') s/input)"

# Collect either way. On a clean exit the shard is closed and its tree cleared; on a failure the
# table still records what did finish, and --allow-cleanup is withheld so every file stays put.
CLEAN=()
[ "$RC" -eq 0 ] && CLEAN=(--allow-cleanup)
"$BOLTZ2" python "$DRIVER" collect --run-dir "$RUN_DIR" --shard "$SHARD" ${CLEAN[@]+"${CLEAN[@]}"}
CRC=$?
if [ "$RC" -ne 0 ]; then
    echo "prediction failed; $OUT_DIR and $IN_DIR are kept on scratch for debugging" >&2
    exit 1
fi
echo "=== $SHARD finished at $(date '+%Y-%m-%d %H:%M:%S'), collect exited $CRC ==="
exit "$CRC"
