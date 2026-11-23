#!/bin/bash -l
#SBATCH --job-name=boltz2_cmp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=9216
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=/cluster/scratch/%u/boltz2/slurm/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/boltz2/slurm/%x-%j.err
# =============================================================================
# Does the image fold the same as the virtual environment the reference folds ran in?
#
#   sbatch boltz2/compare_to_venv.sbatch <run_dir> <shard>
#
# <run_dir>/<shard> is a shard planned by `boltz2_run.py plan`; its inputs are folded three times
# ON ONE GPU, IN ONE JOB:
#
#   venv     the old environment at $BOLTZ_VENV
#   image_a  the image
#   image_b  the image again
#
# Everything but the software is held constant: same inputs, same options, same card, same job.
# image_a against image_b is the yardstick, because Boltz-2 on a GPU is not deterministic even
# with a fixed seed -- the same software disagrees with itself by up to ~0.1 ipTM. Asking the
# image to match the environment exactly would therefore fail for a reason that has nothing to do
# with the environment. The question is whether venv-against-image is any further apart than
# image-against-image.
#
# Three folds of the shard, so size the shard accordingly: 10-20 small pairs is plenty.
# =============================================================================
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

RUN_DIR="${1:-}"
SHARD_IN="${2:-0}"
[ -n "$RUN_DIR" ] || fatal "usage: sbatch compare_to_venv.sbatch <run_dir> [shard]"
[ -d "$RUN_DIR" ] || fatal "no such run directory: $RUN_DIR"
case "$SHARD_IN" in shard_*) SHARD="$SHARD_IN" ;; *) SHARD=$(printf 'shard_%04d' "$((10#$SHARD_IN))") ;; esac

HERE="${HPC_TOOLS_BOLTZ2_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$HERE/boltz2" ] || HERE="${SLURM_SUBMIT_DIR:-$PWD}/boltz2"
[ -f "$HERE/boltz2" ] || fatal "cannot find the boltz2 wrapper; submit from the hpc-tools checkout"

VENV="${BOLTZ_VENV:-/cluster/project/beltrao/lucla/boltz2/venv}"
VENV_CACHE="${BOLTZ_CACHE_DIR:-/cluster/work/beltrao/lucla/boltz2/cache}"
[ -x "$VENV/bin/boltz" ] || fatal "no boltz at $VENV/bin/boltz (set BOLTZ_VENV)"
[ -d "$VENV_CACHE" ]     || fatal "no cache at $VENV_CACHE (set BOLTZ_CACHE_DIR)"

IN_DIR="$RUN_DIR/inputs/$SHARD"
[ -d "$IN_DIR" ] || fatal "no inputs at $IN_DIR"
OUT="$RUN_DIR/compare"
mkdir -p "$OUT"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.yaml' | wc -l)
echo "=== comparing on $GPU_NAME, $N_IN pair(s) from $IN_DIR ==="

mapfile -t PROTO < <("$HERE/boltz2" python "$HERE/boltz2_run.py" protocol --run-dir "$RUN_DIR") \
    || fatal "could not read the protocol from the plan"
echo "protocol: ${PROTO[*]}"

# The wrapper's defaults have to be applied to the venv by hand, or the two would differ in their
# options as well as in their software, and the comparison would mean nothing.
VENV_ARGS=(--out_dir "$OUT/venv" --cache "$VENV_CACHE" --accelerator gpu --devices 1
           --num_workers "${SLURM_CPUS_PER_TASK:-2}" --no_kernels)
for a in "${PROTO[@]}"; do
    [ "$a" = "--no_potentials" ] && continue        # ours, not boltz's: absence is "off"
    VENV_ARGS+=("$a")
done

echo "=== 1/3 venv ==="
BOLTZ_CACHE="$VENV_CACHE" "$VENV/bin/boltz" predict "$IN_DIR" "${VENV_ARGS[@]}" || fatal "venv fold failed"

for rep in a b; do
    echo "=== $( [ $rep = a ] && echo 2 || echo 3 )/3 image ($rep) ==="
    "$HERE/boltz2" predict "$IN_DIR" --out_dir "$OUT/image_$rep" \
        --accelerator gpu --devices 1 --num_workers "${SLURM_CPUS_PER_TASK:-2}" \
        "${PROTO[@]}" || fatal "image fold $rep failed"
done

echo "=== comparison ==="
"$HERE/boltz2" python "$HERE/compare_runs.py" \
    --run "venv=$OUT/venv" --run "image_a=$OUT/image_a" --run "image_b=$OUT/image_b" \
    --baseline image_a:image_b \
    --out "$OUT/comparison.csv"
RC=$?
echo "(GPU: $GPU_NAME; table: $OUT/comparison.csv)"
exit "$RC"
