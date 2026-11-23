#!/bin/bash -l
#SBATCH --job-name=nesso_cmp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8192
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=/cluster/scratch/%u/nesso/slurm/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/nesso/slurm/%x-%j.err
# =============================================================================
# Does the image score the same as the virtual environment the reference scores ran in?
#
#   sbatch nesso/compare_to_venv.sbatch <run_dir> [shard]
#
# <run_dir>/<shard> is a shard planned by `nesso_run.py plan`; its pairs are scored three times
# ON ONE GPU, IN ONE JOB:
#
#   venv     the old environment at $NESSO_VENV
#   image_a  the image
#   image_b  the image again
#
# Everything but the software is held constant: same inputs, same options, same card, same job.
# image_a against image_b is the yardstick, because Nesso-1 on a GPU is not deterministic even
# with a fixed seed -- three runs of the same venv over the same 24 pairs disagreed by up to
# 0.254 in affinity_pred_value. Asking the image to match the venv exactly would therefore fail
# for a reason that has nothing to do with the image. The question is whether venv-against-image
# is any further apart than image-against-image.
#
# Three passes over the shard, so keep it small: 10-25 pairs is plenty. Memory is 16 GB, not the
# 14 GB a prediction shard is sized at: the first run of this job (21641487, 24 pairs) peaked at
# 13.90 GB, 99.3% of 14 GB, because the image needs about 2 GB more than the venv does and the
# peak here is taken across all three passes. The walltime allows three passes over ~25 pairs
# with room for the model to load three times; 21641487 took 25:55 on an RTX 3090.
# =============================================================================
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

RUN_DIR="${1:-}"
SHARD_IN="${2:-0}"
[ -n "$RUN_DIR" ] || fatal "usage: sbatch compare_to_venv.sbatch <run_dir> [shard]"
[ -d "$RUN_DIR" ] || fatal "no such run directory: $RUN_DIR"
case "$SHARD_IN" in shard_*) SHARD="$SHARD_IN" ;; *) SHARD=$(printf 'shard_%04d' "$((10#$SHARD_IN))") ;; esac

HERE="${HPC_TOOLS_NESSO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$HERE/nesso" ] || HERE="${SLURM_SUBMIT_DIR:-$PWD}/nesso"
[ -f "$HERE/nesso" ] || fatal "cannot find the nesso wrapper; submit from the hpc-tools checkout"

VENV="${NESSO_VENV:-/cluster/project/beltrao/$USER/nesso/venv}"
VENV_CACHE="${NESSO_CACHE:-/cluster/work/beltrao/$USER/nesso/cache}"
SEED="${NESSO_SEED:-42}"
[ -x "$VENV/bin/nesso" ] || fatal "no nesso at $VENV/bin/nesso (set NESSO_VENV)"
[ -d "$VENV_CACHE" ]     || fatal "no cache at $VENV_CACHE (set NESSO_CACHE)"

IN_DIR="$RUN_DIR/inputs/$SHARD"
[ -d "$IN_DIR" ] || fatal "no inputs at $IN_DIR"
OUT="$RUN_DIR/compare"
mkdir -p "$OUT"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.yaml' | wc -l)
echo "=== comparing on $GPU_NAME, $N_IN pair(s) from $IN_DIR ==="

module load stack/2024-06 >/dev/null 2>&1
module load cuda/12.8.0   >/dev/null 2>&1 || true

# Options common to all three passes. The wrapper adds --cache, --model_revision and --no_kernels
# to the image passes on its own; the venv has to be given them by hand, or the two would differ
# in their options as well as in their software and the comparison would mean nothing.
COMMON=(--accelerator gpu --devices 1 --num_workers "${SLURM_CPUS_PER_TASK:-2}"
        --seed "$SEED" --require_affinity)
REVISION="$(cat "$VENV_CACHE/REVISION" 2>/dev/null || true)"
IMG_REVISION="$("$HERE/nesso" info 2>/dev/null | sed -n 's/^weights_revision *: *//p')"
echo "venv revision : ${REVISION:-<unpinned>}"
echo "image revision: ${IMG_REVISION:-<unknown>}"
[ -n "$REVISION" ] && [ -n "$IMG_REVISION" ] && [ "$REVISION" != "$IMG_REVISION" ] && \
    fatal "the venv holds weights '$REVISION' and the image '$IMG_REVISION'; a difference in
       scores would then be the weights, not the environment"

echo "=== 1/3 venv ==="
NESSO_CACHE="$VENV_CACHE" HF_HOME="$VENV_CACHE/huggingface" HF_HUB_CACHE="$VENV_CACHE/huggingface" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
"$VENV/bin/nesso" predict "$IN_DIR" --out_dir "$OUT/venv" --cache "$VENV_CACHE" \
    ${REVISION:+--model_revision "$REVISION"} "${COMMON[@]}" --no_kernels \
    || fatal "venv pass failed"

for rep in a b; do
    echo "=== $( [ "$rep" = a ] && echo 2 || echo 3 )/3 image ($rep) ==="
    "$HERE/nesso" predict "$IN_DIR" --out_dir "$OUT/image_$rep" "${COMMON[@]}" \
        || fatal "image pass $rep failed"
done

echo "=== comparison ==="
"$HERE/nesso" python "$HERE/compare_runs.py" \
    --run "venv=$OUT/venv" --run "image_a=$OUT/image_a" --run "image_b=$OUT/image_b" \
    --baseline image_a:image_b \
    --out "$OUT/comparison.csv"
RC=$?
echo "(GPU: $GPU_NAME; table: $OUT/comparison.csv)"
exit "$RC"
