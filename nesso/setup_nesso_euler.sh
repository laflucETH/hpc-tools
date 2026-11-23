#!/bin/bash -l
# =============================================================================
# Stage Nesso-1 on Euler in its own virtual environment.
#
#     bash nesso/setup_nesso_euler.sh
#
# This is NOT how you run Nesso-1 any more. Jobs run from the shared Apptainer image named in
# site.conf, which costs three files instead of the 41,358 this script installs. Two things still
# need it:
#
#   1. Building an image. build_image.sbatch takes its weights from the cache this stages, so the
#      image can only ever hold weights whose revision was recorded here.
#   2. Developing against a modified Nesso-1, where rebuilding an image per edit would be absurd.
#      Point a job at it with NESSO_VENV=<path>; it is never picked up on its own.
#
# Run on a login node. Compute nodes have no external network, so the packages, the model
# checkpoint, the chemical-component dictionary and the ESM-2 weights are all downloaded here,
# before any job is submitted. Nothing is computed.
#
# Python 3.12.8 (stack/2024-06) is used because nesso 1.0.0 requires Python >=3.10,<3.14. It is
# the same interpreter the group's Boltz-2 environment uses.
#
# Placement follows the Boltz-2 setup:
#   virtual environment   /cluster/project  (NFS: many small files)
#   model and ESM weights /cluster/work     (Lustre: a few large files)
# Override with NESSO_VENV and NESSO_CACHE.
#
# The resolved Hugging Face revision is written to $NESSO_CACHE/REVISION. Jobs read it and pass
# it to --model_revision, so every prediction of a run uses the weights staged here rather than
# whatever the Hub currently tags. Without that pin, a re-run months later can silently use
# different weights and produce numbers that no longer match the ones already analysed.
# =============================================================================

set -euo pipefail

VENV="${NESSO_VENV:-/cluster/project/beltrao/$USER/nesso/venv}"
CACHE="${NESSO_CACHE:-/cluster/work/beltrao/$USER/nesso/cache}"
# nesso is not on PyPI yet, and its GitHub repo carries no tags at all (checked 2026-11-16: only
# refs/heads/main), so the install is pinned to a commit. A branch name would let the code under
# a finished analysis change without the analysis changing, which is the thing a pin prevents.
# The model weights are pinned separately, below, to a Hugging Face revision -- that repo does
# tag v1.0.0, so code and weights are pinned by different means and both must be recorded.
NESSO_COMMIT="${NESSO_COMMIT:-6c72f66720d9d3447fd73c515cda963e39128b1f}"
NESSO_SPEC="${NESSO_SPEC:-git+https://github.com/recursionpharma/nesso.git@$NESSO_COMMIT}"
# cuEquivariance kernels speed up inference but are CUDA 12 only and may fail to build; the
# install is attempted and allowed to fail, and jobs pass --no_kernels unless NESSO_KERNELS=1.
TRY_KERNELS="${NESSO_TRY_KERNELS:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== modules ==="
module load stack/2024-06 >/dev/null 2>&1
module load python/3.12.8 >/dev/null 2>&1
module load cuda/12.8.0   >/dev/null 2>&1 || log "(cuda module not loaded; torch brings its own runtime)"
module load eth_proxy     >/dev/null 2>&1 || log "(eth_proxy not loaded; downloads may fail)"
python3 -V

# Create both targets up front, and say what to do if that is refused. The defaults live under
# the Beltrao group's storage; anyone outside that group has to point them somewhere writable,
# and a bare "Permission denied" from mkdir on a new user's very first command is a poor way to
# discover it. The mkdir itself is the test -- predicting permissions from the mode bits is
# unreliable on Lustre, where `stat` on these paths reports the mount stub rather than the
# directory (checked 2026-11-19: it claims root:root drwxr-xr-x for a group-writable directory).
make_dir() {
    local target="$1" name="$2" var="$3"
    mkdir -p "$target" 2>/dev/null && return 0
    echo "FATAL: cannot create the $name at" >&2
    echo "         $target" >&2
    echo "       Point $var somewhere you can write and run this again, for example" >&2
    echo "         $var=\$HOME/nesso/${name// /_} bash $0" >&2
    exit 1
}
make_dir "$(dirname "$VENV")" "virtual environment" NESSO_VENV
make_dir "$CACHE"             "weights cache"       NESSO_CACHE

if [ ! -x "$VENV/bin/python" ]; then
    log "=== creating $VENV ==="
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -V

log "=== installing nesso ($NESSO_SPEC); this pulls torch, several GB ==="
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "nesso @ $NESSO_SPEC"

if [ "$TRY_KERNELS" = "1" ]; then
    log "=== optional: cuEquivariance kernels (allowed to fail) ==="
    if "$VENV/bin/pip" install --quiet cuequivariance-torch cuequivariance-ops-torch-cu12; then
        log "    kernels installed"
    else
        log "    kernels NOT installed; jobs must pass --no_kernels"
    fi
fi

log "=== versions ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for p in ('nesso', 'torch', 'numpy', 'rdkit', 'lightning', 'transformers', 'huggingface_hub'):
    try:
        print(f'  {p}: {md.version(p)}')
    except Exception as e:
        print(f'  {p}: NOT INSTALLED ({type(e).__name__})')
try:
    import cuequivariance_torch  # noqa: F401
    print('  cuequivariance_torch: importable')
except Exception as e:
    print(f'  cuequivariance_torch: not importable ({type(e).__name__})')
import torch
print(f'  torch.cuda available here: {torch.cuda.is_available()} (False is expected on a login node)')
PY

log "=== staging checkpoint and CCD into $CACHE ==="
# nesso's own ensure_cache() is called so the files land under the exact cache keys the CLI will
# look for at run time. The default revision is v{installed version}; if that tag does not exist
# on the Hub the fallback is the main branch. Whichever resolves is written to $CACHE/REVISION.
NESSO_CACHE="$CACHE" "$VENV/bin/python" - <<'PY'
import os, sys
from pathlib import Path
from nesso.main import ensure_cache, get_default_model_revision, FALLBACK_REVISION

cache = Path(os.environ['NESSO_CACHE'])
cache.mkdir(parents=True, exist_ok=True)

last = None
for rev in (get_default_model_revision(), FALLBACK_REVISION):
    try:
        ccd, model_dir = ensure_cache(cache, revision=rev)
    except Exception as e:                       # tag missing, network refused, ...
        print(f'  revision {rev!r}: {type(e).__name__}: {e}')
        last = e
        continue
    print(f'  revision {rev!r}: OK')
    print(f'  ccd      : {ccd}')
    print(f'  model_dir: {model_dir}')
    (cache / 'REVISION').write_text(rev + '\n')
    break
else:
    sys.exit(f'could not stage the checkpoint from any revision: {last}')
PY

log "=== staging ESM-2 weights into $CACHE/huggingface ==="
# nesso embeds every protein sequence with ESM-2 650M at run time and would download it on first
# use; on a compute node that download cannot happen.
NESSO_CACHE="$CACHE" "$VENV/bin/python" - <<'PY'
import os, torch
from pathlib import Path
from nesso.data.esm import setup_esm_model, extract_esm_embedding, DEFAULT_ESM2_MODEL

hf = Path(os.environ['NESSO_CACHE']) / 'huggingface'
model, tok = setup_esm_model(DEFAULT_ESM2_MODEL, torch.device('cpu'), cache_dir=hf)
emb = extract_esm_embedding('MKTAYIAKQRQISFVKSHFSRQ', model, tok)
print(f'  {DEFAULT_ESM2_MODEL} staged; test embedding shape {tuple(emb.shape)}')
PY

echo "$NESSO_COMMIT" > "$CACHE/NESSO_COMMIT"

log "=== cache contents ==="
du -sh "$CACHE" 2>/dev/null || true
echo "  weights revision pin: $(cat "$CACHE/REVISION" 2>/dev/null || echo MISSING)"
echo "  code commit pin     : $(cat "$CACHE/NESSO_COMMIT" 2>/dev/null || echo MISSING)"

cat <<SUMMARY

Done. Point jobs at this install with:
    NESSO_VENV=$VENV
    NESSO_CACHE=$CACHE

Next: run the tutorial check on a GPU to confirm the install predicts before using it on real
inputs (nesso/README.md, "Checking the install").
SUMMARY
