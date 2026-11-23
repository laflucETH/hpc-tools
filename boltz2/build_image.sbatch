#!/bin/bash -l
#SBATCH --job-name=boltz2_build
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=80G
#SBATCH --output=/cluster/scratch/%u/hpc-tools/boltz2_build/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/hpc-tools/boltz2_build/%x-%j.err
# =============================================================================
# Build the Boltz-2 Apptainer image on a compute node and install it as a new release.
#
#   mkdir -p /cluster/scratch/$USER/hpc-tools/boltz2_build
#   sbatch boltz2/build_image.sbatch                       # weights downloaded from Hugging Face
#   SRC_CACHE=<dir with the ckpts and mols.tar> sbatch --export=ALL boltz2/build_image.sbatch
#
# Writes $BOLTZ2_HOME/<release>/boltz2.sif and BUILD_INFO.txt. An existing release is never
# replaced, and `current` is not moved: a new image becomes the default only after it has been
# checked (README, "Releases").
#
# The weights are checked against fixed md5 sums before they go into the image, whichever way
# they arrive, so an image can only ever hold the files the reference folds used.
#
# Resources measured on the 2026-11-16 builds, both of which reached the self-test: MaxRSS 16.2 GB
# (job 21181125) and 21.8 GB (job 21184527) of the 32 GB requested, under 4 min of the 3 h. The
# peak is mksquashfs compressing the ~12 GB tree, and it varies by 5 GB between identical builds,
# which is why this is not trimmed to the larger observation either.
# =============================================================================
set -euo pipefail

HERE="${HPC_TOOLS_BOLTZ2_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# sbatch runs a copy of this script from a spool directory, so BASH_SOURCE does not point into
# the checkout; fall back to the submit directory.
[ -f "$HERE/boltz2.def" ] || HERE="${SLURM_SUBMIT_DIR:-$PWD}/boltz2"
[ -f "$HERE/boltz2.def" ] || { echo "FATAL: cannot find boltz2.def; submit from the hpc-tools checkout" >&2; exit 2; }
. "$HERE/site.conf"

BOLTZ_VERSION=2.2.1
RELEASE="${RELEASE:-boltz-${BOLTZ_VERSION}_$(date +%Y%m%d)}"
DEST="$BOLTZ2_HOME/$RELEASE"
SRC_CACHE="${SRC_CACHE:-}"

# Weight files and their checksums. URLs: boltz 2.2.1 main.py (MOL_URL and the Hugging Face entry
# of BOLTZ2_URL_WITH_FALLBACK / BOLTZ2_AFFINITY_URL_WITH_FALLBACK). md5 sums: boltz2_conf.ckpt as
# pinned in the reference environment's conda spec; boltz2_aff.ckpt and mols.tar measured on
# 2026-11-16 in the cache the reference folds used.
HF=https://huggingface.co/boltz-community/boltz-2/resolve/main
declare -A MD5=(
    [boltz2_conf.ckpt]=2f0a1775bf8fc366a1a85e2019eca288
    [boltz2_aff.ckpt]=8e93dadedd6edb7a4d170f6051b99ec0
    [mols.tar]=5399454dcbbba695ab01f88deede577c
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 2; }

[ -e "$DEST" ] && die "$DEST already exists; releases are never replaced (set RELEASE=...)"
command -v apptainer >/dev/null || die "apptainer not found"
module load eth_proxy >/dev/null 2>&1 || die "module load eth_proxy failed (the build needs network)"

WORK="${TMPDIR:?no TMPDIR; request local disk with --tmp}/boltz2_build"
CTX="$WORK/ctx"
mkdir -p "$CTX/cache" "$WORK/apptainer_tmp" "$WORK/apptainer_cache"
export APPTAINER_TMPDIR="$WORK/apptainer_tmp" APPTAINER_CACHEDIR="$WORK/apptainer_cache"

log "=== staging the build context in $CTX ==="
cp "$HERE/boltz2.def" "$HERE/requirements.lock.txt" "$CTX/"
for f in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
    if [ -n "$SRC_CACHE" ]; then
        log "copying $f from $SRC_CACHE"
        cp "$SRC_CACHE/$f" "$CTX/cache/$f"
    else
        log "downloading $f from $HF"
        curl -fsSL --retry 3 -o "$CTX/cache/$f" "$HF/$f"
    fi
    got=$(md5sum "$CTX/cache/$f" | cut -d' ' -f1)
    [ "$got" = "${MD5[$f]}" ] || die "$f md5 $got, expected ${MD5[$f]}"
    log "  $f md5 ok"
done

log "=== apptainer build ($(apptainer --version)) ==="
# %files sources are resolved against the working directory, not the definition file's folder
cd "$CTX"
if ! apptainer build --fakeroot --build-arg "PIP_PROXY=${https_proxy:-}" \
        "$WORK/boltz2.sif" "$CTX/boltz2.def" 2>&1 | tee "$WORK/build.log"; then
    die "apptainer build failed (its output is in this job's .out file)"
fi

# The base image's registry digest: the tag in boltz2.def can be re-pointed upstream, the digest
# cannot. Recorded, not enforced; failure to look it up does not fail the build.
base_tag=$(sed -n 's/^ *PYTHON_TAG=//p' "$CTX/boltz2.def" | head -1)
base_digest=$(
    tok=$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull" \
          | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])') &&
    curl -fsSI -H "Authorization: Bearer $tok" \
         -H 'Accept: application/vnd.oci.image.index.v1+json' \
         -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
         "https://registry-1.docker.io/v2/library/python/manifests/$base_tag" \
      | tr -d '\r' | sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: *//p'
) || base_digest="(lookup failed)"

log "=== installing to $DEST ==="
mkdir -p "$DEST"
cp "$WORK/boltz2.sif" "$DEST/boltz2.sif.part"
mv "$DEST/boltz2.sif.part" "$DEST/boltz2.sif"
sha=$(sha256sum "$DEST/boltz2.sif" | cut -d' ' -f1)
commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
[ -n "$(git -C "$HERE" status --porcelain 2>/dev/null)" ] && commit="$commit (dirty)"
versions=$(apptainer exec --cleanenv "$DEST/boltz2.sif" python -c \
    'import importlib.metadata as m;print(" ".join(f"{p}=={m.version(p)}" for p in ("boltz","torch","numpy","rdkit","pytorch-lightning")))')

cat > "$DEST/BUILD_INFO.txt" <<INFO
release          : $RELEASE
built            : $(date '+%Y-%m-%d %H:%M:%S') on $(hostname), SLURM job ${SLURM_JOB_ID:-none}, by $USER
apptainer        : $(apptainer --version)
hpc-tools commit : $commit
base image       : docker://python:$base_tag
base digest      : $base_digest
definition md5   : $(md5sum "$CTX/boltz2.def" | cut -d' ' -f1)
lock md5         : $(md5sum "$CTX/requirements.lock.txt" | cut -d' ' -f1)
key packages     : $versions
boltz2_conf.ckpt : ${MD5[boltz2_conf.ckpt]}
boltz2_aff.ckpt  : ${MD5[boltz2_aff.ckpt]}
mols.tar         : ${MD5[mols.tar]} (extracted into the image; the file there is an empty placeholder)
weights from     : ${SRC_CACHE:-$HF}
image_sha256     : $sha
image size       : $(du -h "$DEST/boltz2.sif" | cut -f1)
INFO
cp "$CTX/requirements.lock.txt" "$DEST/requirements.lock.txt"
chmod 0644 "$DEST"/*
chmod 2755 "$DEST"
log "=== done ==="
cat "$DEST/BUILD_INFO.txt"
echo
echo "Not yet the default. After checking it (README, 'Releases'):"
echo "    ln -sfn $RELEASE $BOLTZ2_HOME/current"
