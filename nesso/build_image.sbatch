#!/bin/bash -l
#SBATCH --job-name=nesso_build
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=30G
#SBATCH --output=/cluster/scratch/%u/hpc-tools/nesso_build/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/hpc-tools/nesso_build/%x-%j.err
# =============================================================================
# Build the Nesso-1 Apptainer image on a compute node and install it as a new release.
#
#   mkdir -p /cluster/scratch/$USER/hpc-tools/nesso_build
#   sbatch nesso/build_image.sbatch                      # weights from the default cache below
#   SRC_CACHE=<a staged nesso cache> sbatch --export=ALL nesso/build_image.sbatch
#
# Writes $NESSO_HOME/<release>/nesso.sif and BUILD_INFO.txt. An existing release is never
# replaced, and `current` is not moved: a new image becomes the default only after it has been
# checked against the venv install (README, "Releases").
#
# The weights are not downloaded here. They are taken from a cache that setup_nesso_euler.sh
# already staged and whose revision is recorded, so an image can only ever hold the files the
# reference scores were produced with. The revision is checked against EXPECT_REVISION below.
#
# Sized from the first build here (job 21639381, 2026-11-20): 8 min 08 s wall, 25.57 GB peak RSS
# of 32 GB, 51% CPU efficiency over 8 cores. The memory peak is mksquashfs compressing the image
# tree and is already at 80% of the request, so that stays as it is; the walltime is five times
# what the build took, which covers a slow day on the proxy. --tmp holds, at once, the unpacked
# image tree (measured at 10 GB), the finished 5.5 GB .sif and the 3 GB cache.tar: 19 GB, so 30 G
# leaves room for pip's transient unpacking without reserving local disk nothing will use.
# =============================================================================
set -euo pipefail

HERE="${HPC_TOOLS_NESSO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# sbatch runs a copy of this script from a spool directory, so BASH_SOURCE does not point into
# the checkout; fall back to the submit directory.
[ -f "$HERE/nesso.def" ] || HERE="${SLURM_SUBMIT_DIR:-$PWD}/nesso"
[ -f "$HERE/nesso.def" ] || { echo "FATAL: cannot find nesso.def; submit from the hpc-tools checkout" >&2; exit 2; }
. "$HERE/site.conf"

NESSO_VERSION=1.0.0
EXPECT_REVISION="${EXPECT_REVISION:-v1.0.0}"
RELEASE="${RELEASE:-nesso-${NESSO_VERSION}_$(date +%Y%m%d)}"
DEST="$NESSO_HOME/$RELEASE"
SRC_CACHE="${SRC_CACHE:-/cluster/work/beltrao/$USER/nesso/cache}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 2; }

[ -e "$DEST" ] && die "$DEST already exists; releases are never replaced (set RELEASE=...)"
command -v apptainer >/dev/null || die "apptainer not found"
module load eth_proxy >/dev/null 2>&1 || die "module load eth_proxy failed (pip needs network)"

[ -d "$SRC_CACHE/huggingface" ] || die "no Hugging Face cache under $SRC_CACHE; run setup_nesso_euler.sh first, or set SRC_CACHE"
[ -f "$SRC_CACHE/REVISION" ] || die "no $SRC_CACHE/REVISION; the weights revision must be recorded before it can be pinned"
REVISION="$(tr -d '[:space:]' < "$SRC_CACHE/REVISION")"
[ "$REVISION" = "$EXPECT_REVISION" ] || die "cache holds revision '$REVISION', expected '$EXPECT_REVISION' (set EXPECT_REVISION to build from another one)"
log "weights revision: $REVISION"

WORK="${TMPDIR:?no TMPDIR; request local disk with --tmp}/nesso_build"
CTX="$WORK/ctx"
mkdir -p "$CTX" "$WORK/apptainer_tmp" "$WORK/apptainer_cache"
export APPTAINER_TMPDIR="$WORK/apptainer_tmp" APPTAINER_CACHEDIR="$WORK/apptainer_cache"

log "=== staging the build context in $CTX ==="
cp "$HERE/nesso.def" "$HERE/requirements.lock.txt" "$CTX/"
# Tarred rather than copied: the Hugging Face cache stores each file once under blobs/ and points
# at it from snapshots/ with a relative symlink, which a recursive copy can duplicate or break.
# .locks only exist to stop concurrent downloaders racing and nothing downloads inside the image.
tar -C "$SRC_CACHE" --exclude=huggingface/.locks -cf "$CTX/cache.tar" REVISION huggingface
log "  cache.tar: $(du -h "$CTX/cache.tar" | cut -f1), $(tar -tf "$CTX/cache.tar" | wc -l) entries"

# Every blob, by its own checksum. These are what the image will hold, and what BUILD_INFO.txt
# records, so a later image can be shown to carry the same weights as this one.
BLOBS="$(cd "$SRC_CACHE/huggingface" && find . -path './blobs/*' -type f -exec sha256sum {} + | sort -k2)"
log "  $(echo "$BLOBS" | wc -l) weight blob(s) checksummed"

log "=== apptainer build ($(apptainer --version)) ==="
# %files sources are resolved against the working directory, not the definition file's folder
cd "$CTX"
if ! apptainer build --fakeroot --build-arg "PIP_PROXY=${https_proxy:-}" \
        "$WORK/nesso.sif" "$CTX/nesso.def" 2>&1 | tee "$WORK/build.log"; then
    die "apptainer build failed (its output is in this job's .out file)"
fi

# The base image's registry digest: the tag in nesso.def can be re-pointed upstream, the digest
# cannot. Recorded, not enforced; failure to look it up does not fail the build.
base_tag=$(sed -n 's/^ *PYTHON_TAG=//p' "$CTX/nesso.def" | head -1)
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
cp "$WORK/nesso.sif" "$DEST/nesso.sif.part"
mv "$DEST/nesso.sif.part" "$DEST/nesso.sif"
sha=$(sha256sum "$DEST/nesso.sif" | cut -d' ' -f1)
commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
[ -n "$(git -C "$HERE" status --porcelain 2>/dev/null)" ] && commit="$commit (dirty)"
versions=$(apptainer exec --cleanenv "$DEST/nesso.sif" python -c \
    'import importlib.metadata as m;print(" ".join(f"{p}=={m.version(p)}" for p in ("nesso","torch","numpy","rdkit","lightning","transformers")))')

cat > "$DEST/BUILD_INFO.txt" <<INFO
release          : $RELEASE
built            : $(date '+%Y-%m-%d %H:%M:%S') on $(hostname), SLURM job ${SLURM_JOB_ID:-none}, by $USER
apptainer        : $(apptainer --version)
hpc-tools commit : $commit
base image       : docker://python:$base_tag
base digest      : $base_digest
definition md5   : $(md5sum "$CTX/nesso.def" | cut -d' ' -f1)
lock md5         : $(md5sum "$CTX/requirements.lock.txt" | cut -d' ' -f1)
key packages     : $versions
weights_revision : $REVISION
weights from     : $SRC_CACHE
image_sha256     : $sha
image size       : $(du -h "$DEST/nesso.sif" | cut -f1)

weight blobs (sha256, as cached by Hugging Face):
$BLOBS
INFO
cp "$CTX/requirements.lock.txt" "$DEST/requirements.lock.txt"
chmod 0644 "$DEST"/*
chmod 2755 "$DEST"
log "=== done ==="
cat "$DEST/BUILD_INFO.txt"
echo
echo "Not yet the default. After checking it (README, 'Releases'):"
echo "    ln -sfn $RELEASE $NESSO_HOME/current"
