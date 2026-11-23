#!/bin/bash
#
# archive_run_logs.sh -- keep batch-infer's per-job logs off the group's
# file-count quota, and archive them into project storage when a run succeeds.
#
# VERIFICATION STATUS (2026-09-10)
#   `adopt` and `archive` have been exercised successfully: 265 leftover files
#   were relocated onto scratch, and a 397-file archive was built, verified by
#   member count and `unzip -t`, its members removed, and a record read back out
#   of the zip. The `setup` path's refusal on a non-empty directory was also
#   observed working.
#   NOT yet verified: the automatic invocation from submit_d0_pipeline.sh, which
#   first runs on the next wave submission.
#   2026-10-23: `archive-dir` archived MD4Kiki's 114 umbrella-stage SLURM logs:
#   member count and `unzip -t` verified, originals removed, a log read back out
#   of the zip.
#   2026-10-23, Euler, after adding the SAFEGUARDS: a 3-file test archive was
#   verified and cleared; a destination outside the four directories was refused
#   with nothing deleted or created (exit 1, and exit 0 with --non-fatal); a
#   source off scratch was refused.
#
# HOME
#   github.com/laflucETH/hpc-tools. Point a project at the checkout (on Euler:
#   /cluster/project/beltrao/lucla/repos/hpc-tools) rather than copying this file.
#
# WHY
#   Shared project storage is limited by its FILE-COUNT (inode) quota, not by
#   space: a share can sit within a percent of its hard file limit with terabytes
#   of space still free. The batch-infer profile writes one file per
#   SLURM job into .snakemake-eu/logs/<date>/<rule>/, and its status hook writes
#   one more into .snakemake-eu/sacct/ and .snakemake-eu/myjobs/ each. Across
#   ~55,500 jobs that reached ~187,800 filesystem entries. Meanwhile
#   /cluster/scratch/$USER held 63 files against a 1,000,000-entry quota.
#
# WHAT IT DOES
#   `setup` points .snakemake-eu/{logs,myjobs,sacct} at /cluster/scratch, so a
#   run's per-job files never land on project storage at all. `archive` collapses
#   them into one verified zip under .snakemake-eu/archive/ and clears scratch.
#
# SAFEGUARDS (added 2026-10-23, from BADASP's independently written version)
#   Every path it reads, writes or deletes is resolved (symlinks followed) and
#   refused if it lies outside the four directories this account owns on Euler.
#   That scope is otherwise only a convention; this makes it a mechanism.
#   --non-fatal (anywhere on the command line) makes any failure exit 0 after
#   reporting it, for a pipeline driver under `set -e`: housekeeping must never
#   fail a run. A refused or failed step still stops before anything is deleted.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   It does not modify software/batch-infer or its Snakemake profile: that code
#   is maintained by someone else. Redirection works purely through symlinks,
#   which the profile's relative `mkdir -p` and `install -D` calls follow with no
#   change on their side.
#
# SCRATCH IS PURGED PERIODICALLY. `setup` is idempotent and re-creates its
# targets, so a purge cannot break a future run -- but it can lose the logs of a
# past run that was never archived. The pipeline driver archives on success; a
# FAILED run's logs are left on scratch on purpose, so they stay loose and
# greppable while you debug. Archive those by hand when you are done with them.
#
# USAGE
#   archive_run_logs.sh [--non-fatal] <mode> ...
#   archive_run_logs.sh setup   <repo_dir>   # strict: refuse if logs already exist
#   archive_run_logs.sh adopt   <repo_dir>   # relocate any existing logs, then link
#   archive_run_logs.sh status  <repo_dir>
#   archive_run_logs.sh archive <repo_dir> [label]
#   archive_run_logs.sh archive-dir <scratch_dir> <dest_dir> [label]
#                                   # any plain SLURM array's scratch log dir
#
#   repo_dir is the project whose logs are handled. It is required: this script
#   lives in hpc-tools, not in that project.
#   Override the scratch root with BATCH_INFER_SCRATCH.

set -uo pipefail

SUBDIRS=(logs myjobs sacct)
SCRATCH_ROOT="${BATCH_INFER_SCRATCH:-/cluster/scratch/$USER/batch-infer-logs}"
# The only directories this account owns on Euler; see SAFEGUARDS.
ALLOWED_PREFIXES=(
    "/cluster/home/$USER"
    "/cluster/scratch/$USER"
    "/cluster/project/beltrao/$USER"
    "/cluster/work/beltrao/$USER"
)
NON_FATAL=0

die() {
    echo "ERROR: $*" >&2
    if [[ "$NON_FATAL" -eq 1 ]]; then echo "[archive_run_logs] --non-fatal: exiting 0" >&2; exit 0; fi
    exit 1
}
info() { echo "[archive_run_logs] $*"; }

# Count regular files without depending on line splitting: batch-infer log names
# contain ':' and other awkward characters.
count_files() { find "$1" -type f -printf '.' 2>/dev/null | wc -c; }

# These two run inside $( ): they report and return 1 rather than die, because
# exiting there only ends the subshell. Their callers stop on the failure.
in_scope() {
    local p="$1" what="$2" resolved prefix
    resolved=$(readlink -f "$p" 2>/dev/null) || resolved="$p"
    for prefix in "${ALLOWED_PREFIXES[@]}"; do
        case "$resolved" in "$prefix"|"$prefix"/*) printf '%s\n' "$resolved"; return 0 ;; esac
    done
    echo "ERROR: $what resolves to '$resolved', outside this account's directories (${ALLOWED_PREFIXES[*]})" >&2
    return 1
}

resolve_repo() {
    local d="${1:-}"
    [[ -n "$d" ]] || { echo "ERROR: give the project's repo dir: this script lives in hpc-tools, not in the project" >&2; return 1; }
    [[ -d "$d" ]] || { echo "ERROR: repo dir does not exist: $d" >&2; return 1; }
    in_scope "$(cd "$d" && pwd)" "repo dir"
}

scratch_base_for() { echo "$SCRATCH_ROOT/$(basename "$1")"; }

do_setup() {
    local repo="$1"
    local base; base=$(in_scope "$(scratch_base_for "$repo")" "scratch base") || die "refusing to setup"
    local smk="$repo/.snakemake-eu"

    mkdir -p "$smk" || die "cannot create $smk"

    local s link target cur n
    for s in "${SUBDIRS[@]}"; do
        target="$base/$s"
        link="$smk/$s"
        mkdir -p "$target" || die "cannot create scratch target $target"

        if [[ -L "$link" ]]; then
            cur=$(readlink -f "$link" 2>/dev/null)
            if [[ "$cur" == "$(readlink -f "$target")" ]]; then
                info "$s already points at scratch"
            else
                die "$link points at '$cur', not '$target'. Refusing to repoint it."
            fi
        elif [[ -d "$link" ]]; then
            n=$(count_files "$link")
            if [[ "$n" -gt 0 ]]; then
                die "$link is a real directory holding $n files. Archive it first, then re-run setup. Refusing to touch existing logs."
            fi
            rmdir "$link" || die "could not remove empty dir $link"
            ln -s "$target" "$link" || die "could not create symlink $link"
            info "$s linked to $target (replaced empty dir)"
        elif [[ -e "$link" ]]; then
            die "$link exists and is neither directory nor symlink. Refusing to touch it."
        else
            ln -s "$target" "$link" || die "could not create symlink $link"
            info "$s linked to $target"
        fi
    done
    info "per-job logs for this repo will be written to $base"
}

do_adopt() {
    local repo="$1"
    local base; base=$(in_scope "$(scratch_base_for "$repo")" "scratch base") || die "refusing to adopt"
    local smk="$repo/.snakemake-eu"

    mkdir -p "$smk" || die "cannot create $smk"

    local s link target cur n before after aside
    for s in "${SUBDIRS[@]}"; do
        target="$base/$s"
        link="$smk/$s"
        mkdir -p "$target" || die "cannot create scratch target $target"

        if [[ -L "$link" ]]; then
            cur=$(readlink -f "$link" 2>/dev/null)
            [[ "$cur" == "$(readlink -f "$target")" ]] \
                || die "$link points at '$cur', not '$target'. Refusing to repoint it."
            info "$s already points at scratch"
            continue
        fi

        if [[ -e "$link" && ! -d "$link" ]]; then
            die "$link exists and is neither directory nor symlink. Refusing to touch it."
        fi

        if [[ ! -e "$link" ]]; then
            ln -s "$target" "$link" || die "could not create symlink $link"
            info "$s linked to $target"
            continue
        fi

        n=$(count_files "$link")
        if [[ "$n" -eq 0 ]]; then
            rmdir "$link" || die "could not remove empty dir $link"
            ln -s "$target" "$link" || die "could not create symlink $link"
            info "$s linked to $target (replaced empty dir)"
            continue
        fi

        # Relocate rather than refuse, so a leftover real directory cannot
        # silently keep the redirect switched off run after run.
        #
        # Order matters. The directory is moved aside and the symlink put in place
        # FIRST, so anything that begins writing mid-migration lands on scratch
        # instead of into a directory about to be removed. Only then are the
        # contents copied across, the count verified, and the set-aside copy
        # deleted. Deleting before verifying is what cost 132 sacct records on
        # 2026-09-10.
        aside="${link}.migrating.$$"
        mv "$link" "$aside" || die "could not move $link aside"
        if ! ln -s "$target" "$link"; then
            mv "$aside" "$link"
            die "could not create symlink $link -- original directory restored"
        fi

        info "relocating $n existing $s file(s) onto scratch"
        before=$(count_files "$target")
        if ! ( cd "$aside" && tar cf - . ) | ( cd "$target" && tar xf - ); then
            die "could not copy $s onto scratch -- originals KEPT at $aside"
        fi
        after=$(count_files "$target")
        # `-lt`, not `-ne`: a live run may legitimately add files to the target
        # while the copy runs, so require at least n moved rather than exactly n.
        if [[ $((after - before)) -lt "$n" ]]; then
            die "copied $((after - before)) of $n $s file(s) -- originals KEPT at $aside"
        fi
        rm -rf "$aside" || info "warning: could not remove $aside (harmless, but it still costs inodes)"
        info "relocated $n $s file(s); they will be swept into the next archive"
    done
    info "per-job logs for this repo will be written to $base"
}

do_status() {
    local repo="$1"
    local base; base=$(scratch_base_for "$repo")
    local smk="$repo/.snakemake-eu"
    local s link

    echo "repo:         $repo"
    echo "scratch base: $base"
    echo
    for s in "${SUBDIRS[@]}"; do
        link="$smk/$s"
        if [[ -L "$link" ]]; then
            printf "  %-8s -> %s\n" "$s" "$(readlink "$link")"
        elif [[ -d "$link" ]]; then
            printf "  %-8s REAL DIR on project storage (%s files) -- not redirected\n" "$s" "$(count_files "$link")"
        else
            printf "  %-8s absent\n" "$s"
        fi
    done
    echo
    echo "unarchived files on scratch:"
    for s in "${SUBDIRS[@]}"; do
        printf "  %-8s %s\n" "$s" "$(count_files "$base/$s")"
    done
    echo
    echo "archives in $smk/archive:"
    ls -lh "$smk/archive" 2>/dev/null | tail -n +2 || echo "  (none)"
    echo
    echo "NOTE: scratch is purged periodically; unarchived logs above are not durable."
}

# ---------------------------------------------------------------------------
# The verified core, shared by `archive` and `archive-dir` so the two can never
# drift apart: build a zip, prove it matches what was on disk, write a manifest,
# and delete ONLY what the manifest lists.
#   $1 base dir to archive from   $2 destination dir   $3 label
#   $4.. the paths under $base to include (use "." for everything)
# ---------------------------------------------------------------------------
zip_verify_and_clear() {
    local base="$1" adir="$2" label="$3"; shift 3
    local -a parts=("$@")
    base=$(in_scope "$base" "archive source") || die "refusing to archive"
    adir=$(in_scope "$adir" "archive destination") || die "refusing to archive"

    local n_disk=0 s n
    for s in "${parts[@]}"; do
        n=$(count_files "$base/$s")
        n_disk=$((n_disk + n))
    done
    if [[ "$n_disk" -eq 0 ]]; then
        info "nothing on scratch to archive under $base"
        return 0
    fi

    mkdir -p "$adir" || die "cannot create $adir"

    local ts; ts=$(date +%Y%m%dT%H%M%S)
    local zipf="$adir/${label}_${ts}.zip"
    local manifest="$adir/manifest_${label}_${ts}.tsv"

    info "archiving $n_disk files from $base"
    # Level 6 (zip's default): the work is NFS-latency bound, so a better ratio
    # is effectively free. Built as .part and renamed, so an interrupted run
    # never leaves a truncated archive that looks finished.
    # zip APPENDS to an existing archive. A .part left behind by a killed run
    # would be appended to here, yielding duplicate members and a count
    # mismatch, so clear it before building.
    rm -f "${zipf}.part"
    if ! ( cd "$base" && zip -q -r -6 "${zipf}.part" "${parts[@]}" ); then
        rm -f "${zipf}.part"
        die "zip failed -- scratch files left untouched"
    fi
    mv "${zipf}.part" "$zipf" || die "could not finalise $zipf"

    local n_zip
    n_zip=$(unzip -Z1 "$zipf" | grep -v '/$' | wc -l)
    if [[ "$n_zip" -ne "$n_disk" ]]; then
        die "member count $n_zip != file count $n_disk -- scratch files KEPT, archive left at $zipf for inspection"
    fi
    if ! unzip -t "$zipf" > /dev/null 2>&1; then
        die "integrity test failed -- scratch files KEPT, archive left at $zipf for inspection"
    fi

    unzip -Z1 "$zipf" | grep -v '/$' > "$manifest"
    info "verified $n_zip files -> $zipf ($(du -sh "$zipf" | cut -f1))"

    # Delete ONLY the files this archive actually contains, read back from the
    # manifest. Clearing the whole tree instead would also destroy anything
    # written while the zip was being built -- a real data-loss window, because
    # the pre-zip file count cannot see those later files, yet a tree-wide delete
    # removes them. That mistake cost 132 sacct records on 2026-09-10.
    local removed=0 member
    while IFS= read -r member; do
        [[ -n "$member" ]] || continue
        if rm -f "$base/$member" 2>/dev/null; then removed=$((removed + 1)); fi
    done < "$manifest"

    # Prune directories left empty, but never the roots themselves: a symlink
    # pointing at one must stay valid.
    for s in "${parts[@]}"; do
        find "$base/$s" -mindepth 1 -type d -empty -delete 2>/dev/null
    done

    local left=0
    for s in "${parts[@]}"; do left=$((left + $(count_files "$base/$s"))); done
    info "removed $removed archived file(s)"
    if [[ "$left" -gt 0 ]]; then
        info "$left file(s) appeared during archiving and were LEFT IN PLACE for the next archive"
    fi
    info "read a log with:      unzip -p '$zipf' <member>"
    info "find a job id with:   grep <jobid> '$manifest'"
}


# ---------------------------------------------------------------------------
# archive-dir: the same discipline applied to ANY scratch directory, for the
# plain SLURM arrays this project also runs (the Boltz-2 jobs write
# /cluster/scratch/$USER/<run>/logs/ directly and have no .snakemake-eu tree).
# Added rather than forking the script, because two divergent copies of this
# logic is precisely the failure mode the convention warns about.
# ---------------------------------------------------------------------------
do_archive_dir() {
    local src="$1" dest="$2" label="${3:-run}"
    [[ -n "$src" && -n "$dest" ]] || die "usage: archive-dir <scratch_dir> <dest_dir> [label]"
    [[ -d "$src" ]] || die "not a directory: $src"
    case "$src" in
        /cluster/scratch/*) : ;;
        *) die "refusing to archive-dir from '$src': the source must be under /cluster/scratch, since this DELETES the originals after verifying" ;;
    esac
    zip_verify_and_clear "$src" "$dest" "$label" "."
}


do_archive() {
    local repo="$1"
    local base; base=$(scratch_base_for "$repo")
    local adir="$repo/.snakemake-eu/archive"
    local label="${2:-run}"
    zip_verify_and_clear "$base" "$adir" "$label" "${SUBDIRS[@]}"
}

args=()
for a in "$@"; do
    if [[ "$a" == "--non-fatal" ]]; then NON_FATAL=1; else args+=("$a"); fi
done
set -- ${args[@]+"${args[@]}"}

cmd="${1:-}"
case "$cmd" in
    setup)   shift; repo=$(resolve_repo "${1:-}") || die "no usable repo dir"; do_setup   "$repo" ;;
    adopt)   shift; repo=$(resolve_repo "${1:-}") || die "no usable repo dir"; do_adopt   "$repo" ;;
    status)  shift; repo=$(resolve_repo "${1:-}") || die "no usable repo dir"; do_status  "$repo" ;;
    archive) shift; repo=$(resolve_repo "${1:-}") || die "no usable repo dir"; do_archive "$repo" "${2:-run}" ;;
    archive-dir) shift; do_archive_dir "${1:-}" "${2:-}" "${3:-run}" ;;
    ""|-h|--help|help) sed -n '/^# USAGE/,/^$/p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown command '$cmd' (try: setup | adopt | status | archive | archive-dir)" ;;
esac
