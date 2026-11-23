# hpc-tools

Small cluster tools shared by several projects. Each project calls the copy here rather than keeping its own, so
there is one version of each tool to fix and verify.

| tool | for |
|---|---|
| [`archive_run_logs.sh`](#archive_run_logssh) | keeping per-job SLURM logs off a shared file-count quota, and archiving them when a run succeeds |
| [`boltz2/`](boltz2/README.md) | predicting structures with Boltz-2 on Euler — proteins, complexes, nucleic acids, bound small molecules and binding affinity — from one shared image |
| [`nesso/`](nesso/README.md) | scoring protein-ligand pairs with Nesso-1 on Euler, at a scale that would otherwise exhaust a file-count quota — from one shared image |

Both `boltz2/` and `nesso/` have a cell-by-cell example notebook under `examples/`, which runs one complete job
on public example inputs. That is the place to start with either of them.

They exist for the same reason: shared project storage runs out of file-count (inode) quota long before it runs
out of space, and the things that exhaust it are tools that write one small file per unit of work.

Both answer it a second way as well: each tool is **one Apptainer image the whole group reads**, rather than an
environment installed per person. A `pip install` of Nesso-1 is 41,358 files; its image is one, so a colleague
who wants to try it needs no install, no clone and no quota of their own. Being read-only, it also means nobody's
analysis moves under them because someone else upgraded a package.

## `archive_run_logs.sh`

Keeps per-job SLURM logs off the file-count (inode) quota of shared project storage, and archives them when a run
succeeds.
- **During a run:** per-job logs and status files go to `/cluster/scratch/$USER/...`.
- **When a run succeeds:** the logs become one zip in project storage, beside the results, with a manifest listing
  its members. The originals are deleted only after the zip's member count matches the files on disk and `unzip -t`
  passes, and then only the files the manifest lists.
- **When a run fails:** its logs stay loose on scratch, so they can be searched while debugging.

Modes:

| mode | for |
|---|---|
| `setup <repo_dir>` | point a batch-infer repo's `.snakemake-eu/{logs,myjobs,sacct}` at scratch (refuses if logs already exist) |
| `adopt <repo_dir>` | the same, moving any existing logs onto scratch first |
| `status <repo_dir>` | where a repo's logs go, and what is still unarchived |
| `archive <repo_dir> [label]` | zip, verify and clear a batch-infer repo's scratch logs |
| `archive-dir <scratch_dir> <dest_dir> [label]` | the same for any plain SLURM log directory on scratch |

Safeguards:
- **Four directories only:** every path is resolved and refused if it lies outside the account's four Euler
  directories (`/cluster/home/$USER`, `/cluster/scratch/$USER`, `/cluster/project/beltrao/$USER`,
  `/cluster/work/beltrao/$USER`).
- **Scratch sources only:** `archive-dir` archives only from scratch, since it deletes the originals.
- **`--non-fatal`:** given anywhere on the command line, it makes every failure exit 0 after reporting it. That
  is for pipeline drivers under `set -e`. A refused or failed step still deletes nothing.

Read an archived log with `unzip -p <zip> <member>`. Find a job with `grep <jobid> <manifest>`.

The script's header has the full rationale, what it deliberately does not do, and its verification record.

## Using it from a project

Point the project at this checkout, for example with a config variable, and call the script there; do not copy it.
On Euler the checkout is `/cluster/project/beltrao/lucla/repos/hpc-tools`, kept fast-forward only like the project
checkouts. A pipeline driver should treat a failed archive call as a warning, not as a failed run.

## Licence

MIT, see [`LICENSE`](LICENSE) — copyright (c) 2026 ETH Zurich. That covers the scripts, job
templates, notebooks and documentation in this repository, and nothing else.

The tools here are wrappers: they install and run third-party software under its own licence.
Nesso-1 is Apache-2.0 and Boltz-2 is MIT, both covering code and weights, and both permitting
redistribution. Each built image carries the upstream licence text inside it, under
`<package>.dist-info/licenses/` in site-packages, together with those of every dependency that
ships one — checked in both images on 2026-11-23. The repository itself redistributes neither:
it holds the recipes that fetch and build them.
