# hpc-tools

Cluster tools shared by several projects, run on ETH Zurich's Euler.

| tool | for |
|---|---|
| [`boltz2/`](boltz2/README.md) | predicting structures with Boltz-2 — proteins, complexes, nucleic acids, bound small molecules and binding affinity — from one shared image |
| [`nesso/`](nesso/README.md) | scoring protein–ligand pairs with Nesso-1 from one shared image |
| [`archive_run_logs.sh`](#archive_run_logssh) | keeping per-job SLURM logs off a shared file-count quota, and archiving them when a run succeeds |

Both `boltz2/` and `nesso/` have a cell-by-cell example notebook under `examples/`, which runs one
complete job on public example inputs. Start there.

## `archive_run_logs.sh`

- **During a run:** per-job logs and status files go to `/cluster/scratch/$USER/...`.
- **When a run succeeds:** the logs become one zip in project storage, beside the results, with a
  manifest listing its members. The originals are deleted only after the zip's member count
  matches the files on disk and `unzip -t` passes, and then only the files the manifest lists.
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

- **Four directories only:** every path is resolved and refused if it lies outside the account's
  four Euler directories (`/cluster/home/$USER`, `/cluster/scratch/$USER`,
  `/cluster/project/beltrao/$USER`, `/cluster/work/beltrao/$USER`).
- **Scratch sources only:** `archive-dir` archives only from scratch, since it deletes the
  originals.
- **`--non-fatal`:** given anywhere on the command line, it makes every failure exit 0 after
  reporting it. That is for pipeline drivers under `set -e`. A refused or failed step still
  deletes nothing.

Read an archived log with `unzip -p <zip> <member>`. Find a job with `grep <jobid> <manifest>`.

The script's header has what it deliberately does not do, and its verification record.

### Calling it from a project

Point the project at this checkout, for example with a config variable, and call the script there;
do not copy it. On Euler the checkout is `/cluster/project/beltrao/lucla/repos/hpc-tools`, kept
fast-forward only like the project checkouts. A pipeline driver should treat a failed archive call
as a warning, not as a failed run.

## Licence

MIT, see [`LICENSE`](LICENSE). That covers the scripts, job templates, notebooks and
documentation in this repository, and nothing else.

The tools here are wrappers: they install and run third-party software under its own licence.
Nesso-1 is Apache-2.0 and Boltz-2 is MIT, both covering code and weights, and both permitting
redistribution. Each built image carries the upstream licence text inside it, under
`<package>.dist-info/licenses/` in site-packages, together with those of every dependency that
ships one — checked in both images on 2026-11-23. The repository itself redistributes neither:
it holds the recipes that fetch and build them.
