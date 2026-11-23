# Nesso-1 on Euler

[Nesso-1](https://github.com/recursionpharma/nesso) is a coarse-grained cofolding model that
scores a protein against a small molecule: given a protein sequence and a ligand SMILES it
returns a predicted binding affinity, a binder/non-binder probability, and an entropy term that
says how confidently it placed the ligand. It needs no MSA and no structure, and the authors
report about one second per pair on one GPU — roughly ten to twenty times faster than Boltz-2,
which is what makes screens of a whole protein family against a whole compound library worth
considering.

It is an affinity model, not a structure model. It does not produce a usable pose, and its own
preprint is explicit that coarse-graining costs it accuracy on structural tasks.

## What this wrapper is for

Running it once is a one-liner; running it over a few hundred thousand pairs on shared storage is
not. Nesso-1 writes about **five filesystem entries per pair** — an input YAML, a structure
`.npz`, a record `.json`, and a `predictions/<id>/affinity.json`. An all-against-all screen of
roughly 300 proteins against roughly 1,000 ligands would be some 1.5 M entries, which is more
than a personal scratch file-count quota allows and far more than shared project storage can
absorb: such a share is quota-limited in files rather than bytes, and can sit within a few
percent of its file limit with terabytes of space still free.

So this wrapper keeps every per-pair file on scratch and **collapses each shard into one table
and one zip as soon as that shard succeeds**, deleting the shard's tree. Peak file count scales
with how many shards run at once, not with how big the run is. Project storage only ever receives
one scores table, one archive of the raw model output, a manifest and a provenance record.

## The contract

Input is a CSV. Three columns are required; every other column is carried through to the output
untouched, so a project's own identifiers travel with the scores.

| column | meaning |
|---|---|
| `pair_id` | unique, and usable as a filename: `[A-Za-z0-9][A-Za-z0-9._-]*` |
| `protein_sequence` | one-letter amino acids |
| `ligand_smiles` | the ligand whose affinity is predicted |
| `extra_ligands` | *optional*, `;`-separated: further ligands placed in the complex as their own chains. Each entry is a SMILES, or `ccd:<CODE>` for a Chemical Component Dictionary code — which is how a metal ion is given beside the macrocycle it sits in (`ccd:FE2`). Affinity is always predicted against `ligand_smiles`. |

Output is one row per input row, with these added:

| column | meaning |
|---|---|
| `nesso:affinity_pred_value` | log<sub>10</sub>(IC<sub>50</sub> / µM). **Lower is stronger** binding: −3 ≈ 1 nM, 0 ≈ 1 µM, +2 ≈ 100 µM |
| `nesso:affinity_pred_value1`, `…2` | the two ensemble members behind that mean |
| `nesso:affinity_probability_binary` | P(binder), 0–1. **Higher is more likely** a binder |
| `nesso:affinity_logits_binary` | the logit behind that probability |
| `nesso:entropy_crop_pl` | entropy of the predicted protein–ligand distance distribution at the interface. The model card calls this the confidence to read; **0.0 means the ligand could not be placed confidently** and that pair's prediction should not be trusted |
| `nesso:entropy_crop_pp`, `…_ll`, `nesso:entropy_pp`, `…_pl`, `…_ll` | the remaining entropy terms |
| `nesso:status` | `scored`, or why the pair has no score |
| `nesso:shard` | which shard scored it |

### Known limitation: the same molecule twice

Nesso-1 1.0.0 cannot score a complex that holds **two copies of one molecule** — ferric
dicitrate, say, which is two citrates around an iron. It writes one RDKit conformer per
*distinct* ligand entity, named after that entity's first chain id, but asks for a conformer per
ligand *chain* at inference, so the second copy raises `No pkl for ligand chain '<id>'`. The
documented `id: [B, C]` spelling for repeated copies groups the same way and does not avoid it.

The runner therefore refuses such a pair at plan time, with
`skipped: repeated_ligand_entity`, so it never reaches a GPU and the reason travels in the output
table. If you need those pairs scored, the workaround is to give the repeated molecules as one
entity with a multi-fragment SMILES (`A.A`) — chemically the same set of molecules, but a
different entity partition from the one another tool may have used, so say so when comparing.

**No row is ever dropped.** A pair that cannot be built, or that the model fails to score, keeps
its row with a `nesso:status` saying why. An output table shorter than its input has lost
something silently, which is the failure this rule exists to prevent.

## Install: there isn't one

Nesso-1 runs from a single Apptainer image that the whole group reads:

```
/cluster/work/beltrao/lucla/hpc-tools/nesso/current/nesso.sif
```

It holds the pinned package set, the model checkpoint, the chemical component dictionary and the
ESM-2 650M weights -- 5.5 GB in **one file**. `nesso/site.conf` names the directory; `nesso info`
says which image would run and records what went into it.

**Why an image and not a `pip install` each.** The install is 41,358 files and 7.1 GB, and
`/cluster/project` is quota-limited in *files*, not bytes. Three people installing their own
would spend a large slice of a share's remaining file headroom on three copies of the same
software -- in a repository whose other tool exists to stop exactly that. One image costs
three files, once. It is also read-only, so nobody's analysis can shift under them because a
colleague upgraded something.

```bash
nesso/nesso info          # which image, and the checksums of everything in it
nesso/nesso info --verify # ... and recompute the image's own sha256
```

### Building a new image

```bash
mkdir -p /cluster/scratch/$USER/hpc-tools/nesso_build
sbatch nesso/build_image.sbatch
```

The weights are not downloaded during the build. They are taken from a cache that
`setup_nesso_euler.sh` already staged and whose Hugging Face revision is recorded, and the build
refuses to proceed unless that revision is the expected one -- so an image can only ever hold the
weights the reference scores were produced with. Without that pin, a rebuild months later can
quietly pick up re-uploaded weights and produce numbers that no longer match the ones already
analysed.

A new image is installed as its own release and `current` is **not** moved. Point it there only
after checking it against the environment the reference scores ran in:

```bash
sbatch nesso/compare_to_venv.sbatch $RUN 0      # scores one shard venv / image / image again
ln -sfn nesso-1.0.0_YYYYMMDD /cluster/work/beltrao/lucla/hpc-tools/nesso/current
```

### Running a virtual environment instead

`bash nesso/setup_nesso_euler.sh` still builds one, and `NESSO_VENV=<path>` makes a job use it.
That is the route for developing against a modified Nesso-1, and it is what the image is checked
against. Set it explicitly: nothing falls back to a virtual environment on its own, because a
silent fallback would let two people score the same pairs against two different builds with
nothing in either log to say so.

## Starting from scratch, as a new user

Nothing to install. Both the tooling and the model are already readable by anyone in
`biol-imsb-beltrao`, so you can start from the group's checkout as it stands:

```bash
cd /cluster/project/beltrao/lucla/repos/hpc-tools
nesso/nesso info                                  # confirms you can read the image
mkdir -p /cluster/scratch/$USER/nesso/slurm       # where your job logs go
```

Only the tooling is shared. Your runs, your scratch directory, your results and your SLURM
logs are all your own -- the paths carry `$USER`, and nothing writes into anyone else's space.

Then open [`examples/01_first_run.ipynb`](examples/01_first_run.ipynb) in Euler's JupyterHub and
run it cell by cell. It takes **twenty to thirty minutes**, most of it waiting for one GPU job
-- 9 minutes of prediction on an RTX 3090, 17 on an older TITAN RTX -- and walks through one
complete run on 24 example pairs, explaining each step where it happens: the input contract, why
the per-pair files stay on scratch, how to size the request, and how to read the scores. Any
Python 3 kernel with `pandas` works; every call into Nesso-1 is a subprocess into the shared
image, so nothing has to be installed into the kernel.

The example inputs come from `example_pairs.py`, which builds a 24-pair table from public records
-- four drug targets fetched from UniProt by accession, six drugs fetched from PubChem by name.
Nothing is pasted in, so swapping those two lists for your own makes your own table. The set is a
control as well as a demonstration: each drug is the established inhibitor of one of the four
targets, marked in a `cognate` column, so a working score has to rank each target's own drug
first.

### Getting the code

The shared checkout is readable by the group but **not writable by you**, and an editor wants
somewhere it can save. Either of these works:

```bash
# A. your own clone -- 672 files including .git, 1.7 MB. Needs no GitHub account at all:
git clone /cluster/project/beltrao/lucla/repos/hpc-tools ~/repos/hpc-tools

# B. just the notebook, pointed at the shared tooling:
mkdir -p /cluster/scratch/$USER/nesso && cd /cluster/scratch/$USER/nesso
cp /cluster/project/beltrao/lucla/repos/hpc-tools/nesso/examples/01_first_run.ipynb .
export HPC_TOOLS_NESSO_DIR=/cluster/project/beltrao/lucla/repos/hpc-tools/nesso
```

`origin` then points at the shared checkout, so `git pull` picks up changes with no GitHub
account anywhere in the loop. `site.conf` travels with the clone, so `nesso/nesso` finds the
same shared image from your copy.

Cloning the repository is **not** what the shared image exists to avoid. A clone is 672 files;
the `pip install` the image replaces was 46,039. Clone it freely -- just do not install Nesso-1
again.

### Working from your own IDE (Euler tunnel)

JupyterHub is the quickest way in, but if you would rather use your own editor, ETH's
[`euler-tunnel`](https://docs.hpc.ethz.ch/connections/euler-tunnel/) submits a batch job running
an SSH server and tunnels into it, so VS Code's Remote-SSH reaches a **compute node** instead of
a login node -- which is the point, since computation is not permitted on login nodes.

Setup, from ETH's documentation (`euler-tunnel config` generates an SSH key on first run, so run
it yourself):

```bash
euler-tunnel config    # prints a ~/.ssh/known_hosts line and a Host block for ~/.ssh/config
```

Add both on your own machine -- on Windows, drop the `Control*` lines. Then per session, from a
login node:

```bash
euler-tunnel start --time=04:00:00 --cpus-per-task=2 --mem-per-cpu=4G
euler-tunnel status
```

and in VS Code: Remote-SSH, "Connect to Host...", `euler-tunnel`.

**Do not give the tunnel job a GPU.** It runs your editor and the notebook kernel and nothing
else; every prediction goes to its own `sbatch` job with its own GPU. A GPU held by an idle
editor is one nobody else can schedule.

Two things this pipeline needs from a compute node, both checked here (2026-11-23, job 22026196)
rather than assumed:

- **`module load eth_proxy` gives it a route out**, setting the proxy variables for you,
  so step 1 still fetches from UniProt and PubChem. Compute nodes have no direct internet.
- **`sbatch` is accepted from inside the tunnel job**, so step 3 can submit the prediction array
  without dropping back to a login node.

**The kernel.** Euler's module Python already carries `ipykernel` and `pandas` and runs with an
empty environment, so point the IDE straight at it -- there is nothing to install:

```bash
module load stack/2024-06 python/3.11.6 && which python   # paste that path into the IDE
```

Nothing else belongs in the kernel: every call into Nesso-1 is a subprocess into the image.

Prefer the command line? The same four steps are below.

## Running

```bash
RUN=/cluster/scratch/$USER/nesso/<label>

# 1. plan: pairs table -> per-pair YAMLs on scratch, grouped into shards
nesso/nesso python nesso/nesso_run.py plan \
    --pairs <pairs.csv> --run-dir $RUN --pairs-per-shard 2000

# 2. predict: one array task per shard (the plan step prints the array range)
sbatch --array=0-N nesso/submit_nesso.sh $RUN

# 3. finalize: shard tables -> one scores table in project storage
nesso/nesso python nesso/nesso_run.py finalize \
    --run-dir $RUN --dest <project dir> --label <label>

# 4. archive the SLURM logs beside the results
./archive_run_logs.sh archive-dir /cluster/scratch/$USER/nesso/slurm <project dir> <label> --non-fatal
```

`nesso/nesso python` is the image's own interpreter, so steps 1 and 3 need no Python environment
of yours. They run on a login node: planning writes small YAML files and finalizing reads tables,
neither of which is heavy enough for a compute node. Step 1 needs `module load eth_proxy` if the
pairs table is built by fetching sequences, since login nodes reach the internet only through the
proxy and compute nodes not at all.

Pairs are sharded **by protein**: the plan sorts by sequence so one protein's pairs stay together
and its ESM-2 embedding is computed once rather than once per shard it appears in.

`plan --override` replaces a plan **and deletes the previous run's `inputs/`, `work/` and
`shards/` with it.** It has to: a shard whose table already exists is skipped by the submit
script, so a stale table left in place would be finalized as the new run's result, invisibly so
when the pair ids match. It refuses to delete anything outside a scratch root.

A task whose shard table already exists exits 0 without recomputing, so a partly finished array
is resubmitted as-is. A shard that fails keeps every one of its files on scratch — that is when
they are needed — and cleanup happens only after the prediction exits cleanly, every pair is
scored, and the zip's member count and `unzip -t` both check out.

## Scores are not reproducible between runs

Measured, 2026-11-19: the same 24 pairs scored three times -- same pairs table, same seed (42),
same model revision, same cores and memory, cuEquivariance kernels off throughout -- gave
`affinity_pred_value` differing by up to **0.25 log<sub>10</sub> units** between two runs on the
*same* GPU model. Most pairs are stable (18 of 24 agreed within 0.03); the drift concentrates on
a few, and on these inputs mostly on the longest protein.

The cause is not the seed, the weights or the hardware, all of which were held constant.
Nesso-1 calls `pl.seed_everything(seed, workers=True)` and nothing else: it never asks for
deterministic CUDA kernels (`torch.use_deterministic_algorithms`), so operations implemented with
atomics are free to sum in a different order each run. Seeding makes the *random* choices
repeatable; it does not make floating-point reductions repeatable. Upstream's documentation
calls the seed a "random seed for reproducible predictions", which is true only of the former.

What follows from it:

- **Keep a comparison inside one run.** Two pairs scored in the same job are comparable; the same
  pair scored in two jobs is not, to better than about 0.25.
- **Never top up a finished set with a later job** and rank the combined table. Re-plan and
  re-run the whole thing, the same rule that already applies to changing the kernel setting.
- **Shard by protein and this mostly takes care of itself**, which is what `plan` already does.
  In a 4,929-pair production run, 245 of 252 proteins had all their pairs in one shard, hence
  one job, so a within-protein ranking was internal to a single run. The 7 that straddled a
  shard boundary are the exposed ones.
- **For scale**: `affinity_pred_value` is log10(IC50 / uM), so 0.25 is most of a factor of two
  in predicted IC50 -- the same order as the gap between neighbouring pairs in a within-protein
  ranking. It did not change the example's conclusion -- each target's own drug still ranked
  first in all three runs -- but it would move pairs that sit close together.

## Sizing a run

The `#SBATCH` lines in `submit_nesso.sh` are **starting points for a pilot, not measured values.**
Run one or two shards, then read the real usage and pass the measured numbers on the command
line:

```bash
sbatch --array=0 nesso/submit_nesso.sh $RUN      # pilot
seff <jobid>
sacct -j <jobid> --format=JobID,MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS,ReqTRES%40
sbatch --array=1-N --time=HH:MM:SS --mem-per-cpu=NNNN nesso/submit_nesso.sh $RUN
```

Over-requesting costs the shared allocation and other users' scheduling headroom even when it
does not change our own turnaround, so trim it whichever way the pilot points. Re-measure when
the workload changes shape — much longer proteins, many more ligands per protein, a different
shard size.

Each job prints its own seconds-per-pair, which is the number to size `--time` from.

**The image needs about 2 GB more host memory than a virtual environment does.** Measured on the
same 24 pairs: 10.82 and 10.84 GB over two venv runs, against 12.84 GB through the image -- 92%
of a 14 GB request, where the venv sat at 77%. The likely cause is the image's own library pages
held in the page cache and charged to the job's cgroup, which would make it a roughly fixed cost
rather than one that grows with the shard; that has **not** been measured on a larger shard, so
re-check it if you size a big run from a small pilot. Either way, size from a pilot run through
the engine you will actually use -- a number measured under the venv is now an underestimate.

### What one measured run looked like

A 600-pair shard on one RTX 4090, **without** the cuEquivariance kernels (2026-11-16):

| | |
|---|---|
| throughput | **2.47 s/pair** (25 min for the shard) |
| memory | **15.95 GB**, against the 16 GB requested — 99.7%, uncomfortably close to an OOM |
| CPU | 25.7% of 4 cores, so most of the cores were idle |
| failures | none; 600 of 600 scored |

Sized from that, later shards were given **2 cores and 24 GB**. At 2 cores their peak was only
11.5 GB, so the pilot's 15.95 GB was largely its four dataloader workers, and **2 cores with
14 GB is the right request** for a shard of this shape. Memory is the binding constraint here,
not cores and not GPU memory, so that is the number to watch when the workload changes shape.

**Runtime scales with protein length, steeply, and with the card you land on.** The figures
above are for proteins of a median 238 residues. The 24-pair drug-target example in
[`examples/`](examples/01_first_run.ipynb), whose targets run from 260 to 1,210
residues, took 15.8 s/pair on an RTX 4090 and 21.1-22.0 s/pair on an RTX 3090 -- though those
numbers also carry the fixed cost of loading the model and embedding the proteins, spread over
only 24 pairs. Its memory was 10.8 GB of the 14 GB requested. Size a run of larger proteins from
its own pilot, not from the figures here.

The 2.47 s/pair is about 2.5x the ~1 s/pair the preprint reports, which was measured on an H100
with the kernels enabled. Turning the kernels on (`NESSO_KERNELS=1`) is the obvious lever if
throughput matters, but it changes the numerics, so never mix kernel and no-kernel scores inside
one run -- re-run the whole thing rather than part of it.

## Tests

```bash
python3 -m pytest nesso/test_nesso_run.py
```

The prediction step is the only part needing a GPU, so the tests simulate it and check what the
pipeline actually depends on: that no input row is lost, that a failed shard keeps its files,
that a finished one is cleared, and that cleanup refuses to delete outside the scratch root.

## Verification status

- The driver's planning, collection and finalisation are covered by the tests above.
- Everything that needs a GPU — the install, the tutorial check, timings and resource sizing —
  is recorded in [VERIFICATION.md](VERIFICATION.md) as it is carried out.
