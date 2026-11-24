# Nesso-1 on Euler

[Nesso-1](https://github.com/recursionpharma/nesso) scores a protein against a small molecule:
given a protein sequence and a ligand SMILES it returns a predicted binding affinity, a
binder/non-binder probability, and an entropy term saying how confidently it placed the ligand.
No MSA, no structure, about one second per pair on one GPU. It predicts affinity, not structure,
and does not produce a usable pose.

Everything runs from one Apptainer image. There is nothing to install.

---

## Run the example

```bash
cd /cluster/project/beltrao/lucla/repos/hpc-tools
nesso/nesso info                                  # confirms you can read the image
mkdir -p /cluster/scratch/$USER/nesso/slurm       # where your job logs go
```

Then open [`examples/01_first_run.ipynb`](examples/01_first_run.ipynb) on Euler and run it cell
by cell. It takes **twenty to thirty minutes**, most of it waiting for one GPU job —
9 minutes of prediction on an RTX 3090, 17 on an older TITAN RTX — and does one complete run on
24 pairs.

The pairs come from `example_pairs.py`: four drug targets fetched from UniProt by accession, six
drugs fetched from PubChem by name. Swapping those two lists gives you your own table. Each drug
is the established inhibitor of one of the four targets, marked in a `cognate` column, so a
working score has to rank each target's own drug first.

Any Python 3 kernel with `pandas` works. Every call into Nesso-1 is a subprocess into the image,
so nothing has to be installed into the kernel. Give it no GPU: every prediction goes to its own
`sbatch` job with a GPU of its own.

A kernel on a compute node has no direct internet, so step 1 goes through `module load
eth_proxy`; `sbatch` is accepted from there, so step 3 submits without dropping back to a login
node (both checked 2026-11-23, job 22026196).

### Getting a copy you can edit

The shared checkout is readable but **not writable by you**, and an editor needs somewhere to
save. Either of these works:

```bash
# A. your own clone — 672 files including .git, 1.7 MB:
git clone /cluster/project/beltrao/lucla/repos/hpc-tools ~/repos/hpc-tools

# B. just the notebook, pointed at the shared tooling:
mkdir -p /cluster/scratch/$USER/nesso && cd /cluster/scratch/$USER/nesso
cp /cluster/project/beltrao/lucla/repos/hpc-tools/nesso/examples/01_first_run.ipynb .
export HPC_TOOLS_NESSO_DIR=/cluster/project/beltrao/lucla/repos/hpc-tools/nesso
```

`site.conf` travels with the clone, so `nesso/nesso` finds the same image from your copy.

## Run your own data

### Input

A CSV. Three columns are required; every other column is carried through to the output
untouched, so your own identifiers travel with the scores.

| column | meaning |
|---|---|
| `pair_id` | unique, and usable as a filename: `[A-Za-z0-9][A-Za-z0-9._-]*` |
| `protein_sequence` | one-letter amino acids |
| `ligand_smiles` | the ligand whose affinity is predicted |
| `extra_ligands` | *optional*, `;`-separated: further ligands placed in the complex as their own chains. Each entry is a SMILES, or `ccd:<CODE>` for a Chemical Component Dictionary code, which is how a metal ion is given beside the macrocycle it sits in (`ccd:FE2`). Affinity is always predicted against `ligand_smiles`. |

### Output

`finalize` writes four things into `--dest`:

| file | what it holds |
|---|---|
| `<label>_nesso_scores.csv` | one row per input pair, with every score |
| `<label>_nesso_raw.zip` | the raw model output for every pair |
| `<label>_nesso_raw_manifest.csv` | which pair sits where in that zip |
| `<label>_provenance.txt` | what was run, with which image, on which GPUs |

The scores table has one row per input row, with these columns added:

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

**No row is ever dropped.** A pair that cannot be built, or that the model fails to score, keeps
its row with a `nesso:status` saying why. An output table shorter than its input has lost
something silently.

### Running

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
of yours, and both run on a login node. Step 1 needs `module load eth_proxy` if the pairs table
is built by fetching sequences.

Pairs are sharded **by protein**: the plan sorts by sequence so one protein's pairs stay together
and its ESM-2 embedding is computed once rather than once per shard.

A task whose shard table already exists exits 0 without recomputing, so a partly finished array
can be resubmitted as-is. A shard that fails keeps every one of its files on scratch; cleanup
happens only after the prediction exits cleanly, every pair is scored, and the zip's member count
and `unzip -t` both check out.

`plan --override` replaces a plan **and deletes the previous run's `inputs/`, `work/` and
`shards/` with it.** It has to: the submit script skips a shard whose table already exists, so a
stale table left in place would be finalized as the new run's result — invisibly, when the pair
ids match. It refuses to delete anything outside a scratch root.

### Pairs that cannot be scored: the same molecule twice

Nesso-1 1.0.0 cannot score a complex holding **two copies of one molecule** — ferric dicitrate,
say, which is two citrates around an iron. It writes one RDKit conformer per *distinct* ligand
entity but asks for one per ligand *chain* at inference, so the second copy raises `No pkl for
ligand chain '<id>'`. The documented `id: [B, C]` spelling groups the same way and does not avoid
it.

`plan` therefore refuses such a pair with `skipped: repeated_ligand_entity`, so it never reaches
a GPU and the reason travels in the output table. To score them anyway, give the repeated
molecules as one entity with a multi-fragment SMILES (`A.A`) — chemically the same set of
molecules, but a different entity partition from the one another tool may have used, so say so
when comparing.

### Scores are not reproducible between runs

Measured 2026-11-19: the same 24 pairs scored three times — same pairs table, same seed (42),
same model revision, same cores and memory, cuEquivariance kernels off throughout — gave
`affinity_pred_value` differing by up to **0.25 log<sub>10</sub> units** between two runs on the
*same* GPU model. 18 of 24 agreed within 0.03; the drift concentrates on a few pairs, here mostly
on the longest protein.

Nesso-1 seeds with `pl.seed_everything(seed, workers=True)` and never asks for deterministic CUDA
kernels, so operations implemented with atomics are free to sum in a different order each run.
Seeding makes the *random* choices repeatable, not the floating-point reductions.

- **Keep a comparison inside one run.** Two pairs scored in the same job are comparable; the same
  pair scored in two jobs is not, to better than about 0.25.
- **Never top up a finished set with a later job** and rank the combined table. Re-plan and re-run
  the whole thing — the same rule that applies to changing the kernel setting.
- **Shard by protein and this mostly takes care of itself**, which is what `plan` already does. In
  a 4,929-pair production run, 245 of 252 proteins had all their pairs in one shard, hence one
  job. The 7 that straddled a shard boundary are the exposed ones.
- **For scale**: 0.25 is most of a factor of two in predicted IC50, the same order as the gap
  between neighbouring pairs in a within-protein ranking.

### Sizing the request

The `#SBATCH` lines in `submit_nesso.sh` are **starting points for a pilot, not measured values.**
Run one or two shards, read the real usage, then pass the measured numbers on the command line:

```bash
sbatch --array=0 nesso/submit_nesso.sh $RUN      # pilot
seff <jobid>
sacct -j <jobid> --format=JobID,MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS,ReqTRES%40
sbatch --array=1-N --time=HH:MM:SS --mem-per-cpu=NNNN nesso/submit_nesso.sh $RUN
```

Trim it whichever way the pilot points, and re-measure when the workload changes shape — much
longer proteins, many more ligands per protein, a different shard size. Each job prints its own
seconds-per-pair, which is the number to size `--time` from.

**The image needs about 2 GB more host memory than a virtual environment does.** Measured on the
same 24 pairs: 10.82 and 10.84 GB over two venv runs, against 12.84 GB through the image — 92% of
a 14 GB request, where the venv sat at 77%. The likely cause is the image's own library pages held
in the page cache and charged to the job's cgroup, which would make it a roughly fixed cost rather
than one growing with the shard; that has **not** been measured on a larger shard. Size from a
pilot run through the engine you will actually use.

**What one measured run looked like.** A 600-pair shard on one RTX 4090, **without** the
cuEquivariance kernels (2026-11-16):

| | |
|---|---|
| throughput | **2.47 s/pair** (25 min for the shard) |
| memory | **15.95 GB**, against the 16 GB requested — 99.7%, uncomfortably close to an OOM |
| CPU | 25.7% of 4 cores, so most of the cores were idle |
| failures | none; 600 of 600 scored |

Sized from that, later shards were given 2 cores and 24 GB. At 2 cores their peak was only
11.5 GB, so the pilot's 15.95 GB was largely its four dataloader workers, and **2 cores with 14 GB
is the right request** for a shard of this shape. Memory is the binding constraint here, not cores
and not GPU memory.

**Runtime scales with protein length, steeply, and with the card you land on.** The figures above
are for proteins of a median 238 residues. The 24-pair example in
[`examples/`](examples/01_first_run.ipynb), whose targets run from 260 to 1,210 residues, took
15.8 s/pair on an RTX 4090 and 21.1–22.0 s/pair on an RTX 3090 — though those also carry the fixed
cost of loading the model and embedding the proteins, spread over only 24 pairs. Its memory was
10.8 GB of the 14 GB requested. Size a run of larger proteins from its own pilot.

Turning the cuEquivariance kernels on (`NESSO_KERNELS=1`) is the obvious lever if throughput
matters, but it changes the numerics: never mix kernel and no-kernel scores inside one run.

---

## Reference

**The image.** `nesso/site.conf` names the directory; `nesso info` says which image would run.

```
/cluster/work/beltrao/lucla/hpc-tools/nesso/current/nesso.sif
```

It holds the pinned package set, the model checkpoint, the chemical component dictionary and the
ESM-2 650M weights — 5.5 GB in one file.

```bash
nesso/nesso info          # which image, and the checksums of everything in it
nesso/nesso info --verify # ... and recompute the image's own sha256
```

**Building a new one.** The weights are not downloaded during the build; they come from a cache
`setup_nesso_euler.sh` staged, whose Hugging Face revision is recorded, and the build refuses to
proceed unless that revision is the expected one — so an image can only hold the weights the
reference scores were produced with.

```bash
mkdir -p /cluster/scratch/$USER/hpc-tools/nesso_build
sbatch nesso/build_image.sbatch
```

A new image installs as its own release and `current` is **not** moved. Point it there only after
checking it against the environment the reference scores ran in:

```bash
sbatch nesso/compare_to_venv.sbatch $RUN 0      # scores one shard venv / image / image again
ln -sfn nesso-1.0.0_YYYYMMDD /cluster/work/beltrao/lucla/hpc-tools/nesso/current
```

**A virtual environment instead.** `bash nesso/setup_nesso_euler.sh` builds one, and
`NESSO_VENV=<path>` makes a job use it — the route for developing against a modified Nesso-1, and
what the image is checked against. Set it explicitly: nothing falls back to a virtual environment
on its own, since a silent fallback would let two people score the same pairs against two
different builds with nothing in either log to say so.

**Tests.**

```bash
python3 -m pytest nesso/test_nesso_run.py
```

The prediction step is the only part needing a GPU, so the tests simulate it and check that no
input row is lost, that a failed shard keeps its files, that a finished one is cleared, and that
cleanup refuses to delete outside the scratch root.

**[VERIFICATION.md](VERIFICATION.md)** — everything needing a GPU: the install, the tutorial
check, timings and resource sizing, recorded as it was carried out.
