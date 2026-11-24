# Boltz-2 on Euler

> Written for ETH Zurich's Euler cluster. If you are not in `biol-imsb-beltrao` there, read
> [Who this is for](../README.md#who-this-is-for) first — the images named below are readable
> by that group, and building your own is one job.

Predict the structure of a protein, a complex, or a protein with a bound small molecule, on the
cluster. Boltz-2 is an open co-folding model: you give it the molecules, it gives you a
structure, a confidence score and — if you ask — a predicted binding affinity.

Everything runs from one shared Apptainer image, so there is nothing to install.

---

## 1. Set up (once per shell)

```bash
TOOL=/cluster/project/beltrao/lucla/repos/hpc-tools/boltz2
export PATH="$TOOL:$PATH"
boltz2 info
```

`boltz2 info` prints the image it found. If it prints nothing useful, see
[If something goes wrong](#if-something-goes-wrong). The commands below reach the scripts
through `$TOOL`, so they work from any folder.

Check the installation works before trusting it:

```bash
"$TOOL/check_install.sh"
```

---

## 2. Your first run

Four commands. They are explained one by one below; this is the whole shape of it.

```bash
RUN=/cluster/scratch/$USER/boltz2/myrun
STORE=/cluster/project/beltrao/$USER/boltz2_msas      # alignments, reused across all your runs
mkdir -p /cluster/scratch/$USER/boltz2/slurm "$STORE"

sbatch "$TOOL/submit_msa.sh" my_inputs/ "$STORE"      # 1. alignments  (CPU, needs network)
boltz2 python "$TOOL/boltz2_run.py" plan \
    --inputs my_inputs/ --run-dir "$RUN" --msa-store "$STORE"   # 2. plan
bash "$RUN/plan/submit.sh"                            # 3. predict     (GPU) -- read it first
boltz2 python "$TOOL/boltz2_run.py" finalize \
    --run-dir "$RUN" --dest /cluster/project/beltrao/$USER/results --label myrun   # 4. collect
```

There is a worked example you can actually run, as a notebook:
[`examples/01_first_run.ipynb`](examples/01_first_run.ipynb). Copy it into your home directory
and open it at jupyter.euler.hpc.ethz.ch:

```bash
cp /cluster/project/beltrao/lucla/repos/hpc-tools/boltz2/examples/01_first_run.ipynb ~/
```

Copy it rather than opening the shared one: JupyterHub can reach the shared copy, but running
it there would save your outputs into the shared checkout. A session of 1 core and 1 GB is
enough — the notebook peaked at about 380 MB over two full runs — but JupyterLab also reopens your previous tabs,
and large files among them can use up 1 GB before the notebook starts. If the session dies at
startup, ask for 2 GB. Ask for no GPU: the predictions run as their own GPU jobs, so a GPU on
the session would sit idle. Two hours is plenty.

### Step 1 — alignments

```bash
sbatch "$TOOL/submit_msa.sh" my_inputs/ "$STORE"
```

An **MSA** (multiple sequence alignment) is a set of evolutionarily related sequences lined up
against your protein. Boltz-2 uses it to see which positions are conserved, and it is the single
biggest factor in how good the prediction is. This step fetches one from the public ColabFold
server.

It is a separate step because alignments are slow to fetch and completely reusable. The store is
keyed by the sequence itself, so a protein fetched today is never fetched again — not by this
run, not by anyone else pointing at the same store. Put the store in project storage and keep it.

If your input has **more than one different protein chain**, this step also fetches a *paired*
alignment for that combination, which tells the model which rows of the two chains' alignments
come from the same organism. That is what makes a complex of two different proteins work
properly, and it is cached per combination.

Run it again whenever you add inputs; it only fetches what is missing.

### Step 2 — plan

```bash
boltz2 python "$TOOL/boltz2_run.py" plan \
    --inputs my_inputs/ --run-dir "$RUN" --msa-store "$STORE" --per-shard 50
```

Turns your inputs into a run directory on scratch: alignments filled in, inputs grouped into
**shards** (one shard = one SLURM array task = one GPU), and the sampling options fixed. Fixing
them here is what makes scores from different shards comparable with each other — to change
them, plan a new run.

It also works out what each shard needs from SLURM, and groups shards into **bands** by size:

```
  band <= 400 tokens: array 0-1, gpumem 20g, time 00:10:00  (4 input(s), memory measured)
  band <= 400 tokens [affinity]: array 2-2, gpumem 20g, time 00:10:00  (1 input(s), memory measured)
  band <= 800 tokens: array 3-3, gpumem 24g, time 00:10:00  (2 input(s), memory measured)
```

A **token** is one residue, or one heavy atom of a ligand. Memory grows roughly with the square
of the token count, so a run of mostly small complexes should not be charged the memory its three
biggest need — hence one band each. Inputs that ask for affinity get shards of their own (see
[Inputs](#inputs) for why).

**Read the memory basis.** `measured` means jobs have actually completed at that setting here.
`extrapolated` means none have, and the number comes from the shape of the model rather than an
observation.

Nothing is ever dropped silently: an input that cannot be planned keeps a row in
`$RUN/plan/jobs.csv` saying why.

### Step 3 — predict

`plan` wrote the submission out as a short script, one `sbatch` line per band. Read it, then run
it:

```bash
cat "$RUN/plan/submit.sh"
bash "$RUN/plan/submit.sh"
```

Each task predicts its shard, then collapses the output into one table and one zip and deletes
the shard's scratch tree — but only if every prediction finished and the archive verifies. **A
failed shard keeps all of its files**, because that is when you need them.

If one band fails for lack of memory or time, edit that band's line in `submit.sh` and run just
that line again. A shard that already finished exits at once, and Boltz skips predictions that
already exist, so a resubmission redoes only what is missing.

### Step 4 — finalize

```bash
boltz2 python "$TOOL/boltz2_run.py" finalize --run-dir "$RUN" \
    --dest /cluster/project/beltrao/$USER/results --label myrun
```

Collects everything into project storage. Refuses to run while shards are unfinished, unless you
pass `--allow-open`.

### Before and after: what does it cost?

To estimate a run before planning it — say, to see whether it needs scarce large-memory cards:

```bash
boltz2 python "$TOOL/boltz2_run.py" size --inputs my_inputs/ --samples 5
```

And after a run, to see what it really used:

```bash
python3 "$TOOL/boltz2_run.py" size --from-run "$RUN"
```

That one runs with plain `python3`, outside the image, because `sacct` (where SLURM keeps what a
job used) is not inside the image. For a size nobody has run before, predict one shard first,
read this back, and put the real numbers into [`sizing.tsv`](sizing.tsv), which says how.

> SLURM cannot measure GPU memory — `sacct` reports host memory only. So a `gpumem` band is only
> ever confirmed by jobs not failing at it. Raise one only after seeing a real CUDA
> out-of-memory error, and lower it if jobs never come close.

---

## Inputs

One file per complex, in Boltz-2's own YAML format. The file name is the job name, so
`lysozyme.yaml` gives you results called `lysozyme`.

### The shapes

**A protein on its own**

```yaml
version: 1
sequences:
  - protein: {id: A, sequence: MKALIVLG..., msa: auto}
```

`msa: auto` means "fill this in from the store" — it is this tool's marker, not a Boltz keyword,
and `plan` replaces it with a real path. Use `msa: empty` to predict from the single sequence
with no alignment: much faster, and much less accurate.

**A protein with a ligand, and an affinity prediction**

```yaml
version: 1
sequences:
  - protein: {id: A, sequence: KETAAAKFER..., msa: auto}
  - ligand:  {id: B, smiles: 'CC(=O)[O-]'}        # or: ccd: ATP
properties:
  - affinity: {binder: B}
```

A ligand is named either by **SMILES** (a text form of a chemical structure) or by a **CCD code**
(the PDB's three-letter dictionary code, e.g. `ATP`). The image carries the whole dictionary, so
CCD codes need no download.

Affinity is scored for exactly one ligand chain. Boltz-2's limit is 128 atoms, and its authors do
not recommend much over 56; `plan` refuses an input that breaks this rather than letting it fail
on the GPU.

> Inputs that ask for affinity are put in **their own shards**, automatically. Boltz-2 2.2.1 runs
> its affinity stage over every input in a call, including ones that never asked for it, and then
> dies on a file it never wrote for them — after predicting the structures, so the GPU time is
> gone. You do not need to do anything about this, but it is why an affinity run has more shards
> than you might expect. Affinity also costs extra time — about a minute per input on top of the
> structure prediction, measured on one shard — because a second model runs.

**Several copies of the same chain** — one entity, several ids:

```yaml
  - protein: {id: [A, B], sequence: MKPVTLYDV..., msa: auto}
```

**Two different proteins** — two entities. This is the case that needs a paired alignment, which
step 1 handles for you:

```yaml
  - protein: {id: A, sequence: MKPVTLYDV..., msa: auto}
  - protein: {id: B, sequence: MGSSHHHHH..., msa: auto}
```

**DNA or RNA** — same shape, no `msa`. Double-stranded means two chains, the second the reverse
complement of the first:

```yaml
  - dna: {id: B, sequence: AATTGTGAGCGGATAACAATT}
  - dna: {id: C, sequence: AATTGTTATCCGCTCACAATT}
```

**Pointing a ligand at a pocket** — optional, when you know roughly where it binds:

```yaml
constraints:
  - pocket: {binder: B, contacts: [[A, 12], [A, 41]], max_distance: 6, force: false}
```

Residue numbering starts at 1, **counted along the sequence you put in the file** —
`force: true` makes it a hard restraint instead of a hint.

> That numbering is a common place to go wrong. A UniProt record is the precursor, signal peptide
> and all, so its position 12 is usually not position 12 of the mature protein that the
> literature and the PDB number against. Decide which you want, and number against the sequence
> actually in your YAML.

Boltz-2 also takes covalent bond constraints, structural templates and modified residues. Those
are in [the Boltz-2 input documentation](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md);
anything valid there works here, and `plan` passes it through untouched.

> Any file path inside an input (a template, your own alignment) must be somewhere the job can
> see it. The wrapper binds `/cluster`, so paths under `/cluster` work and paths elsewhere do not.

### From a table instead

If you are building many simple complexes, write a table and convert it — one row per chain:

```csv
id,entity,sequence,msa,copies,affinity
complex1,protein,MKALIVLG...,auto,1,
complex1,ccd,ATP,,1,yes
complex2,protein,MGSSHHHH...,auto,2,
```

```bash
boltz2 python "$TOOL/boltz2_run.py" prep --table chains.csv --out my_inputs/
```

`entity` is `protein`, `dna`, `rna`, `ligand` (SMILES) or `ccd`. Chains are lettered in row order
unless you add a `chain` column. Constraints and templates cannot be expressed in a table — write
those as YAML.

---

## Outputs

`finalize` writes four things into `--dest`:

| File | What it holds |
|---|---|
| `<label>_boltz2_scores.csv` | one row per input, with every confidence score |
| `<label>_boltz2_shard_*.zip` | the structures and full records |
| `<label>_boltz2_manifest.csv` | which zip holds which prediction |
| `<label>_boltz2_provenance.txt` | what was run, with which image, on which GPUs |

### The scores table

One row per input — including inputs that failed, which keep a row saying why. Columns:

| Column | Meaning |
|---|---|
| `job_id` | the input file's name |
| `boltz2:status` | `scored`, or why not |
| `boltz2:n_tokens` | the size that drove the resource request |
| `boltz2:confidence_score` | Boltz-2's overall confidence, 0–1, higher is better |
| `boltz2:ptm` / `iptm` | predicted TM-score for the whole structure / across chain interfaces |
| `boltz2:ligand_iptm` | the same, restricted to the ligand interface |
| `boltz2:complex_plddt` | mean per-atom confidence, 0–1 |
| `boltz2:complex_pde`, `ipde` | predicted distance error, in Ångström — **lower is better** |
| `boltz2:affinity_pred_value` | log₁₀(IC50 in µM) — **lower means stronger** predicted binding |
| `boltz2:affinity_probability_binary` | probability it binds at all, 0–1, higher is more likely |
| `boltz2:gpu` | which card predicted it |

Each score appears twice: `boltz2:<name>` is the top-ranked sample, and `boltz2:<name>_mean` is
the mean over all samples. Which columns exist depends on what you predicted — `ligand_iptm`
only appears if there was a ligand, `affinity_*` only if you asked for affinity.

### The structures

In the zips, not loose, because shared storage runs out of **file-count quota** long before it
runs out of space — one prediction writes about twenty files. Read one out without unpacking:

```bash
unzip -p myrun_boltz2_shard_0000.zip lysozyme/lysozyme_model_0.cif > lysozyme.cif
```

`model_0` is the sample Boltz ranked first by confidence.

---

## Things to adjust

| What | How | Default, and when to change it |
|---|---|---|
| Samples per input | `plan --samples N` | 5. Each is an independent attempt; more gives a better best-of and a usable spread, at linear cost. 1 is fine for "does it run". |
| Steering | `plan --no-potentials` | On. It enforces physically sensible geometry and costs about 2.4× the runtime. Leave it on unless you are reproducing something that ran without it. |
| Seed | `plan --seed N` | 1. Boltz-2 has no default seed, so one is always set. Note that the same seed on a *different GPU model* still gives slightly different numbers. |
| Affinity | per input, `properties:` | Off. One ligand chain only. |
| Inputs per shard | `plan --per-shard N` | 50. Fewer = more parallel jobs, more scheduler overhead, more host memory headroom. |
| Recycling / sampling steps | `plan --recycling-steps`, `--sampling-steps` | 3 / 200, Boltz's defaults. Raising them trades time for a little accuracy. |
| Anything else | `plan --boltz-arg=--foo` | Repeatable; passed straight to `boltz predict` and recorded in the provenance. |
| Pin the GPU model | `EXPECT_GPU="RTX 4090" sbatch ...` | Off. Turn it on when every score in a run must be directly comparable. |

---

## If something goes wrong

**`boltz2 info` finds no image** — the install moved, or `BOLTZ2_HOME` is set wrong. Check
[`site.conf`](site.conf).

**A job fails with CUDA out of memory** — raise `--gres=gpumem:` on that band's line in
`$RUN/plan/submit.sh`, run that line again, and record the new value in `sizing.tsv`. This is the
one case where raising a resource request without a pilot is right: the failure *is* the
measurement.

**A job hits its walltime** — check which GPU it got (`boltz2:gpu`, or the shard receipt). Runtime
varies several-fold between card types, so the same work can fit on one and time out on another.
Either raise `--time` on that band's line, or pin the card with `EXPECT_GPU`.

**A shard failed and I want to see why** — nothing was deleted. The inputs and partial outputs are
under `$RUN/work/<shard>/` and `$RUN/inputs/<shard>/`, and the SLURM logs under
`/cluster/scratch/$USER/boltz2/slurm/`.

**Some inputs were skipped** — `$RUN/plan/jobs.csv` has a row for every input with the reason.

**`msa` step fails** — it needs network, so it must run as a job with `eth_proxy` loaded (which
`submit_msa.sh` does) or on a login node. Compute nodes have no network by default. The store
keeps whatever arrived, so just run it again.

**`finalize` says more than one image** — shards were predicted with different image versions, so
their scores are not strictly comparable. The provenance file names them.

---

## Moving the image elsewhere

The image is relocatable by design. To move it to a group-wide location:

```bash
cp -a /cluster/work/beltrao/lucla/hpc-tools/boltz2/<release> /new/place/
ln -sfn <release> /new/place/current
```

Then change the one line in [`site.conf`](site.conf), and confirm the copy is identical:

```bash
BOLTZ2_HOME=/new/place boltz2 info --verify
```

`current` is a relative symlink, so it survives the move.

---

## More

- [`VERIFICATION.md`](VERIFICATION.md) — what has actually been tested, with the evidence, and
  what has not.
- [`sizing.tsv`](sizing.tsv) — the resource bands, each with its provenance.
- [Boltz-2 input documentation](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md)
  — the full input format. This tool pins Boltz-2 **2.2.1**.
- Tests: `python3 -m pytest "$TOOL/test_boltz2_run.py" -q` — they need pytest and PyYAML, not Boltz, so
  any Python with those will do. The image has no pytest by design: it carries the reference
  environment's package list exactly.
