# Verification record — boltz2

What has actually been run, and what it showed. Anything not listed here has not been checked.
Claims of correctness belong in this file, with the evidence; not in the README or in a comment.

The raw records behind the job ids below are kept in one archive. It holds the SLURM logs of the
prediction, alignment and comparison jobs, plus the shard receipts, plans, score tables and the
venv comparison's per-sample confidence files; alignments and structures were not kept:
`/cluster/project/beltrao/lucla/run_archives/hpc-tools_boltz2/boltz2_tool_tests_20261120T142559.zip`.
A manifest sits beside it. Read one file with `unzip -p <zip> <member>`.

## Release `boltz-2.2.1_20261117`

`image_sha256 cb5884488699d2c3db4343e10f765f3bc8d2a7dad463bb1472695b6bbde618e8`, 7.2 GB,
built by SLURM job 21222014 on eu-a2p-279 from hpc-tools `d0d332d`.

### Built, on Euler, without admin rights — **confirmed 2026-11-17**

`apptainer build --fakeroot` works on a compute node. The account is not in `/etc/subuid`, so
Apptainer falls back to a root-mapped namespace; that is enough. Three attempts were needed:

| job | outcome | cause |
|---|---|---|
| 21179111 | failed in 38 s | `%files` sources are resolved against the working directory, not the definition file's directory |
| 21184527 | failed at the self-test | numba's compile cache was pointed at `$TMPDIR`, a host path (`/scratch/tmp.<jobid>.<user>`) that does not exist inside the container |
| 21222014 | **built** | cache pointed at `/tmp`, which is mounted inside the container |

The second is worth keeping in mind: boltz 2.2.1 compiles one function with
`@numba.njit(cache=True)`, and numba **raises** rather than running uncached when it can write
neither beside the source file (read-only in an image) nor at `NUMBA_CACHE_DIR`. So the image
cannot even print `boltz predict --help` without a writable cache directory that exists inside
the container. The wrapper sets one under scratch and binds it; the image carries a `/tmp`
fallback for anyone calling `apptainer exec` directly.

Build resources, measured: MaxRSS 16.2 GB (21181125) and 21.8 GB (21184527) of 32 GB requested,
under 9 min of the 3 h. The peak is `mksquashfs`; it varied by 5 GB between otherwise identical
builds, which is why the request is not trimmed to either observation.

### Contents match the reference environment — **confirmed 2026-11-17**

`requirements.lock.txt` is a `pip freeze --all` of the reference virtual environment, the one
every steered production fold ran in. Every package in that environment was installed on
2026-09-09 and none has changed since. The image installs that list with `--no-deps`, so nothing is re-resolved at build time.
`pip check` passed inside the image.

Weights are checked against fixed md5 sums before they enter the image, whichever way they
arrive: `boltz2_conf.ckpt 2f0a1775bf8fc366a1a85e2019eca288` (as pinned in the reference
environment's conda spec), `boltz2_aff.ckpt 8e93dadedd6edb7a4d170f6051b99ec0` and
`mols.tar 5399454dcbbba695ab01f88deede577c` (both measured in the production cache on
2026-11-16). The ligand dictionary holds 45,227 component files inside the image.

### Folds end to end — **confirmed 2026-11-17**

SLURM job 21228744, one pair, on an NVIDIA TITAN RTX: bovine ribonuclease A (UniProt P61823, 150
residues, fetched by `check_install.sh`) with cytidine monophosphate (`ccd:C5P`), single-sequence
mode, 1 diffusion sample, steering on.

The whole pipeline ran: the fold job read the protocol back from the plan, Boltz predicted in
57 s, `collect` scored it, verified the 5-member zip, cleared the shard's scratch tree and marked
it done. `boltz2:confidence_score` 0.909, which is a plausible value for a small, well-determined
protein, though nothing here tests whether it is the *right* value.

Job MaxRSS 5.8 GB of the 18 GB requested, 5 min 41 s wall for one pair — most of it loading the
2.3 GB checkpoint and compiling the numba function with a cold cache, not predicting.

`boltz2 info --verify` recomputed the image's sha256 and matched `BUILD_INFO.txt`.

### Alignments fetch — **confirmed 2026-11-17**

SLURM job 21229936: `submit_msa.sh` loaded `eth_proxy`, reached `api.colabfold.com` from a compute
node, and stored one alignment for the RNase A sequence in 2 s, named by the sha1 of that
sequence. Only one sequence was requested, so this shows the path works, not how it behaves under
load or when the server queues.

The alignment is written as boltz's parsed `.csv`; an existing ColabFold store holds `.a3m`. Both
are found by sequence sha1 and both are accepted, so an existing store can be used unchanged —
that store was in fact the input source for the comparison below.

### Affinity — **confirmed 2026-11-17**

SLURM job 21229952, same RNase A / `ccd:C5P` pair, `plan --affinity`, on an NVIDIA TITAN RTX.
All six affinity fields were produced and carried into the shard table:
`affinity_pred_value 1.049` (log<sub>10</sub>(IC<sub>50</sub> / µM), i.e. about 11 µM) from
ensemble members 1.199 and 0.898, and `affinity_probability_binary 0.151`.

Cytidine monophosphate is an RNase A product and is known to bind it weakly, so a prediction in
the tens of µM is of the right order — but this is one pair folded in single-sequence mode with
one diffusion sample, and nothing here was compared against a measured affinity. It shows the
path runs and the numbers land in the right columns. **It is not evidence that the affinity
predictions are accurate.**

Cost: 254 s against 57 s for the same pair without affinity, i.e. about 4.5x, because a second
model runs. Ask for affinity only when it will be used.

### Reproduces the virtual environment — **confirmed 2026-11-17**

SLURM job 21229859, on one NVIDIA GeForce RTX 3090, in one job: 12 protein–ligand pairs folded
three times, by the virtual environment once and by the image twice, at the production protocol
(5 diffusion samples, seed 1, steering on). Proteins 63–587 residues, sampled across the length
range of an existing ColabFold store and using that store's own alignments; ligands are
CCD cofactor codes. Inputs, options, GPU and job were identical throughout, so the only thing
that differed was which Boltz-2 ran.

**Exact agreement is not the right question.** Boltz-2 on a GPU is not deterministic even with a
fixed seed: the same software disagrees with itself. The image's two runs are therefore the
yardstick, and the test asks whether the venv-against-image difference is any larger than that.
It is paired — each (pair, field) cross-difference sits beside the same-software difference for
that same pair — because the differences vary far more between pairs than between environments.

| | median | max |
|---|---|---|
| venv vs image | 0.00019 | 0.2517 |
| image vs image (yardstick) | 0.00023 | 0.2736 |

Wilcoxon signed-rank over all 108 paired comparisons (12 pairs × 9 fields): **p = 0.498**, and the
cross-environment median is the *smaller* of the two. No field is significant on its own either
(smallest p = 0.131, on a field where the cross difference is smaller). On `iptm`, the field
candidate ligands are ranked by, the largest single venv-vs-image gap is 0.0151 — less than half
the largest same-software gap of 0.0337, and well inside the refold noise this project has
measured elsewhere.

`protein_iptm` is identical to the last digit in all three runs, on every pair. The protein-only
confidence is computed deterministically; everything that moves between runs moves in the terms
that involve the ligand, which is where the diffusion sampling acts. That is a mechanism, not a
coincidence, and it is what one would expect if the two environments hold the same software.

**Conclusion: nothing in this test distinguishes the image from the virtual environment.** That
is absence of evidence over 12 pairs, not proof of identity; but there is no reason here to keep
the two apart.

**A methodological correction, recorded because it nearly produced the wrong answer.** The first
version of `compare_runs.py` decided by comparing the two *maxima* per field. On this very data
it reported 3 of 9 fields as "larger" and returned a failure — while the medians and the paired
test showed nothing whatever. The maximum of a dozen values is the noisiest summary available and
moves with a single observation, so comparing two such maxima is close to a coin toss. The script
now decides by the paired test over all fields, with per-field values shown as description only
(nine uncorrected tests would throw up a "significant" one often enough by themselves). The
numbers in the table above come from the corrected script re-run on the same output.

### Driver behaviour — **confirmed 2026-11-18**

59 tests in `test_boltz2_run.py` pass. They simulate the prediction step, so they check the
pipeline's guarantees rather than the model: no input row is lost, a failed or incomplete shard
keeps every one of its files, a finished shard is cleared only after its archive verifies,
cleanup refuses to act outside the scratch root, the protocol a run was planned with is what its
jobs predict with and what its provenance reports, and tables are written without stray carriage
returns.

The generalised driver adds tests for the input formats (monomer, homomultimer, heteromultimer,
protein–DNA, ligand by SMILES and by CCD code, FASTA), for the malformed inputs it must reject,
and for score collection reading *every* scalar field a confidence record holds rather than a
fixed list.

> **Caveat on the older evidence above.** Jobs 21228744, 21229936, 21229952 and 21229859 ran the
> *previous*, pair-shaped driver. They remain valid evidence about the image, the wrapper and the
> shape of the pipeline — none of which changed — but they did not exercise the current driver's
> code. See "The generalised driver on the cluster" below.

### A sharding bug the sizing work exposed — **found and fixed 2026-11-18**

`size` grouped inputs into size bands and printed one `sbatch --array=` range per band, while
`plan` sharded by raw sorted size. Whenever a band boundary fell inside a shard the two
disagreed, so a band's array range could name shards belonging to a different band — carrying
one band's memory request to complexes it does not fit. It was caught by running both on the
same six example inputs and seeing 4 shards claimed against 3 written.

Both now call one function, `band_plan`, and two tests pin the agreement: that `plan`'s
assignment equals `band_plan`'s, and that no shard ever mixes two bands.

### Boltz-2 cannot mix affinity and non-affinity inputs in one call — **found 2026-11-18**

Euler job 21396655_1 predicted two inputs, one of which requested affinity. Both structures were
produced and scored; then the affinity stage ran — over *both* inputs, including the one that
never asked — and died:

```
FileNotFoundError: .../predictions/05_heterodimer/pre_affinity_05_heterodimer.npz
```

`pre_affinity_<id>.npz` is written only for inputs whose YAML carries a `properties: affinity`
block, but Boltz-2 2.2.1's affinity stage iterates over every record in the call. The failure
comes *after* the structure prediction, so the GPU time is spent and the job still exits
non-zero.

`plan` now shards the two kinds apart, and a test pins it. This is a constraint of Boltz-2, not
of this driver; it is recorded here because nothing in the Boltz-2 documentation mentions it.

### The generalised driver, end to end on Euler — **confirmed 2026-11-18**

Six example inputs covering the shapes the tool claims to support, all generated by fetching
sequences from UniProt by accession (`examples/make_examples.py`). Alignments by SLURM job
21395977; predictions by 21404548, 21404551 and 21404596, all on NVIDIA RTX 4090; 2 diffusion
samples, seed 1, steering on. All four shards completed, all six inputs scored.

| input | tokens | what it exercises | confidence | ipTM |
|---|---|---|---|---|
| `01_monomer` | 147 | a protein alone | 0.919 | — (0, no interface) |
| `02_protein_ligand` | 171 | ligand by CCD code, affinity | 0.933 | 0.948 |
| `03_homodimer` | 720 | one entity, two chain ids | 0.885 | 0.835 |
| `04_protein_dna` | 402 | double-stranded DNA chains | 0.878 | 0.875 |
| `05_heterodimer` | 247 | two different proteins, paired MSA | 0.926 | 0.934 |
| `06_pocket_constraint` | 154 | a pocket constraint, ligand by SMILES | 0.829 | 0.517 |

Affinity for `02_protein_ligand`: `affinity_pred_value` 1.285, i.e. about 19 µM — the same order
as the 11 µM measured for the same pair on 2026-11-17 under a different protocol. Still **no
evidence that affinity predictions are accurate**; nothing here was compared to a measurement.

Two of these rows are worth reading rather than just counting.

**The paired alignment works.** For `05_heterodimer` the two chains' alignment files carry 4,996
rows each with non-negative keys, and those key *sets are identical*, so row *k* of one chain is
the same organism as row *k* of the other — which is the whole point of pairing. Each then has
its own `-1` tail (7,271 and 6,405 rows), and the per-sequence store file for the same sequence
has nothing but `-1`. That is the structure `boltz.main.compute_msa` produces, and it is what
makes the interface score of 0.934 on a known-tight complex meaningful rather than lucky.

**The constrained example scores low, and should.** Its ligand interface score, 0.517, is far
below the unconstrained protein–ligand example's 0.948. The contacts in that input are numbered
against a UniProt precursor, so they are offset by its 26-residue signal peptide and do not point
at the real site. A constraint aimed somewhere unhelpful ought to hurt, and it did.

### What the pilot measured about cost — **2026-11-18**

The striking number: a 2-input shard took 160 s while a 1-input shard took 155 s. The fixed cost
dominates at this size, so the previous model — a flat rate per sample per input — was wrong in
shape, not just in value. Fitting instead

    seconds = startup + rate x (tokens in the shard) x (diffusion samples)

to two of the three structure-only shards gives startup 120 s and rate 0.0670 s, and predicts the
third to within 2.2 s. The affinity shard cost 59 s more than that model predicts for it.

This is a two-parameter fit checked against one held-out point, on one GPU type, with ordinary
ColabFold alignments. It is weak evidence — but it is the right *shape*, and `size` now uses it,
with a safety factor of 1.5 because the same work has been seen to vary several-fold across card
types. `startup` is not a constant of nature: it covers container start, loading the 2.3 GB
checkpoint and processing the alignments, so deeper alignments and slower cards will raise it.

The two memory bands that were exercised (20 GB up to 247 tokens, 24 GB up to 720) are now marked
`measured` in `sizing.tsv`, in the only sense SLURM allows: jobs completed at those settings.
Nothing tried a smaller request, so the headroom is unknown and the bands may be generous.

### The first-run notebook in JupyterHub — **confirmed 2026-11-20**

`examples/01_first_run.ipynb`, copied to the home directory as the README tells a new user to,
run with "Run All" in Euler's JupyterHub: JupyterLab, 1 core, 1 GB, 1 hour (job 21646273). It
ran twice, both times with all 19 code cells completing in order and no error output. Run 1
(jobs 21646527, 21646602/08/09) exposed two display defects; the version fixed for them ran as
run 2 (jobs 21649875, 21650163/65/78). Every step ran from inside the kernel: the image check,
the UniProt fetch through `eth_proxy`, `sbatch`/`squeue`/`sacct` (through a login shell, which
is what puts them on PATH), planning, the generated `submit.sh` with its three bands, finalize,
the scores table, structure extraction and `size --from-run`.

What the test found, in the order it found it:

- **1 GB is enough for the notebook, not for every JupyterLab session.** The first session was
  killed for memory (`OUT_OF_MEMORY`, peak 983 MB of 1000) before a single cell had run:
  JupyterLab was reopening the account's previous workspace, large CSV files included. In a
  clean workspace the session peaked at 383 MB over both full runs. The README and the notebook
  now say both things.
- **JupyterLab's server root here is `/`, not the home directory**; the file browser merely
  opens at home. A user can therefore open the shared copy in place, and running it there would
  write their outputs into the shared checkout. The instruction to copy it first stands, now
  with that reason.
- **Every cell's output appeared twice**, the second time as a string literal full of `\n`. The
  helper both printed a command's output and returned it, and Jupyter displays a cell's last
  returned value. Split into `run` (shows output, returns nothing) and `capture` (returns it,
  shows nothing); run 2 has no such output anywhere.
- **Long commands were silent until they finished**, which reads as a hang. `run` now streams
  output line by line.
- **The provenance recorded the driver commit as "unknown"** — in every run, not only this one.
  `plan` runs inside the image, and the image has no git binary. The commit is now read from
  the repository's files when git is missing, and the record says the dirty state was not
  checked. Tested against a loose ref, a packed ref and a worktree, and against the real
  repository, where it returns the same hash as git.

Measurements the runs added, recorded in `sizing.tsv`: the timing model was within -12% to +4%
on RTX 4090 at 1 diffusion sample, out of sample since it was fitted at 2; RTX 3090 jobs took
25-41% longer than predicted, all still inside their walltimes; and the affinity shard reached
15.4-15.7 GB of host memory with one input, about 86% of what it asks for.

The two runs landed on a mix of RTX 3090 and 4090 cards, which `finalize` reports in the
provenance. That is expected behaviour rather than a defect, and the notebook now points it out.

**Checking it for a second account, after the runs — 2026-11-20.** Two things held only for the
author's account. Both were found by reading the code, then confirmed on an Euler login node:

- **Most of the README's commands only worked from inside the tool folder.** The wrapper passes
  `boltz2 python boltz2_run.py` through unchanged, so Python looked for the script in the current
  folder. Run from the home folder, it failed with `can't open file`.
  `boltz2/check_install.sh` existed only relative to the repository root. Every command now
  reaches its script through `$TOOL`, and the `$TOOL` forms ran from the home folder. One
  prediction was wrong: `sbatch submit_msa.sh` was expected to fail the same way and did not.
  sbatch looks a script up on PATH when it is not in the current folder. `sbatch --test-only`
  found it with the tool on PATH and reported `Unable to open file` without it. That command
  goes through `$TOOL` too, for consistency. The notebook was never affected, because it uses
  full paths.
- **`capture` returned whatever the user's shell start-up files print.** The helpers run commands
  in a login shell, which runs `~/.bash_profile` first. A profile that prints anything put that
  text in front of the parsed job id. On an empty queue, `wait_for` read the words as job states
  and would never have returned. `capture` now drops everything before a marker line it echoes
  ahead of the command. The committed and the patched helper were run through Euler's own login
  shell with a scratch home whose profile prints a greeting. The committed one parsed a polluted
  job id and read six "states" from an empty `squeue`; the patched one got the clean id and none.
  With a silent profile the two returned identical output, which also shows that Euler's system
  start-up prints nothing of its own. The patched notebook has not been re-run in JupyterHub.

## Why the defaults are what they are

Moved here from the README, which is for running the tool rather than justifying it.

- **Steering on** (`--use_potentials`). Without it Boltz-2 places mirror-image stereocentres:
  measured at 45 of 632 ligands in a production run. It costs about 2.4× the runtime. The
  wrapper therefore turns it on unless told otherwise, and "off" has to be said out loud.
- **5 diffusion samples.** Each is an independent attempt. One sample gives no spread at all, so
  you cannot tell a confident answer from a lucky one.
- **A seed is always set.** Boltz-2 has no default seed. Setting one makes a rerun comparable —
  though only on the same GPU model: the same input and seed on a different card still move the
  scores, measured at up to ~0.1 ipTM.
- **Per-prediction files on scratch.** One prediction writes about twenty files. Shared project
  storage runs out of file-count (inode) quota long before it runs out of space, and can sit
  within a percent of its hard file limit with terabytes of space to spare.

## Not yet verified

### Still untested on the cluster

- **Templates.** `plan` copies a `templates:` block through untouched and a test checks that it
  does, but no templated input has been predicted. In particular nothing has checked that a
  template path outside `/cluster` fails in a way anyone can diagnose.
- **RNA.** DNA has been predicted; RNA has not, though it takes the same form.
- **Covalent bond and contact constraints.** Only a `pocket` constraint has been run.
- **Scale.** The largest run so far is six inputs in four shards. Nothing is known about how the
  tool behaves at hundreds, which is the case the sharding exists for. The alignment step in
  particular has fetched 5 sequences and one paired group, so its batching and its pause between
  requests are untested under real load.
- **`prep`.** The table-to-YAML path is covered by tests but has not been run on Euler.
- **The notebook, by anyone else.** It has run end to end in JupyterHub (above), but only under
  the author's account, whose home and scratch already existed and whose old JupyterLab
  workspace had to be stepped around. A colleague's first attempt is the real test. It will
  also be the first run, inside JupyterHub, of the `capture` change made after the two test runs.

### Sizing

- **Two of the four memory bands are extrapolated**, and the two marked `measured` rest only on
  jobs *not failing*. SLURM never measures GPU memory — `sacct` reports host `MaxRSS` only — so a
  band that is far too generous can stay that way indefinitely without anything noticing. No job
  has been tried below its band's value.
- **The timing fit is two parameters over three shards on one GPU model**, at 147–1122 tokens and
  2 diffusion samples. Beyond that range, and on any other card, it is extrapolation.
- **Host memory per shard.** `mem_mb_per_cpu` is carried over from earlier production runs.
  Host memory grows with inputs per shard, not only with complex size, and that relationship has
  not been measured here at all.

### Carried over from before

- **Long proteins.** The environment comparison capped proteins at 600 residues, so the 800+
  residue range — where production has previously run out of host memory — is not covered.
- **The alignment step under load.** One sequence has been fetched. Nothing is known about how
  the batching and the pause between requests behave over hundreds of sequences, or when the
  server queues.
- **Whether the affinity numbers are any good.** One pair has been predicted; it was not compared
  against a measured affinity.
- **Use by anyone other than the author.** No colleague has run the tool.
