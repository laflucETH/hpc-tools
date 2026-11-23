#!/usr/bin/env python3
"""Drive a Boltz-2 run on a SLURM cluster: any complex Boltz-2 can predict.

The unit of work is one Boltz-2 input file -- a `.yaml` (preferred) or `.fasta` describing one
complex. Whatever Boltz-2 accepts, this accepts: a single protein, a homo- or heteromultimer, a
protein with ligands, DNA or RNA, covalent bonds, pocket and contact constraints, structural
templates, and affinity prediction. This driver does not model the complex for you; it takes a
directory of inputs and runs them at scale.

What it adds over calling `boltz predict` on that directory yourself:

  * File count. One prediction writes roughly twenty files (inputs, processed records, an MSA,
    one structure and one confidence record per diffusion sample, optionally an affinity record).
    Shared project storage runs out of file-count (inode) quota long before it runs out of space,
    so every per-prediction file is kept on scratch and each shard is collapsed into one table
    plus one zip as soon as it succeeds. Peak file count scales with the shards running at once,
    not with the size of the run.
  * Alignments once, not once per job. A protein's MSA is fetched once into a store keyed by the
    sequence and reused by every input that contains it, now and in later runs.
  * One protocol per run. The sampling options are fixed when the run is planned and read back by
    every array task, so scores from different shards are comparable.
  * Right-sized jobs. `size` reads the inputs and says what to ask SLURM for, per size band.

Subcommands, in the order a run uses them:

    prep      a table -> input YAMLs                      (optional; skip if you write your own)
    msa       inputs -> alignments in a shared store       (needs network; CPU only)
    size      inputs -> suggested sbatch resources, per size band
    plan      inputs -> shards on scratch; fixes the protocol
    protocol  print the boltz options the plan fixed       (used by submit_boltz2.sh)
    collect   one finished shard -> CSV + zip, then clear its scratch tree
    finalize  all shards -> one scores table, the zips, a manifest and a provenance record

Run these with the image's Python: `boltz2 python boltz2_run.py <subcommand> ...`. `msa` needs
it (it calls Boltz's own ColabFold client); `size` wants it (RDKit, to count ligand atoms).

The input format is Boltz-2's own, documented at
https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:                                     # pragma: no cover
    yaml = None     # only prep/msa/plan need it; collect, finalize and size do not, and those
                    # are the ones you may want to run with a bare python3 outside the image
                    # (`size --from-run` in particular, because sacct is not inside it)


def require_yaml() -> None:
    if yaml is None:                                    # pragma: no cover
        sys.exit("PyYAML is required for this subcommand; run it with `boltz2 python`")

# --- contract ---------------------------------------------------------------------------------
INPUT_SUFFIXES = (".yaml", ".yml", ".fasta", ".fa")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

POLYMER_KINDS = ("protein", "dna", "rna")
LIGAND_KINDS = ("ligand",)
ENTITY_KINDS = POLYMER_KINDS + LIGAND_KINDS

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWYX")
MSA_AUTO = "auto"                                       # ours: "fill this in from the store"
MSA_EMPTY = "empty"                                     # boltz's: single-sequence mode
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PREFIX = "boltz2:"
STATUS_COLUMN = PREFIX + "status"
SHARD_COLUMN = PREFIX + "shard"
GPU_COLUMN = PREFIX + "gpu"
NSAMPLES_COLUMN = PREFIX + "n_samples"
TOKENS_COLUMN = PREFIX + "n_tokens"

SHARD_FMT = "shard_%04d"
DEFAULT_PER_SHARD = 50

# Protocol defaults. Boltz-2 itself defaults to 1 diffusion sample and no seed; a seed is always
# set here so a run can be repeated, and steering is on. VERIFICATION.md records why.
DEFAULT_SAMPLES = 5
DEFAULT_SEED = 1

COLABFOLD_HOST = "https://api.colabfold.com"

# Affinity limits, from the Boltz-2 docs (prediction.md, "Properties"): one ligand chain only,
# at most 128 atoms, and the authors do not recommend much beyond 56.
AFFINITY_MAX_ATOMS = 128
AFFINITY_ADVISED_ATOMS = 56

SIZING_TABLE = Path(__file__).resolve().parent / "sizing.tsv"


# --- small helpers ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def fail(msg: str) -> None:
    sys.exit(f"FATAL: {msg}")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seq_key(seq: str) -> str:
    """Name (without extension) of a sequence's alignment in the store: the sha1 of the sequence.

    Keyed by the sequence itself, so the same protein under a different label reuses the same
    alignment, and a store survives being moved or shared.
    """
    return hashlib.sha1(seq.strip().upper().encode()).hexdigest()


def group_key(seqs: list[str]) -> str:
    """Name of a paired alignment set: the sha1 of its sequences, order-independent."""
    return hashlib.sha1("\n".join(sorted(s.strip().upper() for s in seqs)).encode()).hexdigest()


def scratch_roots() -> list[Path]:
    """Directories a run may live in, and so the only ones cleanup may ever delete inside.

    Defaults to the account's personal scratch. BOLTZ2_SCRATCH_ROOT overrides it for another site
    or for the tests.
    """
    env = os.environ.get("BOLTZ2_SCRATCH_ROOT")
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p]
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    return [Path(f"/cluster/scratch/{user}").resolve()]


def under_any(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def shard_name(i: int) -> str:
    return SHARD_FMT % i


def run_paths(run_dir: Path) -> dict[str, Path]:
    return {"plan": run_dir / "plan", "inputs": run_dir / "inputs", "work": run_dir / "work",
            "shards": run_dir / "shards"}


def read_csv_rows(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        return rows, list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # lineterminator: csv's default is CRLF, which leaves an invisible \r on the last field of
    # every row for any shell tool that reads the table
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)                                   # atomic: a reader never sees half a table


def git_commit(path: Path) -> str:
    """The commit the code at `path` is at, for the provenance record.

    Asks git when there is one. The image carries no git binary, and `plan` runs inside the
    image, so without the fallback below every run recorded its driver commit as "unknown" --
    found on Euler 2026-11-20, in the provenance of a notebook run.
    """
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + (" (dirty)" if dirty else "")
    except Exception:
        pass
    head = read_head_commit(path)
    # Without git, the commit can still be read from the repository's files, but whether the
    # working tree has uncommitted changes cannot -- so the record says it was not checked.
    return f"{head} (no git here; dirty state not checked)" if head else "unknown"


def read_head_commit(path: Path) -> str | None:
    """HEAD's commit hash, read from .git directly, or None if it cannot be found."""
    for d in [path.resolve(), *path.resolve().parents]:
        git = d / ".git"
        if git.is_file():                       # a worktree or submodule: "gitdir: <path>"
            line = git.read_text().strip()
            if line.startswith("gitdir:"):
                git = (d / line.split(":", 1)[1].strip()).resolve()
        if not git.is_dir():
            continue
        try:
            head = (git / "HEAD").read_text().strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            return head or None                 # detached HEAD holds the hash itself
        ref = head.split(":", 1)[1].strip()
        # A worktree keeps its own HEAD but shares refs with the main repository.
        common = git
        if (git / "commondir").is_file():
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in (git, common):
            loose = base / ref
            if loose.is_file():
                return loose.read_text().strip() or None
        for base in (git, common):
            packed = base / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
        return None
    return None


def load_plan_meta(run_dir: Path) -> dict:
    pj = run_paths(run_dir)["plan"] / "plan.json"
    if not pj.is_file():
        fail(f"no plan at {pj}; run `plan` first")
    return json.loads(pj.read_text())


# --- the input file ---------------------------------------------------------------------------
class Entity:
    """One chain or molecule of a complex: a `sequences:` entry of a Boltz-2 input."""

    __slots__ = ("kind", "ids", "sequence", "smiles", "ccd", "msa", "raw")

    def __init__(self, kind: str, ids: list[str], sequence: str | None = None,
                 smiles: str | None = None, ccd: str | None = None, msa: str | None = None,
                 raw: dict | None = None):
        self.kind, self.ids = kind, ids
        self.sequence, self.smiles, self.ccd, self.msa = sequence, smiles, ccd, msa
        self.raw = raw if raw is not None else {}

    @property
    def n_copies(self) -> int:
        return max(1, len(self.ids))

    @property
    def is_polymer(self) -> bool:
        return self.kind in POLYMER_KINDS


class InputSpec:
    """A parsed Boltz-2 input file."""

    __slots__ = ("path", "job_id", "fmt", "doc", "entities", "has_affinity", "template_paths")

    def __init__(self, path: Path, job_id: str, fmt: str, doc, entities: list[Entity],
                 has_affinity: bool, template_paths: list[str]):
        self.path, self.job_id, self.fmt, self.doc = path, job_id, fmt, doc
        self.entities, self.has_affinity = entities, has_affinity
        self.template_paths = template_paths

    @property
    def protein_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == "protein"]

    @property
    def distinct_protein_sequences(self) -> list[str]:
        seen, out = set(), []
        for e in self.protein_entities:
            s = (e.sequence or "").strip().upper()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


def _as_id_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def parse_yaml_input(path: Path, job_id: str) -> InputSpec:
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError("not a YAML mapping")
    seqs = doc.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        raise ValueError("no `sequences:` list")

    entities: list[Entity] = []
    for i, item in enumerate(seqs):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"sequences[{i}] is not a single ENTITY_TYPE mapping")
        kind, body = next(iter(item.items()))
        if kind not in ENTITY_KINDS:
            raise ValueError(f"sequences[{i}]: unknown entity type {kind!r} "
                             f"(expected one of {', '.join(ENTITY_KINDS)})")
        if not isinstance(body, dict):
            raise ValueError(f"sequences[{i}] ({kind}) is not a mapping")
        ids = _as_id_list(body.get("id"))
        if not ids:
            raise ValueError(f"sequences[{i}] ({kind}) has no id")
        ent = Entity(kind=kind, ids=ids,
                     sequence=(body.get("sequence") or None),
                     smiles=(body.get("smiles") or None),
                     ccd=(body.get("ccd") or None),
                     msa=(str(body["msa"]) if body.get("msa") is not None else None),
                     raw=body)
        if kind in POLYMER_KINDS and not ent.sequence:
            raise ValueError(f"sequences[{i}] ({kind}, id {','.join(ids)}) has no sequence")
        if kind == "ligand" and not (ent.smiles or ent.ccd):
            raise ValueError(f"sequences[{i}] (ligand, id {','.join(ids)}) has neither "
                             f"smiles nor ccd")
        if kind == "ligand" and ent.smiles and ent.ccd:
            raise ValueError(f"sequences[{i}] (ligand, id {','.join(ids)}) has both smiles "
                             f"and ccd; they are exclusive")
        entities.append(ent)

    props = doc.get("properties") or []
    has_affinity = any(isinstance(p, dict) and "affinity" in p for p in props)
    templates = []
    for t in (doc.get("templates") or []):
        if isinstance(t, dict):
            for k in ("cif", "pdb"):
                if t.get(k):
                    templates.append(str(t[k]))
    return InputSpec(path, job_id, "yaml", doc, entities, has_affinity, templates)


def parse_fasta_input(path: Path, job_id: str) -> InputSpec:
    """Boltz's deprecated FASTA form: headers are `>CHAIN_ID|ENTITY_TYPE[|MSA_PATH]`.

    Supported so an existing set of FASTA inputs can be run, but `prep` writes YAML and the Boltz
    docs call FASTA deprecated. Constraints, templates and affinity cannot be expressed in it.
    """
    entities: list[Entity] = []
    chain = kind = msa = None
    buf: list[str] = []

    def flush():
        if chain is None:
            return
        body = "".join(buf).strip()
        if kind in POLYMER_KINDS:
            entities.append(Entity(kind, [chain], sequence=body, msa=msa))
        elif kind == "ccd":
            entities.append(Entity("ligand", [chain], ccd=body))
        else:
            entities.append(Entity("ligand", [chain], smiles=body))

    for line in path.read_text().splitlines():
        if line.startswith(">"):
            flush()
            buf = []
            parts = [p.strip() for p in line[1:].split("|")]
            if len(parts) < 2:
                raise ValueError(f"FASTA header {line!r} is not CHAIN_ID|ENTITY_TYPE[|MSA_PATH]")
            chain, kind = parts[0], parts[1].lower()
            if kind not in ENTITY_KINDS + ("ccd", "smiles"):
                raise ValueError(f"FASTA header {line!r}: unknown entity type {kind!r}")
            msa = parts[2] if len(parts) > 2 and parts[2] else None
        else:
            buf.append(line.strip())
    flush()
    if not entities:
        raise ValueError("no records")
    return InputSpec(path, job_id, "fasta", None, entities, False, [])


def parse_input(path: Path) -> InputSpec:
    job_id = path.stem
    if not JOB_ID_RE.match(job_id):
        raise ValueError(f"file name {path.name!r} is not usable as a job id "
                         f"(must match [A-Za-z0-9][A-Za-z0-9._-]*)")
    if path.suffix.lower() in (".yaml", ".yml"):
        return parse_yaml_input(path, job_id)
    return parse_fasta_input(path, job_id)


def list_inputs(inputs_dir: Path) -> list[Path]:
    if not inputs_dir.is_dir():
        fail(f"no such input directory: {inputs_dir}")
    files = sorted(p for p in inputs_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)
    if not files:
        fail(f"{inputs_dir} holds no {' or '.join(INPUT_SUFFIXES)} file")
    return files


def load_all_inputs(inputs_dir: Path) -> tuple[list[InputSpec], list[dict]]:
    """Every input that parses, plus a row per input that does not."""
    specs, bad, seen = [], [], set()
    for f in list_inputs(inputs_dir):
        try:
            spec = parse_input(f)
        except Exception as e:
            bad.append({"job_id": f.stem, "file": f.name,
                        "reason": f"{type(e).__name__}: {e}"})
            continue
        if spec.job_id in seen:
            bad.append({"job_id": spec.job_id, "file": f.name,
                        "reason": "duplicate job id (two inputs share a file stem)"})
            continue
        seen.add(spec.job_id)
        specs.append(spec)
    return specs, bad


# --- token counting, for sizing ---------------------------------------------------------------
_MOL_CACHE: dict[str, int] = {}


def heavy_atoms(entity: Entity) -> tuple[int, str]:
    """(heavy-atom count of a ligand, how it was obtained).

    Boltz tokenises a non-polymer as one token per atom, so this is what a ligand contributes to
    the token count that drives memory and runtime.
    """
    if entity.smiles:
        key = "smiles:" + entity.smiles
        if key in _MOL_CACHE:
            return _MOL_CACHE[key], "rdkit"
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromSmiles(entity.smiles)
            if mol is not None:
                n = mol.GetNumHeavyAtoms()
                _MOL_CACHE[key] = n
                return n, "rdkit"
        except ImportError:
            pass
        # No RDKit (running outside the image): count element-ish characters. Crude, and the
        # caller is told so, because a wrong count here only ever changes a size estimate.
        n = len(re.findall(r"[A-Z][a-z]?", re.sub(r"\[[^\]]*\]", "X", entity.smiles)))
        return n, "approx"
    if entity.ccd:
        code = entity.ccd.strip().upper()
        if code in _MOL_CACHE:
            return _MOL_CACHE[code], "ccd"
        cache = os.environ.get("BOLTZ_CACHE", "/opt/boltz/cache")
        pkl = Path(cache) / "mols" / f"{code}.pkl"
        if pkl.is_file():
            try:
                import pickle
                mol = pickle.loads(pkl.read_bytes())
                n = mol.GetNumHeavyAtoms() if hasattr(mol, "GetNumHeavyAtoms") else mol.GetNumAtoms()
                _MOL_CACHE[code] = n
                return n, "ccd"
            except Exception:
                pass
        return 25, "guess"                              # a mid-sized cofactor; flagged as a guess
    return 0, "none"


def protein_signature(spec: InputSpec) -> str:
    """A stable key for which proteins an input contains.

    Used only to order inputs within a size band, so that inputs sharing a protein end up in the
    same shard. That is worth a little: they then share an alignment, which Boltz loads once.
    """
    joined = "|".join(sorted(spec.distinct_protein_sequences))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def token_count(spec: InputSpec) -> tuple[int, str]:
    """(Boltz tokens in this complex, the weakest source used for the count).

    One token per polymer residue, one per ligand heavy atom, times the number of copies. Memory
    grows with the square of this number, because the model holds a token-by-token pair
    representation; runtime grows with it too.
    """
    total, quality = 0, "exact"
    for e in spec.entities:
        if e.is_polymer:
            total += len((e.sequence or "").strip()) * e.n_copies
        else:
            n, how = heavy_atoms(e)
            total += n * e.n_copies
            if how in ("approx", "guess"):
                quality = how
    return total, quality


# --- sizing -----------------------------------------------------------------------------------
# Fallbacks if sizing.tsv is missing or lacks a constant. They are the measured values, repeated
# here only so the tool still runs without its table; sizing.tsv is the authority and carries the
# provenance.
TIMING_DEFAULTS = {"startup_s": 120.0, "seconds_per_token_sample": 0.067,
                   "affinity_extra_s_per_input": 59.0, "steering_factor": 2.4}


def load_sizing_table(path: Path = SIZING_TABLE) -> list[dict]:
    if not path.is_file():
        return []
    rows, _ = read_csv_rows_tsv(path)
    out = []
    for r in rows:
        try:
            out.append({"max_tokens": int(r["max_tokens"]), "gpumem_gb": int(r["gpumem_gb"]),
                        "mem_mb_per_cpu": int(r["mem_mb_per_cpu"]),
                        "basis": r.get("basis", ""), "source": r.get("source", "")})
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda r: r["max_tokens"])


def load_timing(path: Path = SIZING_TABLE) -> dict:
    """The `# @name value` constants at the top of sizing.tsv."""
    out = dict(TIMING_DEFAULTS)
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*#\s*@(\w+)\s+([0-9.]+)\s*$", line)
        if m and m.group(1) in out:
            out[m.group(1)] = float(m.group(2))
    return out


def shard_seconds(tokens: int, samples: int, n_inputs: int, affinity: bool, steering: bool,
                  t: dict) -> float:
    """Predicted walltime of one shard.

    Fixed cost plus a cost per token per diffusion sample, which is what the measurements
    actually look like: at these sizes the fixed cost (container start, loading the checkpoint,
    processing the alignments) dominates, so a model that scales only with the number of inputs
    gets a one-input shard badly wrong. See the header of sizing.tsv.
    """
    secs = t["startup_s"] + t["seconds_per_token_sample"] * tokens * samples
    if not steering:                                    # the rate was measured with steering on
        secs = t["startup_s"] + (secs - t["startup_s"]) / max(1e-9, t["steering_factor"])
    if affinity:
        secs += t["affinity_extra_s_per_input"] * n_inputs
    return secs


def read_csv_rows_tsv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    rows = [dict(r) for r in reader]
    return rows, list(reader.fieldnames or [])


def band_for(tokens: int, table: list[dict]) -> dict | None:
    for row in table:
        if tokens <= row["max_tokens"]:
            return row
    return table[-1] if table else None


def band_plan(sized: list[tuple[str, int, str, bool]], per_shard: int,
              table: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Assign every input to a shard, grouping by size band and by whether it wants affinity.

    `sized` is (job id, token count, tiebreak, wants affinity). Returns (job id -> shard name,
    one record per group giving its array range).

    No shard ever straddles a size band, for two reasons. A shard is one SLURM task, so it gets
    one memory request and one walltime, and those have to fit its largest member; and the array
    range printed for a band has to name exactly the shards that band's resources apply to.

    No shard ever mixes inputs that ask for affinity with inputs that do not, for a different
    and much less obvious reason: **Boltz-2 2.2.1 runs its affinity stage over every input in
    the call**, not only those whose YAML requested it. For an input that did not request it,
    the `pre_affinity_<id>.npz` it then goes looking for was never written, and the whole call
    dies with FileNotFoundError -- after the structures have been predicted, so the GPU time is
    spent and lost. Observed on Euler job 21396655_1, 2026-11-18.

    `size` and `plan` both call this. They used to work it out separately, and then disagreed
    whenever a band boundary fell inside a shard -- so the sbatch lines `size` printed could
    carry one band's resources to another band's shards. Keeping it in one function is what
    stops that happening again.

    Within a group, inputs are ordered by token count and then by a caller-supplied tiebreak
    (the protein content), so inputs sharing a protein land together and the assignment is the
    same every time it is computed.
    """
    groups: dict[tuple[int, bool], list[tuple[str, int, str, bool]]] = {}
    for item in sized:
        row = band_for(item[1], table)
        groups.setdefault((row["max_tokens"] if row else 0, bool(item[3])), []).append(item)

    assigned: dict[str, str] = {}
    records: list[dict] = []
    next_shard = 0
    for key in sorted(groups):
        max_tokens, affinity = key
        members = sorted(groups[key], key=lambda x: (x[1], x[2], x[0]))
        n_shards = (len(members) + per_shard - 1) // per_shard
        lo = next_shard
        for i, item in enumerate(members):
            assigned[item[0]] = shard_name(lo + i // per_shard)
        next_shard = lo + n_shards
        row = band_for(members[-1][1], table) or {}
        records.append({"max_tokens": max_tokens, "affinity": affinity,
                        "n_inputs": len(members), "n_shards": n_shards,
                        "array_from": lo, "array_to": next_shard - 1,
                        "largest": members[-1][1], "band": row})
    return assigned, records


def hhmmss(seconds: float) -> str:
    seconds = int(max(600, seconds))                    # never ask for less than 10 minutes
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def cmd_size(args: argparse.Namespace) -> int:
    inputs_dir = Path(args.inputs).expanduser().resolve()
    table = load_sizing_table()
    if not table:
        fail(f"no sizing table at {SIZING_TABLE}")

    specs, bad = load_all_inputs(inputs_dir)
    if bad:
        log(f"{len(bad)} input(s) could not be parsed and are not sized:")
        for b in bad[:5]:
            log(f"  {b['file']}: {b['reason']}")

    sized = []
    worst_quality = "exact"
    for s in specs:
        n, q = token_count(s)
        if q != "exact":
            worst_quality = q
        sized.append((s.job_id, n, protein_signature(s), s.has_affinity))
    if not sized:
        fail("no input could be sized")
    counts = sorted(n for _, n, _, _ in sized)
    log(f"{len(sized)} input(s) in {inputs_dir}")
    log(f"  tokens per input: min {counts[0]}, median {counts[len(counts) // 2]}, "
        f"max {counts[-1]}  (1 token = 1 residue, or 1 ligand heavy atom)")
    if worst_quality != "exact":
        log(f"  NOTE: some ligand atom counts are {worst_quality}, not exact "
            f"(run this through `boltz2 python` so RDKit is available)")

    # Group into the bands of the sizing table, so a run of mostly small complexes is not
    # charged the memory its three biggest need. `plan` assigns shards with this same function,
    # so these array ranges name exactly the shards it will write.
    per_shard = max(1, int(args.per_shard))
    samples = max(1, int(args.samples))
    timing = load_timing()
    assigned, records = band_plan(sized, per_shard, table)

    # Walltime is worked out for the heaviest shard a band actually contains, not for a
    # hypothetical full one, because band_plan has already decided exactly which inputs go
    # where. A band whose last shard holds one small input is not charged for a full shard.
    tokens_by_job = {j: n for j, n, _, _ in sized}
    shard_tokens: dict[str, int] = {}
    shard_inputs: dict[str, int] = {}
    for job_id, shard in assigned.items():
        shard_tokens[shard] = shard_tokens.get(shard, 0) + tokens_by_job[job_id]
        shard_inputs[shard] = shard_inputs.get(shard, 0) + 1

    print()
    print(f"Suggested jobs, at {samples} diffusion sample(s) per input, steering "
          f"{'off' if args.no_potentials else 'on'}, {per_shard} input(s) per shard.")
    print(f"Plan the run with the SAME --per-shard {per_shard}, or these ranges will not match.")
    print()
    plan_lines = []
    for rec in records:
        row = rec["band"]
        lo, hi = rec["array_from"], rec["array_to"]
        names = [shard_name(i) for i in range(lo, hi + 1)]
        secs = max(shard_seconds(shard_tokens[n], samples, shard_inputs[n], rec["affinity"],
                                 not args.no_potentials, timing) for n in names)
        secs *= args.safety
        tag = "  [affinity]" if rec["affinity"] else ""
        print(f"  tokens <= {rec['max_tokens']:>5}   {rec['n_inputs']:>5} input(s)  "
              f"(largest {rec['largest']}){tag}  ->  {rec['n_shards']} shard(s), array {lo}-{hi}")
        print(f"      sbatch --array={lo}-{hi} --gres=gpumem:{row['gpumem_gb']}g "
              f"--time={hhmmss(secs)} --mem-per-cpu={row['mem_mb_per_cpu']} "
              f"submit_boltz2.sh <run_dir>")
        print(f"      memory basis: {row['basis']}")
        plan_lines.append({"max_tokens": rec["max_tokens"], "affinity": rec["affinity"],
                           "n_inputs": rec["n_inputs"],
                           "n_shards": rec["n_shards"], "array_from": lo, "array_to": hi,
                           "gpumem_gb": row["gpumem_gb"], "time": hhmmss(secs),
                           "mem_mb_per_cpu": row["mem_mb_per_cpu"], "basis": row["basis"]})
        print()

    print("These are ESTIMATES.")
    print(f"  Memory: 'basis' says whether that band was measured here or extrapolated.")
    print(f"  Time:   {timing['startup_s']:.0f} s fixed + "
          f"{timing['seconds_per_token_sample']:.4f} s per token per sample"
          + (f" + {timing['affinity_extra_s_per_input']:.0f} s per affinity input" if
             any(r['affinity'] for r in records) else "")
          + f", x{args.safety:g} safety.")
    print("          Fitted on 3 shards on one RTX 4090; runtime varies several-fold by card.")
    print("Before a large run of a shape nobody has run here, predict one shard first and read")
    print("the real usage back with:   python3 boltz2_run.py size --from-run <run_dir>")
    if args.json:
        Path(args.json).write_text(json.dumps(plan_lines, indent=2) + "\n")
        log(f"wrote {args.json}")
    return 0


def cmd_size_from_run(args: argparse.Namespace) -> int:
    """What a finished run actually used, from its receipts and sacct."""
    run_dir = Path(args.from_run).expanduser().resolve()
    paths = run_paths(run_dir)
    meta = load_plan_meta(run_dir)
    receipts = sorted(paths["shards"].glob("shard_*.job.json"))
    if not receipts:
        fail(f"no shard receipts under {paths['shards']}; has anything run?")

    plan_rows, _ = read_csv_rows(paths["plan"] / "jobs.csv")
    tokens_by_shard: dict[str, list[int]] = {}
    for r in plan_rows:
        s = r.get(SHARD_COLUMN)
        if s and r.get(TOKENS_COLUMN):
            try:
                tokens_by_shard.setdefault(s, []).append(int(r[TOKENS_COLUMN]))
            except ValueError:
                pass

    rows = []
    job_ids = []
    for rf in receipts:
        j = json.loads(rf.read_text())
        shard = j.get("shard", rf.stem.split(".")[0])
        toks = tokens_by_shard.get(shard, [])
        rows.append({"shard": shard, "job": j.get("array_job_id", "") + "_" + str(j.get("task_id", "")),
                     "gpu": j.get("gpu", ""), "n_inputs": j.get("n_inputs", ""),
                     "max_tokens": max(toks) if toks else "",
                     "elapsed_s": j.get("elapsed_s", ""), "state": "", "maxrss_mb": ""})
        if j.get("array_job_id") and j.get("array_job_id") != "none":
            job_ids.append(f"{j['array_job_id']}_{j['task_id']}")

    # sacct, if it is here. MaxRSS is host memory, not GPU memory: SLURM does not record GPU
    # memory, so the gpumem band can only ever be confirmed by a job NOT failing at it.
    if job_ids and shutil.which("sacct"):
        try:
            out = subprocess.run(["sacct", "-n", "-P", "-j", ",".join(sorted(set(job_ids))),
                                  "--format=JobID,State,MaxRSS,Elapsed"],
                                 capture_output=True, text=True, timeout=60).stdout
            by_job: dict[str, dict] = {}
            for line in out.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                jid = parts[0].split(".")[0]
                rec = by_job.setdefault(jid, {"state": parts[1], "maxrss": "", "elapsed": parts[3]})
                if parts[2]:
                    rec["maxrss"] = parts[2]
                if parts[1] and not parts[1].startswith("  "):
                    rec["state"] = parts[1]
            for r in rows:
                rec = by_job.get(r["job"])
                if rec:
                    r["state"] = rec["state"]
                    mr = rec["maxrss"].rstrip("KMGT")
                    unit = rec["maxrss"][-1:] if rec["maxrss"] else ""
                    try:
                        v = float(mr)
                        r["maxrss_mb"] = round(v / 1024 if unit == "K" else
                                               v if unit == "M" else v * 1024, 1)
                    except ValueError:
                        pass
        except Exception as e:
            log(f"sacct lookup failed ({type(e).__name__}); reporting receipts only")
    else:
        log("sacct not available here; reporting what the receipts recorded")

    p = meta["protocol"]
    samples = p.get("diffusion_samples", 1)
    print(f"{'shard':<12} {'gpu':<26} {'inputs':>6} {'max_tok':>8} {'elapsed_s':>10} "
          f"{'s/sample/input':>15} {'maxRSS_MB':>10} {'state':<12}")
    for r in sorted(rows, key=lambda r: r["shard"]):
        per = ""
        try:
            per = f"{float(r['elapsed_s']) / (samples * int(r['n_inputs'])):.1f}"
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        print(f"{r['shard']:<12} {str(r['gpu'])[:26]:<26} {str(r['n_inputs']):>6} "
              f"{str(r['max_tokens']):>8} {str(r['elapsed_s']):>10} {per:>15} "
              f"{str(r['maxrss_mb']):>10} {str(r['state'])[:12]:<12}")
    out_csv = run_dir / "plan" / "measured_sizing.csv"
    write_csv_rows(out_csv, rows, ["shard", "job", "gpu", "n_inputs", "max_tokens",
                                   "elapsed_s", "maxrss_mb", "state"])
    print()
    print(f"written to {out_csv}")
    print("To make this the default for a band, add a row to sizing.tsv with its provenance.")
    print("Note: SLURM records host memory (MaxRSS), never GPU memory. A gpumem band is confirmed")
    print("only by jobs not failing at it, so raise a band only after seeing a real CUDA OOM.")
    return 0


# --- prep -------------------------------------------------------------------------------------
PREP_REQUIRED = ("id", "entity", "sequence")


def cmd_prep(args: argparse.Namespace) -> int:
    """A long-format table -> one Boltz-2 YAML per id. One row per chain.

    Columns: id, entity (protein|dna|rna|ligand|ccd), sequence (the residues, the SMILES, or the
    CCD code), and optionally chain, msa, copies, affinity, cyclic.

    This is a convenience for the common cases. Anything it cannot express -- constraints,
    templates, modified residues -- is written as YAML by hand; see the Boltz-2 input docs.
    """
    require_yaml()
    table = Path(args.table).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    rows, fields = read_csv_rows(table)
    missing = [c for c in PREP_REQUIRED if c not in fields]
    if missing:
        fail(f"table is missing column(s): {', '.join(missing)}")
    if not rows:
        fail("table has no rows")

    by_id: dict[str, list[dict]] = {}
    for r in rows:
        jid = (r.get("id") or "").strip()
        if not jid:
            fail("a row has an empty id")
        by_id.setdefault(jid, []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, []
    for jid, members in by_id.items():
        if not JOB_ID_RE.match(jid):
            skipped.append((jid, "id is not usable as a file name"))
            continue
        target = out_dir / f"{jid}.yaml"
        if target.exists() and not args.override:
            skipped.append((jid, "already exists (pass --override to replace)"))
            continue
        sequences, used_chains, binder, problem = [], [], None, None
        auto = iter(CHAIN_ALPHABET)
        for r in members:
            kind = (r.get("entity") or "").strip().lower()
            body_val = (r.get("sequence") or "").strip()
            if not body_val:
                problem = "a row has an empty sequence"
                break
            copies = int((r.get("copies") or "1").strip() or 1)
            ids = []
            explicit = (r.get("chain") or "").strip()
            if explicit:
                ids = [c.strip() for c in explicit.split(";") if c.strip()]
            while len(ids) < copies:
                nxt = next(auto, None)
                if nxt is None:
                    problem = "more chains than the alphabet has letters"
                    break
                if nxt not in used_chains:
                    ids.append(nxt)
            if problem:
                break
            used_chains += ids
            entry_id = ids[0] if len(ids) == 1 else ids
            if kind in POLYMER_KINDS:
                body: dict = {"id": entry_id, "sequence": body_val.upper()}
                if kind == "protein":
                    body["msa"] = (r.get("msa") or MSA_AUTO).strip() or MSA_AUTO
                if (r.get("cyclic") or "").strip().lower() in ("1", "true", "yes"):
                    body["cyclic"] = True
                sequences.append({kind: body})
            elif kind in ("ligand", "smiles"):
                sequences.append({"ligand": {"id": entry_id, "smiles": body_val}})
            elif kind == "ccd":
                sequences.append({"ligand": {"id": entry_id, "ccd": body_val.upper()}})
            else:
                problem = f"unknown entity {kind!r}"
                break
            if (r.get("affinity") or "").strip().lower() in ("1", "true", "yes"):
                if kind not in ("ligand", "smiles", "ccd"):
                    problem = "affinity was asked for on a row that is not a ligand"
                    break
                if binder is not None:
                    problem = "two rows ask for affinity; Boltz-2 allows one binder"
                    break
                binder = ids[0]
        if problem:
            skipped.append((jid, problem))
            continue
        doc: dict = {"version": 1, "sequences": sequences}
        if binder:
            doc["properties"] = [{"affinity": {"binder": binder}}]
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        written += 1

    log(f"{written} input(s) written to {out_dir}")
    for jid, why in skipped:
        log(f"  skipped {jid}: {why}")
    return 0 if not skipped else 1


# --- msa --------------------------------------------------------------------------------------
def store_lookup(store: Path | None, seq: str) -> Path | None:
    if store is None:
        return None
    for ext in (".csv", ".a3m"):
        p = store / (seq_key(seq) + ext)
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def paired_lookup(store: Path | None, seqs: list[str]) -> dict[str, Path] | None:
    """The paired alignment set for this exact group of sequences, if the store holds it."""
    if store is None or len(seqs) < 2:
        return None
    d = store / "paired" / group_key(seqs)
    if not d.is_dir():
        return None
    hits = {}
    for s in seqs:
        p = d / f"{seq_key(s)}.csv"
        if not (p.is_file() and p.stat().st_size > 0):
            return None
        hits[s] = p
    return hits


def a3m_to_boltz_csv(a3m: str, max_seqs: int) -> str:
    """One unpaired ColabFold a3m -> Boltz's MSA csv.

    Follows boltz.main.compute_msa for the single-chain case: sequences only (headers dropped),
    capped at max_seqs, every key -1. In Boltz's parser (data/parse/csv.py) the key is a taxonomy
    id, and -1 means the row is not paired across chains.
    """
    seqs = a3m.strip().splitlines()[1::2][:max_seqs]
    return "\n".join(["key,sequence"] + [f"-1,{s}" for s in seqs])


def combine_paired_unpaired(paired_a3m: str, unpaired_a3m: str, max_paired: int,
                            max_seqs: int) -> str:
    """Boltz's own combination of a paired and an unpaired alignment, for one chain of a complex.

    Transcribed from boltz.main.compute_msa (v2.2.1) so the result is what `--use_msa_server`
    would have written: paired rows first, keyed by their row index so the same index means the
    same organism in every chain of the complex; all-gap paired rows dropped; then unpaired rows
    keyed -1, with the query dropped because the paired block already carries it.
    """
    paired = paired_a3m.strip().splitlines()[1::2][:max_paired]
    keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
    paired = [s for s in paired if s != "-" * len(s)]
    unpaired = unpaired_a3m.strip().splitlines()[1::2][: max_seqs - len(paired)]
    if paired:
        unpaired = unpaired[1:]
    seqs = paired + unpaired
    keys = keys + [-1] * len(unpaired)
    return "\n".join(["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, seqs)])


def cmd_msa(args: argparse.Namespace) -> int:
    require_yaml()
    store = Path(args.msa_store).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    inputs_dir = Path(args.inputs).expanduser().resolve()
    specs, bad = load_all_inputs(inputs_dir)
    for b in bad:
        log(f"  unparseable, no alignment fetched: {b['file']}: {b['reason']}")

    # Two kinds of work. A target with one distinct protein chain needs only an unpaired
    # alignment of that sequence, which every other input containing it can reuse. A target with
    # several distinct protein chains also needs a PAIRED alignment, which only means anything
    # for that exact group of sequences -- so it is fetched and cached per group, not per
    # sequence. Boltz does the same thing in compute_msa; see combine_paired_unpaired.
    singles: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for s in specs:
        seqs = [q for q in s.distinct_protein_sequences if not set(q) - STANDARD_AA]
        wanted = [e for e in s.protein_entities
                  if (e.msa or MSA_AUTO).strip().lower() in (MSA_AUTO, "")]
        if not wanted or not seqs:
            continue
        if len(seqs) == 1:
            if store_lookup(store, seqs[0]) is None:
                singles[seq_key(seqs[0])] = seqs[0]
        else:
            if paired_lookup(store, seqs) is None:
                groups[group_key(seqs)] = seqs
            for q in seqs:                              # the unpaired half is reusable on its own
                if store_lookup(store, q) is None:
                    singles[seq_key(q)] = q

    log(f"{len(singles)} sequence(s) and {len(groups)} paired group(s) missing from {store}")
    if not singles and not groups:
        return 0
    if args.dry_run:
        return 0

    from boltz.data import const
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    import fcntl
    lock = open(store / ".fetch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail(f"another fetch holds {store}/.fetch.lock; two fetchers would query the server twice")

    n_ok = 0
    keys = sorted(singles)
    batch = max(1, args.batch)
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        tmp = store / f".tmp_{os.getpid()}_{i}"
        t0 = time.time()
        try:
            a3ms = run_mmseqs2([singles[k] for k in chunk], str(tmp), use_env=True,
                               use_pairing=False, host_url=args.host)
        except Exception as e:
            log(f"  batch {i // batch}: {type(e).__name__}: {e}")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        shutil.rmtree(tmp, ignore_errors=True)
        for k, a3m in zip(chunk, a3ms):
            if not a3m.strip():
                log(f"  {k}: empty alignment returned; not stored")
                continue
            dest = store / f"{k}.csv"
            part = dest.with_suffix(".csv.part")
            part.write_text(a3m_to_boltz_csv(a3m, const.max_msa_seqs) + "\n")
            part.replace(dest)
            n_ok += 1
        log(f"  batch {i // batch}: {len(chunk)} sequence(s) in {time.time() - t0:.0f} s")
        if i + batch < len(keys):
            time.sleep(args.pause)

    n_groups = 0
    for gkey, seqs in sorted(groups.items()):
        d = store / "paired" / gkey
        tmp = store / f".tmpg_{os.getpid()}_{gkey[:8]}"
        t0 = time.time()
        try:
            paired = run_mmseqs2(seqs, str(tmp) + "_p", use_env=True, use_pairing=True,
                                 host_url=args.host, pairing_strategy=args.pairing_strategy)
            unpaired = run_mmseqs2(seqs, str(tmp) + "_u", use_env=True, use_pairing=False,
                                   host_url=args.host)
        except Exception as e:
            log(f"  group {gkey[:8]}: {type(e).__name__}: {e}")
            for suffix in ("_p", "_u"):
                shutil.rmtree(str(tmp) + suffix, ignore_errors=True)
            continue
        for suffix in ("_p", "_u"):
            shutil.rmtree(str(tmp) + suffix, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        for seq, p_a3m, u_a3m in zip(seqs, paired, unpaired):
            csv_str = combine_paired_unpaired(p_a3m, u_a3m, const.max_paired_seqs,
                                              const.max_msa_seqs)
            dest = d / f"{seq_key(seq)}.csv"
            part = dest.with_suffix(".csv.part")
            part.write_text(csv_str + "\n")
            part.replace(dest)
        (d / "members.txt").write_text("\n".join(seq_key(s) for s in seqs) + "\n")
        n_groups += 1
        log(f"  group {gkey[:8]}: {len(seqs)} chain(s) paired in {time.time() - t0:.0f} s")
        time.sleep(args.pause)

    log(f"stored {n_ok} of {len(singles)} alignment(s) and {n_groups} of {len(groups)} group(s)")
    return 0 if (n_ok == len(singles) and n_groups == len(groups)) else 1


# --- plan -------------------------------------------------------------------------------------
def resolve_msas(spec: InputSpec, store: Path | None,
                 allow_missing: bool) -> tuple[dict[str, str] | None, str, list[dict]]:
    """Decide the `msa:` value of every protein chain of one input.

    Returns (map from chain id to msa value, how it was resolved, records of the files used),
    or (None, reason, []) if the input cannot be planned.
    """
    proteins = spec.protein_entities
    if not proteins:
        return {}, "no protein chain", []
    need = [e for e in proteins if (e.msa or MSA_AUTO).strip().lower() in (MSA_AUTO, "")]
    out: dict[str, str] = {}
    used: list[dict] = []
    for e in proteins:
        given = (e.msa or "").strip()
        if given and given.lower() not in (MSA_AUTO, ""):
            if given.lower() == MSA_EMPTY:
                for cid in e.ids:
                    out[cid] = MSA_EMPTY
                continue
            p = Path(given).expanduser()
            if not p.is_file():
                return None, f"msa not found: {given}", []
            for cid in e.ids:
                out[cid] = str(p.resolve())
            used.append({"msa": str(p.resolve()), "how": "given in the input",
                         "sequence_sha1": seq_key(e.sequence or ""), "md5": md5_file(p)})
    if not need:
        return out, "all given in the input", used

    seqs = spec.distinct_protein_sequences
    group = paired_lookup(store, seqs) if len(seqs) > 1 else None
    how = "store (paired)" if group else "store"
    for e in need:
        s = (e.sequence or "").strip().upper()
        hit = group.get(s) if group else store_lookup(store, s)
        if hit is None:
            if not allow_missing:
                return None, "no alignment in the store", []
            for cid in e.ids:
                out[cid] = MSA_EMPTY
            how = "missing -> empty"
            continue
        for cid in e.ids:
            out[cid] = str(hit)
        used.append({"msa": str(hit), "how": how, "sequence_sha1": seq_key(s),
                     "md5": md5_file(hit)})
    return out, how, used


def apply_msas(spec: InputSpec, msa_by_chain: dict[str, str]) -> dict:
    """A copy of the input's YAML with every protein chain's `msa:` filled in.

    Only the `msa:` fields are touched. Constraints, templates, modifications and anything else
    the input carries are copied through untouched, so this never silently drops a field it does
    not know about.
    """
    doc = copy.deepcopy(spec.doc)
    for item in doc.get("sequences", []):
        kind, body = next(iter(item.items()))
        if kind != "protein":
            continue
        ids = _as_id_list(body.get("id"))
        values = {msa_by_chain.get(cid) for cid in ids}
        values.discard(None)
        if len(values) == 1:
            body["msa"] = values.pop()
    doc.setdefault("version", 1)
    return doc


def check_affinity(spec: InputSpec) -> str | None:
    """Whether this input's affinity request is one Boltz-2 can honour. None if it is."""
    if not spec.has_affinity:
        return None
    binders = []
    for p in (spec.doc.get("properties") or []):
        if isinstance(p, dict) and isinstance(p.get("affinity"), dict):
            binders.append(str(p["affinity"].get("binder", "")))
    if len(binders) != 1 or not binders[0]:
        return "affinity needs exactly one binder chain"
    binder = binders[0]
    ent = next((e for e in spec.entities if binder in e.ids), None)
    if ent is None:
        return f"affinity binder {binder!r} is not a chain of this input"
    if ent.kind != "ligand":
        return f"affinity binder {binder!r} is a {ent.kind}; Boltz-2 only scores a ligand"
    n, _ = heavy_atoms(ent)
    if n > AFFINITY_MAX_ATOMS:
        return f"affinity binder has ~{n} heavy atoms, over Boltz-2's limit of {AFFINITY_MAX_ATOMS}"
    return None


def affinity_advisory(spec: InputSpec) -> str | None:
    """A warning, not a refusal: an affinity binder larger than Boltz-2 was trained on.

    The Boltz-2 docs set a hard limit of 128 atoms but say results are not recommended much
    beyond 56, which is where their training set stopped. Between the two the model will answer,
    and the answer is outside what it has seen.
    """
    if not spec.has_affinity:
        return None
    for p in (spec.doc.get("properties") or []):
        if isinstance(p, dict) and isinstance(p.get("affinity"), dict):
            binder = str(p["affinity"].get("binder", ""))
            ent = next((e for e in spec.entities if binder in e.ids), None)
            if ent is not None and ent.kind == "ligand":
                n, _ = heavy_atoms(ent)
                if n > AFFINITY_ADVISED_ATOMS:
                    return (f"affinity binder {binder} has ~{n} heavy atoms; Boltz-2 trained up "
                            f"to about {AFFINITY_ADVISED_ATOMS} and its authors do not recommend "
                            f"much beyond that, so treat the affinity value with caution")
    return None


def cmd_plan(args: argparse.Namespace) -> int:
    require_yaml()
    run_dir = Path(args.run_dir).expanduser().resolve()
    inputs_dir = Path(args.inputs).expanduser().resolve()
    store = Path(args.msa_store).expanduser().resolve() if args.msa_store else None
    if store is not None and not store.is_dir():
        fail(f"no such MSA store: {store}")
    if not args.allow_any_path and not under_any(run_dir, scratch_roots()):
        fail(f"run directory is outside the scratch root(s) {[str(r) for r in scratch_roots()]}: "
             f"{run_dir}\n       per-prediction files must not be written to project storage; "
             f"pass --allow-any-path only for a test")

    specs, bad = load_all_inputs(inputs_dir)
    paths = run_paths(run_dir)
    if paths["plan"].exists() and not args.override:
        fail(f"a plan already exists at {paths['plan']}; pass --override to replace it")
    if args.override and paths["shards"].is_dir() and any(paths["shards"].iterdir()):
        fail(f"{paths['shards']} already holds collected shards; plan a new run directory "
             f"instead of re-planning this one")
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Resolve everything before writing anything, so a bad input set is rejected whole rather
    # than half-planned.
    rows: list[dict] = []
    ready: list[tuple[InputSpec, dict, int]] = []
    msa_used: dict[str, dict] = {}
    advisories: list[str] = []
    n_no_msa = 0
    for b in bad:
        rows.append({"job_id": b["job_id"], "input_file": b["file"],
                     STATUS_COLUMN: f"skipped: {b['reason']}"})
    for spec in specs:
        row = {"job_id": spec.job_id, "input_file": spec.path.name,
               "input_md5": md5_file(spec.path)}
        n_tokens, quality = token_count(spec)
        row[TOKENS_COLUMN] = n_tokens
        row[PREFIX + "n_chains"] = sum(e.n_copies for e in spec.entities)
        row[PREFIX + "entities"] = ",".join(sorted({e.kind for e in spec.entities}))
        reason = None
        if spec.fmt == "fasta":
            reason = "FASTA inputs are passed through unchanged; convert to YAML to use the store"
        aff = check_affinity(spec) if spec.fmt == "yaml" else None
        if aff:
            reason = aff
        msa_map: dict[str, str] = {}
        if reason is None:
            msa_map, how, used = resolve_msas(spec, store, args.allow_missing_msa)
            if msa_map is None:
                reason = how
                n_no_msa += how == "no alignment in the store"
            else:
                row[PREFIX + "msa"] = how
                for u in used:
                    msa_used[u["msa"]] = u
        row[STATUS_COLUMN] = "planned" if reason is None else f"skipped: {reason}"
        rows.append(row)
        if reason is None:
            ready.append((spec, msa_map, n_tokens))
            if spec.fmt == "yaml":
                note = affinity_advisory(spec)
                if note:
                    advisories.append(f"{spec.job_id}: {note}")
    if n_no_msa and not args.allow_missing_msa:
        fail(f"{n_no_msa} input(s) have no alignment in {store}. Fetch them first "
             f"(`boltz2_run.py msa`), or pass --allow-missing-msa to fold them in "
             f"single-sequence mode, which is markedly less accurate")

    # Shard by size band, with the same function `size` uses, so the sbatch lines `size` prints
    # name exactly these shards. A shard never mixes bands, because it is one SLURM task with
    # one memory request and one walltime.
    per_shard = max(1, int(args.per_shard))
    table = load_sizing_table()
    assigned, band_records = band_plan(
        [(spec.job_id, n, protein_signature(spec), spec.has_affinity) for spec, _, n in ready],
        per_shard, table)

    written = 0
    for spec, msa_map, _ in ready:
        shard = assigned[spec.job_id]
        shard_dir = paths["inputs"] / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        if spec.fmt == "fasta":
            shutil.copy2(spec.path, shard_dir / spec.path.name)
        else:
            doc = apply_msas(spec, msa_map)
            with open(shard_dir / f"{spec.job_id}.yaml", "w", encoding="utf-8") as fh:
                yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False,
                               allow_unicode=True)
        written += 1
    for row in rows:
        if row["job_id"] in assigned:
            row[SHARD_COLUMN] = assigned[row["job_id"]]

    field_order = ["job_id", "input_file", "input_md5", TOKENS_COLUMN, PREFIX + "n_chains",
                   PREFIX + "entities", PREFIX + "msa", STATUS_COLUMN, SHARD_COLUMN]
    write_csv_rows(paths["plan"] / "jobs.csv", rows, field_order)
    write_csv_rows(paths["plan"] / "msas.csv", list(msa_used.values()),
                   ["msa", "how", "sequence_sha1", "md5"])

    counts: dict[str, dict] = {}
    for spec, _, n_tokens in ready:
        s = assigned[spec.job_id]
        c = counts.setdefault(s, {"shard": s, "n_inputs": 0, "max_tokens": 0, "sum_tokens": 0})
        c["n_inputs"] += 1
        c["max_tokens"] = max(c["max_tokens"], n_tokens)
        c["sum_tokens"] += n_tokens
    shard_rows = []
    for c in sorted(counts.values(), key=lambda c: c["shard"]):
        band = band_for(c["max_tokens"], table) if table else None
        shard_rows.append({**c,
                           "suggested_gpumem_gb": band["gpumem_gb"] if band else "",
                           "sizing_basis": band["basis"] if band else ""})
    write_csv_rows(paths["plan"] / "shards.csv", shard_rows,
                   ["shard", "n_inputs", "max_tokens", "sum_tokens", "suggested_gpumem_gb",
                    "sizing_basis"])

    protocol = {"diffusion_samples": args.samples, "seed": args.seed,
                "use_potentials": not args.no_potentials,
                "recycling_steps": args.recycling_steps,
                "sampling_steps": args.sampling_steps,
                "step_scale": args.step_scale,
                "output_format": args.output_format,
                "affinity_mw_correction": args.affinity_mw_correction,
                "extra": list(args.boltz_arg or [])}
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "inputs_dir": str(inputs_dir), "msa_store": str(store) if store else None,
            "n_inputs": len(rows), "n_planned": written, "n_shards": len(shard_rows),
            "per_shard": per_shard, "run_dir": str(run_dir), "protocol": protocol,
            "driver_commit": git_commit(Path(__file__).resolve().parent)}
    (paths["plan"] / "plan.json").write_text(json.dumps(meta, indent=2) + "\n")

    # What to ask SLURM for, per band, from this plan's own protocol and shard contents. Written
    # out as a script rather than printed for copying, so the array ranges cannot drift from
    # the shards and the resources cannot drift from the protocol -- and so it can be read
    # before it runs.
    timing = load_timing()
    bands = []
    for rec in band_records:
        b = rec["band"] or {}
        names = [shard_name(i) for i in range(rec["array_from"], rec["array_to"] + 1)]
        secs = max(shard_seconds(counts[n]["sum_tokens"], protocol["diffusion_samples"],
                                 counts[n]["n_inputs"], rec["affinity"],
                                 protocol["use_potentials"], timing) for n in names)
        bands.append({"max_tokens": rec["max_tokens"], "affinity": rec["affinity"],
                      "n_inputs": rec["n_inputs"], "n_shards": rec["n_shards"],
                      "largest_tokens": rec["largest"],
                      "array_from": rec["array_from"], "array_to": rec["array_to"],
                      "gpumem_gb": b.get("gpumem_gb"),
                      "mem_mb_per_cpu": b.get("mem_mb_per_cpu"),
                      "time": hhmmss(secs * args.time_safety),
                      "predicted_seconds": round(secs),
                      "memory_basis": b.get("basis", "no sizing table")})
    (paths["plan"] / "bands.json").write_text(json.dumps(bands, indent=2) + "\n")
    submit = paths["plan"] / "submit.sh"
    write_submit_script(submit, run_dir, bands, args.time_safety)

    log(f"{len(rows)} input(s): {written} planned, {len(rows) - written} skipped")
    for a in advisories:
        log(f"  NOTE  {a}")
    log(f"{len(shard_rows)} shard(s) of at most {per_shard}; {len(msa_used)} alignment file(s)")
    log(f"protocol: {json.dumps(protocol)}")
    for band in bands:
        log(f"  band <= {band['max_tokens']} tokens{' [affinity]' if band['affinity'] else ''}: "
            f"array {band['array_from']}-{band['array_to']}, gpumem {band['gpumem_gb']}g, "
            f"time {band['time']}  ({band['n_inputs']} input(s), memory {band['memory_basis']})")
    if bands:
        log(f"submit with:  bash {submit}")
        log("  (one sbatch call per band; read it first -- it is short)")
    return 0


def write_submit_script(path: Path, run_dir: Path, bands: list[dict], safety: float) -> None:
    """A bash script that submits the run: one sbatch call per band, nothing else.

    A generated script rather than something `plan` submits itself, for two reasons. It can be
    read before it runs, which is the point of a plan. And `plan` runs inside the image, where
    sbatch does not exist.
    """
    tool = Path(__file__).resolve().parent
    out = [
        "#!/bin/bash",
        "# Submits one Boltz-2 run: one sbatch call per size band. Written by `boltz2_run.py plan`",
        f"# on {datetime.datetime.now():%Y-%m-%d %H:%M}.",
        "#",
        "# Each band asks for what its largest shard needs and no more. Walltimes are the timing",
        f"# model in sizing.tsv times {safety:g}. If a band fails for lack of memory or time, edit",
        "# its line here and resubmit it alone: shards that already finished exit at once.",
        "#",
        "#     bash submit.sh",
        "set -euo pipefail",
        f'TOOL="{tool}"',
        f'RUN="{run_dir}"',
        "# The job scripts write their logs here, and SLURM cannot create the directory itself.",
        'mkdir -p "/cluster/scratch/$USER/boltz2/slurm"',
        "",
    ]
    for b in bands:
        label = (f"tokens<={b['max_tokens']}{' affinity' if b['affinity'] else ''}, "
                 f"{b['n_inputs']} input(s) in {b['n_shards']} shard(s), "
                 f"memory {b['memory_basis']}")
        out.append(f"# {label}")
        out.append(f"j=$(sbatch --parsable --array={b['array_from']}-{b['array_to']} "
                   f"--gres=gpumem:{b['gpumem_gb']}g --time={b['time']} "
                   f"--mem-per-cpu={b['mem_mb_per_cpu']} \"$TOOL/submit_boltz2.sh\" \"$RUN\")")
        out.append(f'echo "job ${{j%%;*}}  array {b["array_from"]}-{b["array_to"]}  '
                   f'gpumem {b["gpumem_gb"]}g  time {b["time"]}  ({label})"')
        out.append("")
    path.write_text("\n".join(out))
    path.chmod(0o755)


def protocol_args(protocol: dict) -> list[str]:
    a = ["--diffusion_samples", str(protocol["diffusion_samples"]),
         "--seed", str(protocol["seed"]),
         "--recycling_steps", str(protocol.get("recycling_steps", 3)),
         "--sampling_steps", str(protocol.get("sampling_steps", 200)),
         "--step_scale", str(protocol.get("step_scale", 1.638)),
         "--output_format", protocol["output_format"]]
    # The wrapper turns steering on unless told otherwise, so "off" must be said explicitly.
    a.append("--use_potentials" if protocol["use_potentials"] else "--no_potentials")
    if protocol.get("affinity_mw_correction"):
        a.append("--affinity_mw_correction")
    a += list(protocol.get("extra") or [])
    return a


def cmd_protocol(args: argparse.Namespace) -> int:
    meta = load_plan_meta(Path(args.run_dir).expanduser().resolve())
    print("\n".join(protocol_args(meta["protocol"])))
    return 0


# --- collect ----------------------------------------------------------------------------------
def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def read_prediction(pred: Path, jid: str) -> tuple[dict, list[str]]:
    """Scores of one prediction directory: model_0 and the mean over samples.

    Every scalar the confidence record holds is reported, rather than a fixed list, so a shape
    whose record carries extra fields (a multimer's per-chain terms, say) is not silently cut.
    """
    out: dict = {}
    confs = []
    for f in sorted(pred.glob(f"confidence_{jid}_model_*.json"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        confs.append(json.loads(f.read_text()))
    out[NSAMPLES_COLUMN] = len(confs)
    fields: list[str] = []
    for c in confs:
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in fields:
                fields.append(k)
    for field in fields:
        vals = [c[field] for c in confs
                if isinstance(c.get(field), (int, float)) and not isinstance(c.get(field), bool)]
        out[PREFIX + field] = confs[0].get(field, "") if confs else ""
        out[PREFIX + field + "_mean"] = _mean(vals)
    cols = [NSAMPLES_COLUMN] + [PREFIX + f + s for f in fields for s in ("", "_mean")]

    aff = pred / f"affinity_{jid}.json"
    if aff.is_file():
        a = json.loads(aff.read_text())
        for k, v in a.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[PREFIX + k] = v
                cols.append(PREFIX + k)
    return out, cols


def cmd_collect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    paths = run_paths(run_dir)
    shard = args.shard if args.shard.startswith("shard_") else shard_name(int(args.shard))
    meta = load_plan_meta(run_dir)
    protocol = meta["protocol"]

    rows, fieldnames = read_csv_rows(paths["plan"] / "jobs.csv")
    planned = [r for r in rows if r.get(SHARD_COLUMN) == shard]
    if not planned:
        fail(f"the plan has no inputs for {shard}")

    pred_root = paths["work"] / shard / f"boltz_results_{shard}" / "predictions"
    out_csv = paths["shards"] / f"{shard}.csv"
    out_zip = paths["shards"] / f"{shard}.zip"
    receipt = paths["shards"] / f"{shard}.job.json"
    job = json.loads(receipt.read_text()) if receipt.is_file() else {}

    out_fields = list(fieldnames)
    n_ok, files = 0, []
    for row in planned:
        jid = row["job_id"].strip()
        pred = pred_root / jid
        row[GPU_COLUMN] = job.get("gpu", "")
        if not pred.is_dir() or not any(pred.glob(f"confidence_{jid}_model_*.json")):
            row[STATUS_COLUMN] = "no_prediction"
            continue
        try:
            scores, cols = read_prediction(pred, jid)
        except Exception as e:
            row[STATUS_COLUMN] = f"unreadable_prediction: {type(e).__name__}"
            continue
        row.update(scores)
        out_fields += [c for c in cols if c not in out_fields]
        files += [p for p in sorted(pred.iterdir()) if p.is_file()]
        if row[NSAMPLES_COLUMN] < protocol["diffusion_samples"]:
            row[STATUS_COLUMN] = (f"incomplete: {row[NSAMPLES_COLUMN]} of "
                                  f"{protocol['diffusion_samples']} samples")
        else:
            row[STATUS_COLUMN] = "scored"
            n_ok += 1
    if GPU_COLUMN not in out_fields:
        out_fields.append(GPU_COLUMN)

    write_csv_rows(out_csv, planned, out_fields)
    log(f"{shard}: {n_ok} of {len(planned)} prediction(s) complete -> {out_csv}")

    # One zip of every file the predictions wrote. The CSV carries the scalars; the zip keeps the
    # structures and the full records for whatever is asked later. Zip, not a solid tar, so one
    # member can be read back without unpacking the rest.
    verified = True
    if files:
        tmp = out_zip.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f"{f.parent.name}/{f.name}")
        tmp.replace(out_zip)
        with zipfile.ZipFile(out_zip) as zf:
            n_members = len(zf.namelist())
            bad = zf.testzip()
        if n_members != len(files) or bad is not None:
            log(f"  VERIFY FAILED: {n_members} member(s) for {len(files)} file(s), bad={bad}")
            verified = False
        else:
            log(f"  zip verified: {n_members} member(s), {out_zip.stat().st_size / 1e6:.1f} MB")

    complete = n_ok == len(planned)
    if not args.allow_cleanup:
        log("  scratch tree kept (cleanup not requested)")
        return 0 if complete else 1
    if not verified:
        log("  scratch tree KEPT: the archive did not verify")
        return 1
    if not complete and not args.accept_incomplete:
        log(f"  scratch tree KEPT: {len(planned) - n_ok} prediction(s) incomplete. Resubmitting "
            f"the shard redoes only those (boltz skips finished predictions). If they fail for "
            f"good, close the shard with `collect --allow-cleanup --accept-incomplete`")
        return 1
    if not under_any(run_dir, scratch_roots()):
        log(f"  scratch tree KEPT: {run_dir} is not under {[str(r) for r in scratch_roots()]}")
        return 1
    for victim in (paths["work"] / shard, paths["inputs"] / shard):
        if victim.is_dir():
            shutil.rmtree(victim)
            log(f"  cleared {victim}")
    (paths["shards"] / f"{shard}.done").write_text(
        f"{datetime.datetime.now().isoformat(timespec='seconds')} {n_ok}/{len(planned)}\n")
    return 0


# --- finalize ---------------------------------------------------------------------------------
def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    paths = run_paths(run_dir)
    meta = load_plan_meta(run_dir)
    label = args.label or run_dir.name

    plan_rows, plan_fields = read_csv_rows(paths["plan"] / "jobs.csv")
    shard_csvs = sorted(paths["shards"].glob("shard_*.csv"))
    if not shard_csvs:
        fail(f"no shard tables under {paths['shards']}; nothing to finalize")
    open_shards = sorted({r[SHARD_COLUMN] for r in plan_rows if r.get(SHARD_COLUMN)}
                         - {p.stem for p in paths["shards"].glob("shard_*.done")})
    if open_shards and not args.allow_open:
        fail(f"{len(open_shards)} shard(s) not finished: {', '.join(open_shards[:10])}"
             f"{' ...' if len(open_shards) > 10 else ''}; pass --allow-open to finalize anyway")

    out_fields = list(plan_fields)
    merged: dict[str, dict] = {}
    for sc in shard_csvs:
        rows, fields = read_csv_rows(sc)
        out_fields += [c for c in fields if c not in out_fields]
        for r in rows:
            merged[r["job_id"].strip()] = r

    final_rows = []
    for prow in plan_rows:
        row = merged.get(prow["job_id"].strip())
        if row is None:
            row = dict(prow)
            if row.get(STATUS_COLUMN) == "planned":
                row[STATUS_COLUMN] = "shard_not_collected"
        final_rows.append(row)
    status_counts: dict[str, int] = {}
    for r in final_rows:
        k = r.get(STATUS_COLUMN, "")
        status_counts[k] = status_counts.get(k, 0) + 1

    # Which image folded each shard. Scores from two images in one table would be read as if they
    # came from one tool, so a mixture is reported loudly.
    jobs = {}
    for jf in sorted(paths["shards"].glob("shard_*.job.json")):
        jobs[jf.name.split(".")[0]] = json.loads(jf.read_text())
    images = sorted({j.get("image_sha256", "unknown") for j in jobs.values()})
    gpus = sorted({j.get("gpu", "unknown") for j in jobs.values()})

    dest.mkdir(parents=True, exist_ok=True)
    scores_csv = dest / f"{label}_boltz2_scores.csv"
    write_csv_rows(scores_csv, final_rows, out_fields)

    manifest = []
    for sz in sorted(paths["shards"].glob("shard_*.zip")):
        target = dest / f"{label}_boltz2_{sz.name}"
        if target.exists() and md5_file(target) == md5_file(sz):
            pass
        elif target.exists():
            fail(f"{target} exists with different contents; archives are never replaced")
        else:
            shutil.copy2(sz, target.with_suffix(".zip.part"))
            target.with_suffix(".zip.part").replace(target)
        with zipfile.ZipFile(target) as zf:
            if zf.testzip() is not None:
                fail(f"{target} did not verify after copying")
            for m in zf.namelist():
                manifest.append({"job_id": m.split("/")[0], "member": m, "zip": target.name})
    write_csv_rows(dest / f"{label}_boltz2_manifest.csv", manifest, ["job_id", "member", "zip"])

    p = meta["protocol"]
    prov = [
        f"Boltz-2 run: {label}",
        f"finalized       : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"run directory   : {run_dir}",
        f"inputs          : {meta.get('inputs_dir')}",
        f"planned         : {meta.get('n_planned')} of {meta.get('n_inputs')} input(s), "
        f"in {meta.get('n_shards')} shard(s)",
        f"MSA store       : {meta.get('msa_store')} (per-file md5: plan/msas.csv in the run dir)",
        f"driver commit   : {meta.get('driver_commit')}",
        f"image sha256    : {', '.join(images)}"
        + ("   <-- MORE THAN ONE IMAGE" if len(images) > 1 else ""),
        f"image paths     : {', '.join(sorted({j.get('image', 'unknown') for j in jobs.values()}))}",
        f"GPU types       : {', '.join(gpus)}",
        f"protocol        : {p['diffusion_samples']} diffusion sample(s), seed {p['seed']}, "
        f"steering (--use_potentials) {'ON' if p['use_potentials'] else 'OFF'}, "
        f"{p['recycling_steps']} recycling steps, {p['output_format']}",
        f"boltz options   : {' '.join(protocol_args(p))}",
        "",
        f"scores table    : {scores_csv.name} ({len(final_rows)} rows, md5 {md5_file(scores_csv)})",
        f"archives        : {len({m['zip'] for m in manifest})} zip(s), {len(manifest)} member(s); "
        f"{label}_boltz2_manifest.csv says which zip holds which prediction",
        "",
        "status counts:",
    ] + [f"  {k or '(blank)'}: {v}" for k, v in sorted(status_counts.items())] + [
        "",
        "Reading the scores: each boltz2:<field> is the top-ranked sample (model_0, which Boltz",
        "ranks by confidence_score); boltz2:<field>_mean is the mean over all samples. Higher is",
        "more confident, except the PDE fields (predicted distance error, in Angstrom, lower is",
        "better). boltz2:affinity_pred_value is log10(IC50 / uM), so LOWER means stronger",
        "predicted binding; boltz2:affinity_probability_binary is P(binder), higher is more",
        "likely a binder. Which fields appear depends on what was folded: ligand_iptm only",
        "exists when there is a ligand, affinity_* only when the input asked for affinity.",
    ]
    (dest / f"{label}_boltz2_provenance.txt").write_text("\n".join(prov) + "\n")

    log(f"{len(final_rows)} row(s) -> {scores_csv}")
    for k, v in sorted(status_counts.items()):
        log(f"  {k or '(blank)'}: {v}")
    if len(images) > 1:
        log(f"WARNING: shards were folded with {len(images)} different images")
        return 1
    return 0


# --- cli --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="a table of chains -> one Boltz-2 YAML per id")
    pp.add_argument("--table", required=True, help="CSV with id, entity, sequence (one row per chain)")
    pp.add_argument("--out", required=True, help="directory to write the YAMLs into")
    pp.add_argument("--override", action="store_true", help="replace inputs that already exist")
    pp.set_defaults(func=cmd_prep)

    m = sub.add_parser("msa", help="fetch the alignments the inputs need into a store (network)")
    m.add_argument("--inputs", required=True, help="directory of .yaml/.fasta inputs")
    m.add_argument("--msa-store", required=True, help="directory of <sha1 of sequence>.csv")
    m.add_argument("--host", default=COLABFOLD_HOST)
    m.add_argument("--batch", type=int, default=30, help="sequences per server request")
    m.add_argument("--pause", type=float, default=5.0, help="seconds between requests")
    m.add_argument("--pairing-strategy", default="greedy", choices=("greedy", "complete"))
    m.add_argument("--dry-run", action="store_true", help="say what is missing, fetch nothing")
    m.set_defaults(func=cmd_msa)

    s = sub.add_parser("size", help="what to ask SLURM for, from the inputs or from a finished run")
    s.add_argument("--inputs", help="directory of inputs to size")
    s.add_argument("--from-run", help="instead: report what this finished run actually used")
    s.add_argument("--per-shard", type=int, default=DEFAULT_PER_SHARD)
    s.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    s.add_argument("--no-potentials", action="store_true", help="size for steering turned off")
    s.add_argument("--safety", type=float, default=1.5,
                   help="margin on the predicted walltime (default 1.5). The timing model was "
                        "fitted on one GPU type, and the same work has been seen to take "
                        "several times longer on a different card; a job that overruns its "
                        "walltime is killed and loses everything it did. Lower it towards 1.1 "
                        "if you pin the card with EXPECT_GPU.")
    s.add_argument("--json", default=None, help="also write the suggestion to this file")
    s.set_defaults(func=lambda a: cmd_size_from_run(a) if a.from_run else
                   (cmd_size(a) if a.inputs else fail("pass --inputs or --from-run")))

    p = sub.add_parser("plan", help="inputs -> shards on scratch; fixes the protocol")
    p.add_argument("--inputs", required=True, help="directory of .yaml/.fasta inputs")
    p.add_argument("--run-dir", required=True, help="run directory on scratch")
    p.add_argument("--msa-store", default=None, help="alignments from `msa`")
    p.add_argument("--allow-missing-msa", action="store_true",
                   help="fold inputs with no alignment in single-sequence mode")
    p.add_argument("--per-shard", type=int, default=DEFAULT_PER_SHARD,
                   help="inputs per array task")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="diffusion samples per input")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-potentials", action="store_true",
                   help="turn inference-time steering off (it is on by default)")
    p.add_argument("--recycling-steps", type=int, default=3)
    p.add_argument("--sampling-steps", type=int, default=200)
    p.add_argument("--step-scale", type=float, default=1.638)
    p.add_argument("--affinity-mw-correction", action="store_true",
                   help="Boltz's molecular-weight correction of the affinity value")
    p.add_argument("--output-format", choices=("mmcif", "pdb"), default="mmcif")
    p.add_argument("--boltz-arg", action="append", default=[],
                   help="any other boltz predict option, repeatable "
                        "(e.g. --boltz-arg=--subsample_msa)")
    p.add_argument("--time-safety", type=float, default=1.5,
                   help="margin on each band's predicted walltime in plan/submit.sh (default "
                        "1.5): the timing model was fitted on one GPU type, and runtime varies "
                        "several-fold between cards")
    p.add_argument("--override", action="store_true", help="replace a plan nothing has run from")
    p.add_argument("--allow-any-path", action="store_true",
                   help="permit a run directory outside scratch (tests only)")
    p.set_defaults(func=cmd_plan)

    t = sub.add_parser("protocol", help="print the boltz options this plan fixed")
    t.add_argument("--run-dir", required=True)
    t.set_defaults(func=cmd_protocol)

    c = sub.add_parser("collect", help="one finished shard -> CSV + zip, then clear its scratch tree")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--shard", required=True, help="shard_0000 or 0")
    c.add_argument("--allow-cleanup", action="store_true",
                   help="delete the shard's scratch tree once its archive verifies")
    c.add_argument("--accept-incomplete", action="store_true",
                   help="with --allow-cleanup: close a shard whose missing inputs will not be retried")
    c.set_defaults(func=cmd_collect)

    f = sub.add_parser("finalize", help="shards -> one scores table in project storage")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--dest", required=True, help="destination directory, in project storage")
    f.add_argument("--label", default=None, help="prefix for the output files (default: run name)")
    f.add_argument("--allow-open", action="store_true",
                   help="finalize although some shards never finished")
    f.set_defaults(func=cmd_finalize)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
