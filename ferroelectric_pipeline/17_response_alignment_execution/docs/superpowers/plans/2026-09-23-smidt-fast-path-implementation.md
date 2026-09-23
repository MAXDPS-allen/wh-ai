# Smidt Fast DFT Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a Stage 17 nonpolar-to-polar ten-image static/Berry fast path for `mp-aaacrlli`, reusing the Stage 9 Smidt method and Stage 16 VASP machinery without disturbing live DFPT work.

**Architecture:** Pure path physics lives in one deterministic module, VASP output validation in a parser adapter, and immutable campaign lifecycle plus cluster dispatch in separate modules. A thin CLI connects these pieces. Existing Stage 9 and Stage 16 remain read-only dependencies; all new code, policy, tests, and run artifacts stay in Stage 17.

**Tech Stack:** Python 3.11, pytest, pymatgen, NumPy, SciPy, Stage 16 `dfc_dft`, VASP 6.4 CPU/GPU executables, SSH/HPC-X launchers, JSON/SHA-256 provenance.

---

## File map

- Modify `src/stage17/run_contract.py`: add backward-compatible terminal-record hashing and `DISPATCHED` support.
- Create `src/stage17/smidt_fast_path.py`: interpolation, three-dimensional Berry branch, metrics, and frozen verdict.
- Create `src/stage17/smidt_outputs.py`: strict static/Berry output parsing and structure-identity checks.
- Create `src/stage17/smidt_campaign.py`: source binding, attempt/work/snapshot lifecycle, preparation, collection, and endpoint reuse.
- Create `src/stage17/smidt_cluster.py`: policy validation, task batching, no-clobber queue scripts, and dispatch records.
- Create `scripts/run_r3_smidt_fast_path.py`: `prepare`, `collect-static`, `prepare-berry`, `collect-berry`, `dry-run`, and `launch` commands.
- Create `configs/r3_smidt_fast_path_execution_policy.json`: separate GPU-static/CPU-Berry execution policy.
- Modify `tests/test_workspace.py`: lifecycle compatibility tests.
- Create `tests/test_smidt_fast_path.py`: interpolation, branch, thresholds, and decision tests.
- Create `tests/test_smidt_outputs.py`: real-output parsing and failure classification tests.
- Create `tests/test_smidt_campaign.py`: source/reuse/no-clobber/integration tests.
- Create `tests/test_smidt_cluster.py`: policy, routing, cap, and mocked dispatch tests.
- Modify `work_packages/03_r3_candidate_screening/README.md`: operator commands and state meanings.

The test interpreter is:

```bash
export PYTHONPATH="$PWD/ferroelectric_pipeline/17_response_alignment_execution/src:$PWD/ferroelectric_pipeline/16_method_validation_and_completion/src"
PY=/share/home/caiby/miniforge3/envs/fe_dft/bin/python
```

## Task 1: Extend the Stage 17 attempt lifecycle

**Files:**

- Modify: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/run_contract.py`
- Modify: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for all terminal records**

Add tests showing that legacy `SUCCESS + result.json` still validates, while new attempts support exactly one of:

```python
NEW_TERMINALS = {
    "COMPLETED": "result.json",
    "FAILED": "failure.json",
    "DISPATCHED": "dispatch.json",
}
```

Test multiple markers, marker-before-record, a missing record, and terminal-record hash drift. Update the resume fixture to use:

```python
resume_from = {
    "attempt": "attempt-01",
    "terminal_record": "dispatch.json",
    "terminal_record_sha256": sha256(dispatch_path.read_bytes()).hexdigest(),
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py::RunContractTests -q
```

Expected: new `COMPLETED`/`DISPATCHED` and terminal-record resume tests fail; existing tests pass.

- [ ] **Step 3: Implement backward-compatible terminal inspection**

Add:

```python
TERMINAL_RECORDS = {
    "SUCCESS": "result.json",      # existing Stage 17 compatibility
    "COMPLETED": "result.json",
    "FAILED": "failure.json",
    "DISPATCHED": "dispatch.json",
}

def terminal_record_path(attempt_dir: Path) -> Path | None:
    markers = [name for name in TERMINAL_RECORDS if (attempt_dir / name).exists()]
    if len(markers) != 1:
        return None
    return attempt_dir / TERMINAL_RECORDS[markers[0]]
```

Refactor `inspect_attempt()` to validate one marker/record pair. Refactor `validate_resume()` to bind the previous terminal record rather than assuming `result.json`; retain validation of old `request_sha256`/`result_sha256` records so existing attempts remain readable.

- [ ] **Step 4: Run focused and full Stage 17 tests**

Run:

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py -q
```

Expected: PASS, including all legacy lifecycle tests.

- [ ] **Step 5: Commit the lifecycle slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/src/stage17/run_contract.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py
git commit -m "feat(stage17): support dispatched campaign attempts"
```

## Task 2: Implement mapped ten-image interpolation and static gating

**Files:**

- Create: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_fast_path.py`
- Create: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py`

- [ ] **Step 1: Write failing interpolation tests**

Construct a two-species skew-cell parent/polar pair with a non-identity mapping and a periodic-boundary crossing. Assert ten images at `lambda = i / 9`, unchanged lattice, polar-site order, and exact recovery of the mapped unwrapped endpoints.

The test calls:

```python
images = interpolate_mapped_half_path(
    parent,
    polar,
    polar_to_parent=(1, 0),
    mapping_jimages=((0, 0, 0), (1, 0, 0)),
    image_count=10,
)
```

Also test atom-count, species, lattice, non-bijection, image-count, and image-shift failures.

- [ ] **Step 2: Run interpolation tests and verify RED**

Run:

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py -k interpolation -q
```

Expected: import failure because `smidt_fast_path.py` does not exist.

- [ ] **Step 3: Implement immutable path observations and interpolation**

Start the module with frozen records and constants:

```python
GAP_STOP_EV = 0.01
PS_MIN_UC_CM2 = 0.1
POL_SMOOTH_MAX_UC_CM2 = 0.1
ENERGY_SMOOTH_MAX_EV_ATOM = 0.01
POLAR_LOWER_MIN_EV_ATOM = 0.001

@dataclass(frozen=True)
class StaticObservation:
    image_index: int
    energy_eV_atom: float
    gap_eV: float

@dataclass(frozen=True)
class FastPathDecision:
    state: str
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, object]
```

Implement the frozen coordinate equation from the spec directly with NumPy. Accept
only `image_count in {10, 19}`: 10 is the coarse path and 19 is the mandatory
midpoint-refined path. Do not call `Structure.interpolate`, autosort, or
recompute shortest images.

- [ ] **Step 4: Write failing static-decision boundary tests**

Cover both `refinement_level="coarse"` (10 images) and
`refinement_level="dense"` (19 images):

```python
assert classify_static(complete(10, gap=0.0099), "coarse").state == "path_metallic_stop"
assert classify_static(complete(10, gap=0.0100), "coarse").state == "static_pass"
assert classify_static(complete(19, gap=0.0100), "dense").state == "static_pass"
assert classify_static(missing_one_of_19(), "dense").state == "operational_inconclusive"
```

Bind expected size to refinement level: coarse requires exactly 10 unique
ordered images and dense requires exactly 19. Require finite energy/gap values.
Verify a level/count mismatch and the internal `static_inconclusive` both map to
public `operational_inconclusive` with their original reasons.

- [ ] **Step 5: Implement minimal static classification and rerun tests**

Implement state precedence for static data only: malformed/missing first, then metallic, otherwise pass.

Run:

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py -k 'interpolation or static' -q
```

Expected: PASS.

- [ ] **Step 6: Commit the interpolation/static slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_fast_path.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py
git commit -m "feat(stage17): add mapped Smidt half path"
```

## Task 3: Implement the full three-dimensional polarization branch and verdict

**Files:**

- Modify: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_fast_path.py`
- Modify: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py`

- [ ] **Step 1: Write failing quantum-lattice tests**

Use an explicitly non-orthogonal lattice and raw vectors that cross a branch boundary. Test that:

- the quantum basis is `1602.176634 * lattice.matrix / volume`;
- closest-vector unwrapping returns a continuous Cartesian path;
- the answer differs from independent per-component rounding for the skew cell;
- two equidistant images or a step at half the shortest quantum is ambiguous;
- a path and the same path plus one global quantum vector are equivalent.

- [ ] **Step 2: Run branch tests and verify RED**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py -k 'quantum or branch' -q
```

- [ ] **Step 3: Implement exact local lattice enumeration**

Implement:

```python
def polarization_quantum_lattice(structure: Structure) -> np.ndarray:
    return 1602.176634 * np.asarray(structure.lattice.matrix) / structure.volume

def unwrap_cartesian_branch(raw_cart: np.ndarray, quantum_basis: np.ndarray) -> BranchResult:
    ...
```

For each step, use `pymatgen.core.Lattice(quantum_basis)` to obtain the nearest lattice image. Determine the shortest nonzero quantum by enumerating lattice points inside a sphere whose radius is the shortest basis-vector norm; enumerate candidates through `best_distance + shortest_quantum` to identify the second-best image. Do not use a fixed `[-2, 2]^3` cube.

- [ ] **Step 4: Write failing pymatgen cross-check and metric tests**

Monkeypatch the cross-check once to introduce a `2e-5 uC/cm2` mismatch and assert:

```python
decision.state == "operational_inconclusive"
decision.reason_codes == ("polarization_branch_crosscheck_mismatch",)
```

Test all exact verdict boundaries: Ps `0.1`, smoothness `0.1`, energy smoothness `0.01`, and parent-polar energy `0.001 eV/atom`. Test precedence: operational issue > metallic > dense path > inconclusive > pass. The public analysis call receives `refinement_level="coarse" | "dense"`; the same ambiguity/smoothness failure returns `needs_dense_path` at coarse level and `path_inconclusive` at dense level. Assert the serialized result contains no `is_ferroelectric`, `confirmed_positive`, or training-label field.

- [ ] **Step 5: Implement pymatgen cross-check, spline metrics, and verdict**

Use `Polarization(...).get_same_branch_polarization_data(...)` only as a cross-check. Convert its lattice-direction components to Cartesian, fit one global quantum shift at image zero, and require maximum residual `<= 1e-5 uC/cm2`. Use `UnivariateSpline` with the same cubic default for polarization components and `EnergyTrend.smoothness()` for per-atom energy. Require exactly 10 observations for `coarse` and exactly 19 for `dense`; reject any inconsistent pair before analysis.

- [ ] **Step 6: Run all physics tests**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the branch/verdict slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_fast_path.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py
git commit -m "feat(stage17): analyze three-dimensional Berry branches"
```

## Task 4: Build strict static and Berry output adapters

**Files:**

- Create: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_outputs.py`
- Create: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_outputs.py`

- [ ] **Step 1: Write failing parser tests using real repository outputs**

Use these read-only real fixtures:

- static: `ferroelectric_pipeline/test_run/GeTe_mp-938/vasp/static_image_00/`;
- Berry: `ferroelectric_pipeline/test_run/SrAlGeH_mp-980057/vasp/polar_image_00/`.

Copy fixtures to `tmp_path` inside tests before truncating or mutating them. Assert distinct reason codes for missing, truncated, VASP failure marker, electronic nonconvergence, non-finite result, atom/species mismatch, lattice mismatch, coordinate mismatch, missing `p_elec`, and missing `p_ion`.

- [ ] **Step 2: Run output tests and verify RED**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_outputs.py -q
```

- [ ] **Step 3: Implement strict parsers**

Define:

```python
@dataclass(frozen=True)
class ParsedStatic:
    status: str
    reason_codes: tuple[str, ...]
    energy_eV_atom: float | None
    gap_eV: float | None
    final_structure: Structure | None

@dataclass(frozen=True)
class ParsedBerry:
    status: str
    reason_codes: tuple[str, ...]
    p_elec: tuple[float, float, float] | None
    p_ion: tuple[float, float, float] | None
```

Reuse `dfc_dft.vasp_outputs.parse_completion()`. Parse `Vasprun` with `exception_on_bad_xml=True`, require `converged_electronic`, finite final energy/gap, exact site count/order, lattice `atol=1e-8`, and periodic coordinate displacement `<=1e-6`. Parse Berry vectors with `pymatgen.io.vasp.outputs.Outcar` only after normal termination.

- [ ] **Step 4: Run parser tests and existing Stage 16 parser tests**

```bash
$PY -m pytest \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_outputs.py \
  ferroelectric_pipeline/16_method_validation_and_completion/tests/dfc_dft/test_vasp_outputs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the parser slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_outputs.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_outputs.py
git commit -m "feat(stage17): validate Smidt VASP outputs"
```

## Task 5: Implement campaign preparation, reuse, and collection

**Files:**

- Create: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_campaign.py`
- Create: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py`

- [ ] **Step 1: Write failing source-binding tests**

Create temporary copies of the endpoint-gate JSON, endpoint `CONTCAR`/`OUTCAR`, and mapping data. Test exact SHA-256 acceptance and one-byte drift rejection. Require `gate_decision == eligible_for_parent_and_polar_gamma_dfpt`, both force gates, and accepted strict validation.

The production sources are fixed as:

```python
ENDPOINT_RUN = Path("runs/R3_candidate_screening/endpoint-gate-parents-78587656accad94e")
GATE_RESULTS = ENDPOINT_RUN / "endpoint_relax_gate_results.json"
CANDIDATE_ID = "mp-aaacrlli"
```

- [ ] **Step 2: Run source tests and verify RED**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py -k source -q
```

- [ ] **Step 3: Implement source resolution and path preparation**

Load relaxed parent/polar structures from the gate row's `source/CONTCAR`, verify the recorded hashes, then rerun `dfc_dft.parents.validate_parent(polar, parent, config)`. Freeze its `mapping` and `mapping_jimages` in `source_manifest.json`. Generate ten structures and write Stage 16 `write_switching_static()` inputs to create-once `work/static/image-000..009` directories.

Campaign layout separates the two coordinate grids so image indices cannot
collide:

```text
runs/R3_candidate_screening/smidt-fast-<source-digest>/
  attempts/attempt-01/
  paths/coarse/structures/image-000.json
  paths/dense/structures/image-000.json
  work/coarse/static/image-000/
  work/coarse/berry/image-000/
  work/dense/static/image-000/
  work/dense/berry/image-000/
  snapshots/coarse/static-001/
  snapshots/coarse/berry-001/
  snapshots/dense/static-001/
  snapshots/dense/berry-001/
```

Every prepare/collect/launch request carries `refinement_level`; a command may
read or write only its matching subtree. Dense preparation must fail if any
dense destination already exists rather than reusing a coarse directory.

- [ ] **Step 4: Write failing reuse and Berry-gate tests**

Test ten-image coverage counts. Endpoint reuse must require both canonical relaxed-structure SHA-256 and complete input-manifest SHA-256 equality. Assert any mismatch records a reason and schedules a new task. Assert `prepare_berry()` rejects 9/10 statics, a metallic image, and any operationally inconclusive image.

- [ ] **Step 5: Implement reuse, static collection, and Berry preparation**

Use Stage 16 `write_switching_berry()` only after `classify_static()` returns `static_pass`. Snapshot science-bearing `OUTCAR`, `vasprun.xml`, `CONTCAR` when present, plus canonical JSON results and `manifest.sha256`. Never mutate a terminal attempt or snapshot.

- [ ] **Step 6: Write and implement Berry collection and dense-retry tests**

Inject ten parsed Berry records and assert the final decision, metrics, reason codes, timing, and output hashes. Cover `needs_dense_path`, which must create a new `refinement_level=dense` attempt, generate all 19 structures, and run the same 19-image static gate before preparing 19 Berry inputs. A still-ambiguous/non-smooth 19-image collection must become `path_inconclusive`; it must not recursively schedule another refinement.

- [ ] **Step 7: Run campaign plus Stage 16 input tests**

```bash
$PY -m pytest \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py \
  ferroelectric_pipeline/16_method_validation_and_completion/tests/dfc_dft/test_vasp_inputs.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the campaign slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_campaign.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py
git commit -m "feat(stage17): prepare and collect Smidt campaigns"
```

## Task 6: Add the dedicated cluster policy and dispatcher

**Files:**

- Create: `ferroelectric_pipeline/17_response_alignment_execution/configs/r3_smidt_fast_path_execution_policy.json`
- Create: `ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_cluster.py`
- Create: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_cluster.py`

- [ ] **Step 1: Write failing policy/routing tests**

Test that:

- static tasks require `profile=gpu`, one GPU, and the GPU executable;
- Berry tasks require `profile=cpu`, zero GPUs, and the CPU executable;
- a node is rejected without a successful matching smoke hash;
- a probe older than 300 seconds is rejected;
- any live Stage 17 DFPT lock/process rejects admission;
- no submission contains more than 40 tasks;
- task/work paths cannot escape the campaign;
- operational failure permits one versioned retry; scientific failure permits none.

- [ ] **Step 2: Run cluster tests and verify RED**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_cluster.py -q
```

- [ ] **Step 3: Create a deny-by-default dedicated policy**

The initial file lists g1/g3/g4/g6/g7 as probeable but admits a node/profile only when a measured smoke record is present. Freeze:

```json
{
  "schema_version": 1,
  "scope": "R3_smidt_fast_path",
  "max_tasks_per_submission": 40,
  "timeouts_seconds": {"static": 43200, "berry": 86400},
  "operational_retries": 1,
  "executables": {
    "gpu": "/share/apps/vasp.6.4.2_nvhpc2311_hpcx_cuda11.8/bin/vasp_std",
    "cpu": "/share/apps/nvhpc_vasp/vasp.6.4.1/bin/vasp_std"
  },
  "probe_max_age_seconds": 300,
  "nodes": {}
}
```

Each admitted `nodes.<node>.<profile>` entry records
`smoke_record_path`, `smoke_record_sha256`, `max_concurrency`, `cpu_cores`,
`ram_gib`, and `gpu_ids`. Populate it only from Task 8's measured probe/smoke
evidence; do not copy a stale admission claim.

- [ ] **Step 4: Define and implement probe/smoke evidence contracts**

Add frozen `ProbeRecord` and `SmokeRecord` validators. A probe JSON contains:

```python
{
    "schema_version": 1,
    "node": "g4",
    "captured_utc": "...",
    "logical_cpus": 128,
    "available_ram_gib": 300.0,
    "gpus": [{"index": 0, "free_memory_gib": 39.0,
              "utilization_percent": 0, "compute_processes": []}],
    "executables": {"gpu": {"path": "...", "available": True},
                    "cpu": {"path": "...", "available": True}},
    "stage17_locks": [],
    "live_dfpt_processes": [],
}
```

`collect_live_probe(node, runner, now)` uses batch-mode SSH read-only commands
and stores the canonical record under the campaign attempt with its SHA-256.
Both `dry-run` and `launch` collect a new probe; admission requires age `<=300`
seconds and empty `stage17_locks`/`live_dfpt_processes`.

A smoke record contains node/profile, input manifest SHA-256, executable path,
launcher/environment, start/end times, exit code, normal-termination flag, and
output hashes. The policy's smoke path and SHA-256 must match this record, and
its executable/profile must match the proposed task.

- [ ] **Step 5: Implement pure planning and mocked dispatch**

Define `plan_submission(stage, tasks, policy, probe_records)` and `dispatch_submission(plan, runner=subprocess.run)`. The dispatcher writes a per-node queue script and immutable `dispatch.json`, acquires `/tmp/stage17_smidt_fast.lock` atomically, and starts the queue through batch-mode SSH. Tests inject `runner`; tests never SSH or invoke VASP.

GPU queue scripts bind one worker per admitted GPU with `CUDA_VISIBLE_DEVICES`. CPU Berry scripts use the policy's measured ranks/cores and never call the GPU binary. Scripts write PID, start/end time, exit code, and timeout state inside the versioned work directory.

- [ ] **Step 6: Run cluster and preflight regression tests**

```bash
$PY -m pytest \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_cluster.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py -q
```

- [ ] **Step 7: Commit policy and dispatcher**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/configs/r3_smidt_fast_path_execution_policy.json \
  ferroelectric_pipeline/17_response_alignment_execution/src/stage17/smidt_cluster.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_cluster.py
git commit -m "feat(stage17): route Smidt static and Berry jobs"
```

## Task 7: Add the CLI and dry-run integration

**Files:**

- Create: `ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py`
- Modify: `ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py`
- Modify: `ferroelectric_pipeline/17_response_alignment_execution/work_packages/03_r3_candidate_screening/README.md`

- [ ] **Step 1: Write failing CLI tests**

Invoke `main(argv, runner=fake_runner)` for all commands. Test that `dry-run`
makes read-only probe calls but no scientific executable call; `launch` refuses
a failed admission; Berry preparation cannot precede a complete insulating
static collection of the matching size; a second `prepare` creates
`attempt-02` and never clobbers `attempt-01`. Test that every command creates an
immutable attempt with request/source records, stdout/stderr, and exactly one
matching terminal record.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
$PY -m pytest ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py -k cli -q
```

- [ ] **Step 3: Implement the thin CLI**

Commands:

```text
prepare --candidate mp-aaacrlli
prepare-dense --candidate mp-aaacrlli --resume-from <coarse-attempt>
dry-run --candidate mp-aaacrlli --stage static|berry
launch --candidate mp-aaacrlli --stage static|berry
collect-static --candidate mp-aaacrlli
prepare-berry --candidate mp-aaacrlli
collect-berry --candidate mp-aaacrlli
probe --candidate mp-aaacrlli
smoke --candidate mp-aaacrlli --node <node> --profile gpu|cpu
```

`prepare-dense` is accepted only from a terminal coarse
`needs_dense_path` result and creates a new 19-image attempt. `probe` is
read-only. `smoke` is the only command allowed to create a policy-eligible smoke
record and runs one tiny existing Stage 17 input. The script inserts Stage 17
and Stage 16 `src/` paths, confines caches to the campaign, prints one canonical
JSON summary, and contains no scientific logic.

- [ ] **Step 4: Document exact operator workflow and states**

Add commands, paths, `smidt_fast_pass` limitations, metallic/inconclusive distinctions, and the prohibition on converting screen outcomes directly into training labels.

- [ ] **Step 5: Run the complete automated suite**

```bash
$PY -m pytest \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_fast_path.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_outputs.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_cluster.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_workspace.py -q
```

Expected: PASS with zero real SSH/VASP calls.

- [ ] **Step 6: Commit the end-to-end CLI slice**

```bash
git add ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  ferroelectric_pipeline/17_response_alignment_execution/tests/test_smidt_campaign.py \
  ferroelectric_pipeline/17_response_alignment_execution/work_packages/03_r3_candidate_screening/README.md
git commit -m "feat(stage17): expose Smidt fast-path workflow"
```

## Task 8: Prepare, admit, and run `mp-aaacrlli`

**Files:**

- Modify with measured evidence: `ferroelectric_pipeline/17_response_alignment_execution/configs/r3_smidt_fast_path_execution_policy.json`
- Create at runtime: `ferroelectric_pipeline/17_response_alignment_execution/runs/R3_candidate_screening/smidt-fast-*/`
- Create at runtime: `ferroelectric_pipeline/17_response_alignment_execution/reports/R3_candidate_screening/smidt-fast-*/timing.json`

- [ ] **Step 1: Re-run the full suite and record the clean baseline**

Run the Task 7 suite. Expected: PASS. Stop if any regression exists.

- [ ] **Step 2: Prepare the real candidate without launching**

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  prepare --candidate mp-aaacrlli
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  dry-run --candidate mp-aaacrlli --stage static
```

Expected: ten covered static images, with `reused_count + new_task_count == 10`; source hashes and strict relaxed mapping pass. Existing endpoint statics are expected not to reuse if their structures differ from relaxed `CONTCAR`, and the exact mismatch reason must be recorded.

- [ ] **Step 3: Probe all reachable nodes without touching live DFPT**

Run `probe --candidate mp-aaacrlli` to capture g1/g3/g4/g6/g7 CPU, memory,
GPU, VASP executable, active Stage 17 locks, and live VASP process paths in
canonical per-node records. Exclude g6/g7 or any other node while it owns the
existing DFPT campaign. Do not stop, renice, or alter those processes.

- [ ] **Step 4: Run one tiny static smoke per newly admitted node/profile**

Use the `smoke` command with a prepared small Stage 17 static input. Record input
hash, executable path, environment, launcher, normal termination, energy,
duration, and output hashes. Do not repeat a smoke when an unchanged matching
record already exists. Update the dedicated policy only with successful
measured records and bind each entry to the record SHA-256.

- [ ] **Step 5: Launch the non-reused static tasks**

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  launch --candidate mp-aaacrlli --stage static
```

Expected: at most ten tasks total and never more than 40 in one submission; only admitted idle resources are used.

- [ ] **Step 6: Collect static results and apply the gap stop**

After workers terminate:

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  collect-static --candidate mp-aaacrlli
```

If any output is still running, publish a progress/timing report and wait. If metallic, publish `path_metallic_stop` and do not prepare Berry inputs. If operationally inconclusive, use at most one versioned retry for the affected image.

- [ ] **Step 7: Conditionally prepare and launch CPU Berry tasks**

Only after `static_pass`:

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  prepare-berry --candidate mp-aaacrlli
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  dry-run --candidate mp-aaacrlli --stage berry
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  launch --candidate mp-aaacrlli --stage berry
```

Expected: exactly ten CPU Berry tasks, zero GPU Berry tasks.

- [ ] **Step 8: Collect and publish the terminal fast-path result**

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py \
  collect-berry --candidate mp-aaacrlli
```

Publish the state, reason codes, Ps, parent-polar energy, path maximum, minimum gap, smoothness, hashes, reused/new counts, node timings, and total wall time. If `needs_dense_path`, run `prepare-dense` and repeat the 19-image static then conditional Berry sequence before any routine deep-DFPT promotion.

- [ ] **Step 9: Verify artifacts and commit only durable code/policy/docs**

Run workspace validation plus the full tests. Do not commit large VASP raw outputs; keep their SHA-bound runtime paths and commit only policy/docs if repository conventions permit.

```bash
$PY ferroelectric_pipeline/17_response_alignment_execution/scripts/validate_workspace.py
git status --short
```

Expected: workspace valid; no files outside Stage 17 changed by the campaign.

## Stop conditions

Stop scientific dispatch, without weakening checks, if any of these occurs:

- endpoint source hash or relaxed mapping changes;
- the current DFPT directories or processes would be touched;
- no node/profile has a passing matching smoke;
- task count would exceed 40 in one submission;
- a static path image is metallic;
- a required output is truncated, unconverged, structurally mismatched, or unparsable after its single operational retry.

In each case publish a typed terminal/progress record and the smallest safe resume action.
