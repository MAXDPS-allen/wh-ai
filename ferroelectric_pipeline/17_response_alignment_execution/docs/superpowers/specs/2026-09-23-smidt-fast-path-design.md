# Smidt-compatible fast DFT path for Stage 17

## Decision

Add a fast half-path screening branch inside Stage 17. Reuse the mature
Smidt-style physics implemented in Stage 9, but use the stricter Stage 16 input,
mapping, parsing, and state contracts. The fast branch runs before expensive
Gamma-DFPT/Born calculations; it does not replace deep validation for the final
headline candidates.

Stages 8--16 remain read-only. All new code and artifacts live under Stage 17.
The currently running DFPT calculations are neither stopped nor modified.

## Existing implementation found

Stage 9 already contains the complete article-derived sequence:

1. polar/nonpolar pairing;
2. endpoint relaxation;
3. eight intermediate structures, ten path images total;
4. static energy and band-gap calculations;
5. Berry-phase calculations;
6. same-branch polarization, energy trend, and smoothness analysis.

Useful sources are `09_dft_validation/s2_interpolate.py`,
`s3_make_vasp_inputs.py`, and `s5_postprocess.py`. The Stage 9 orchestration and
cluster runner will not be called directly because they write mutable shared
files, use stale node assumptions, and expose an over-broad Boolean
`is_ferroelectric` verdict.

Stage 16 already supplies strict common-cell parent validation, MP-compatible
static/Berry input factories, immutable input manifests, completion parsing,
and Berry/switching primitives. Stage 17 supplies no-clobber attempt contracts
and the current g1/g3/g4/g6/g7 execution policies.

## Alternatives considered

### A. Run Stage 9 `validate.py` unchanged

This is the smallest code change, but it can overwrite endpoint files, uses an
obsolete direct-SSH policy, contains a GPU/CPU Berry-routing inconsistency, and
turns a limited half-path result into an unconditional ferroelectric Boolean.
Rejected.

### B. Remove the DFPT gate from the existing Stage 16 full-switching campaign

This preserves provenance but still constructs and relaxes `-P`, parent, and
`+P`, runs endpoint DFPT, and prepares a full nine-image path. It does not
deliver the intended rapid screen. Retained as the later deep-validation path.

### C. Stage 17 native fast adapter over Stage 9/16 components

Recommended. It creates a nonpolar-to-polar ten-image screen, performs a GPU
static/gap gate, then runs CPU Berry calculations only for insulating paths.
Survivors proceed to the existing full switching and response workflow.

## Scope of the first implementation

The first release supports candidates that already have:

- an accepted strict parent;
- completed fixed-cell parent and polar relaxations;
- force residuals at or below `0.01 eV/A`;
- an accepted post-relax parent/polar mapping.

The current first eligible case is `mp-aaacrlli`. `mp-aaacsman` remains
inconclusive because its polar force exceeds the gate. A candidate excluded by
an existing scientific result is not silently reintroduced.

PSEUDO/commensurate-cell fallback for the 33 parent-inconclusive candidates is a
separate follow-up slice. It must not delay the first fast-path result.

## Architecture

### Pure physics module

Create `src/stage17/smidt_fast_path.py` with small deterministic functions:

- validate a relaxed same-cell parent/polar pair and explicit atom mapping;
- generate exactly ten structures at lambda `0, 1/9, ..., 1`;
- classify static observations as `static_pass`, `path_metallic_stop`, or
  `static_inconclusive`;
- reconstruct a same-branch vector polarization using the full three-dimensional
  polarization lattice, with pymatgen used as an independent cross-check;
- compute spontaneous polarization, polar--parent energy difference, maximum
  path energy, minimum gap, and energy/polarization smoothness;
- return `smidt_fast_pass`, `needs_dense_path`, `path_metallic_stop`,
  `path_inconclusive`, or `operational_inconclusive` without emitting a
  confirmed-ferroelectric label. The static-only `static_inconclusive` is an
  internal gate result and maps to public `operational_inconclusive` with its
  original reason code.

The half-path uses the relaxed common lattice. It does not claim the linear path
is a minimum-energy switching path.

The endpoint mapping is frozen rather than rediscovered. Let `m[j]` map polar
site `j` to a parent site and let `n[j]` be its already accepted periodic-image
shift. In polar-site order, image `lambda` has

`f_j(lambda) = f_parent[m[j]] + lambda * (f_polar[j] + n[j] - f_parent[m[j]])`.

The common relaxed lattice is unchanged, `lambda = 0, 1/9, ..., 1`, and no
autosorting or shortest-image remapping is allowed during interpolation. Exact
endpoint recovery, site order, species, mapping, and image shifts are part of
the input manifest.

### Orchestrator

Create `scripts/run_r3_smidt_fast_path.py` with explicit commands:

- `prepare`: verify source hashes and endpoint gates, create a new immutable
  attempt, write ten structures and ten Stage 16 static inputs;
- `collect-static`: parse normally terminated static calculations, publish the
  gap/energy table, and stop metallic paths before Berry work;
- `prepare-berry`: write ten Stage 16 `berry_static` inputs only for static
  survivors;
- `collect-berry`: parse polarization and publish the fast-path result;
- `dry-run`: validate inputs, resource policy, routing, and task count without
  invoking VASP;
- `launch`: launch one named stage only after the same validations as dry-run.

Every invocation creates an immutable command attempt under one campaign run.
An attempt contains `request.json`, `source_manifest.json`, stdout/stderr, and
exactly one terminal record: `result.json` plus `COMPLETED`, `failure.json` plus
`FAILED`, or `dispatch.json` plus `DISPATCHED`. Its request records the previous
terminal-record hash (whichever of those three applies), so `prepare`, `launch`,
and `collect` form a hash chain without modifying an earlier attempt.
`launch` returns after accepted dispatch and ends with `DISPATCHED`; it does not
pretend the calculation is complete. Create-once VASP work directories live
under the campaign's `work/` area, outside immutable command-attempt records,
and only VASP may append to them. `collect` accepts only normally terminated
work, copies or binds its final outputs into an immutable snapshot, hashes it,
and writes the science result. An interrupted retry receives a new versioned
work directory. Every science-bearing `OUTCAR`, `vasprun.xml`, `CONTCAR`, and
result has a SHA-256 entry; resume creates a new attempt rather than reopening a
terminal one.

### Calculation routing

- Add a dedicated `configs/r3_smidt_fast_path_execution_policy.json`; do not
  reinterpret the general probe-only policy or the DFPT policy. Static/gap jobs
  use the validated GPU executable. `LCALCPOL` Berry jobs use the validated CPU
  executable because the local Stage 9 record says the GPU build can hang.
- g1/g3/g4/g6/g7 are only eligible after a node-local smoke test records the
  executable, environment, launcher, GPU/CPU inventory, and successful normal
  termination. Initial admission can reuse an unchanged, hash-matched Stage 17
  smoke record. A node with a live Stage 17 DFPT lock or observed live DFPT
  process is excluded from this campaign until released.
- Policy records per-node static and Berry concurrency, cores/GPUs per task,
  a 12-hour static timeout, a 24-hour Berry timeout, one retry for typed
  operational failures only, and an atomic campaign lock. Scientific failures
  are never retried automatically.
- No submission contains more than 40 tasks. There is no global queue-size cap.
- The fast path uses a distinct run directory and must not acquire or modify any
  live DFPT calculation directory.

Before generating new endpoint work, the adapter checks existing endpoint
statics. Reuse is allowed only when the relaxed structure hash and the complete
VASP input manifest hash match exactly. Otherwise the result records
`endpoint_static_reuse=false` and a reason such as `structure_hash_mismatch`.

## Data flow

1. Read the R0 parent release and R3 endpoint-relax result as immutable inputs.
2. Verify the recorded OUTCAR/CONTCAR hashes before reading structures.
3. Re-run strict parent validation on the relaxed pair.
4. Generate and hash ten half-path structures.
5. Run/collect static calculations and apply the `0.01 eV` gap stop.
6. Run/collect Berry calculations only for a static survivor.
7. Apply Smidt-compatible branch and smoothness analysis.
8. Publish a machine-readable result and a concise report.
9. Route only `smidt_fast_pass` or reviewed `needs_dense_path` cases to deep
   Gamma-DFPT/Born and full `-P -> parent -> +P` validation.

## Frozen analysis and decision contract

An image is usable only if VASP terminates normally, electronic convergence is
true, energy and gap are finite, and the parsed final structure has the expected
atom count, species order, lattice, and periodic coordinates. For a static
calculation (`NSW=0`), lattice and coordinates must match the input within
`1e-8` and `1e-6`, respectively. A Berry image additionally requires finite
`p_elec` and `p_ion`. Missing, truncated, unconverged, structurally mismatched,
and parser-failed outputs have separate reason codes.

The polarization quantum is the full lattice
`Q = {e(n1*a + n2*b + n3*c)/Omega | n_i in Z}` in Cartesian units. Successive
images are unwrapped with a closest-vector calculation in this lattice; Stage
9's independent component-wise quantum reduction is not reused. The branch is
ambiguous if two lattice images are within `1e-6 uC/cm2` of the best distance or
if the accepted step reaches half the shortest nonzero quantum-vector length.
The custom closest-vector result is authoritative. Pymatgen is an independent
cross-check after both paths are converted to Cartesian coordinates. A single
global quantum-vector shift is fitted at image zero and applied to all pymatgen
images; the maximum remaining per-image norm must be `<= 1e-5 uC/cm2`. A larger
difference yields `operational_inconclusive` with reason
`polarization_branch_crosscheck_mismatch`, and no polarization verdict.
The reported spontaneous polarization is the Cartesian norm of the final minus
initial value on the accepted branch. Polarization smoothness uses pymatgen's
cubic-spline RMS definition applied to the cross-checked same-branch components
along the three lattice directions; energy smoothness is pymatgen's
quartic-spline RMS residual over energies in eV/atom.

Thresholds and boundaries are fixed:

- `gap_min < 0.01 eV` is metallic; exactly `0.01 eV` passes the gap gate;
- `Ps > 0.1 uC/cm2` passes the nonzero-resolution gate; equality does not;
- polarization smoothness `< 0.1 uC/cm2` passes; equality does not;
- energy smoothness `< 0.01 eV/atom` passes; equality does not;
- `E_parent - E_polar >= 0.001 eV/atom` passes the polar-lower ordering gate;
- path maximum is reported relative to the polar endpoint in meV/atom but is
  not assigned an upper cutoff in the fast screen.

State precedence is deterministic:

1. any missing, truncated, unconverged, identity-mismatched, non-finite, or
   parser-failed required output -> `operational_inconclusive`;
2. otherwise `gap_min < 0.01 eV` -> `path_metallic_stop`;
3. otherwise branch ambiguity or either smoothness failure ->
   `needs_dense_path`;
4. otherwise failure of the polarization-resolution or polar-lower gate ->
   `path_inconclusive` with explicit reason codes;
5. otherwise -> `smidt_fast_pass`.

These states are screening outcomes, not positive or negative training labels.

## Adaptive path policy

The first calculation uses ten images, matching the mature Stage 9/Smidt path.
If branch continuity or smoothness fails while all images remain insulating,
the result is `needs_dense_path`, not a failed material. The next scientific
attempt must insert all midpoints to form 19 images before that candidate can be
promoted to `smidt_fast_pass` or enter routine deep DFPT. If the 19-image path
remains ambiguous or non-smooth it becomes `path_inconclusive`. A manual
diagnostic override may send it to deep validation, but may not promote the
fast-path state.

## Output contract

Per candidate, publish:

- source endpoint paths and SHA-256 values;
- parent/polar mapping and validation metrics;
- ten structure hashes and calculation input manifests;
- per-image completion, energy, band gap, polarization vectors, and output
  paths;
- branch-adjusted polarization path and polarization quanta;
- spontaneous polarization, polar--parent energy difference, path maximum,
  minimum gap, and smoothness metrics;
- one explicit fast-path state and reason codes.

Metallicity, missing output, nonconvergence, and branch ambiguity remain
distinct. None becomes a confirmed negative training label automatically.

## Tests

Development follows red--green--refactor. Tests use synthetic structures and
arrays; they never invoke VASP or touch the live R3 run.

Required tests cover:

1. ten deterministic endpoint-inclusive images and exact endpoint recovery;
2. rejection of lattice, species, mapping, or hash drift;
3. metallic early stop before Berry preparation;
4. branch-consistent polarization and known quantum jump examples, including a
   non-orthogonal cell and a periodic-boundary-crossing path;
5. `needs_dense_path` on ambiguous/jumpy but insulating data;
6. no confirmed-ferroelectric Boolean in public results;
7. no-clobber attempts, source hash binding, terminal-record hash chaining, all
   three terminal markers, and the separation of immutable `attempts/` from
   create-once mutable `work/` directories;
8. GPU static versus CPU Berry routing;
9. maximum 40 tasks per submission;
10. real, small completed/truncated/unconverged static and Berry output
    fixtures, including proof that Berry preparation is rejected until all ten
    statics are complete and insulating;
11. a prepare/dry-run integration test using copied small fixtures.

## Acceptance criteria

Delivery is split into three observable milestones:

1. code/preparation: all tests pass and the candidate dry-run is valid;
2. admitted static execution: node smoke and policy are recorded, then the
   non-reused statics are launched without touching DFPT; all ten path images
   must be covered by either an exact-hash reuse or a new task;
3. scientific collection: collect all statics, conditionally launch and collect
   all ten Berry calculations, then publish the terminal fast-path result and
   timing report.

The end-to-end target for `mp-aaacrlli` is a terminal fast-path result within
48 hours of an admitted launch. This is a measurement target, not permission to
weaken convergence or completion checks; an overrun produces a timing report
and revised estimate rather than a fabricated result.

The implementation is complete when:

- all new and existing Stage 17 tests pass;
- `mp-aaacrlli` can be prepared into ten validated static input directories
  without altering the live DFPT run;
- dry-run reports coverage for all ten static images, split into `reused_count`
  and `new_task_count` whose sum is ten; the no-reuse fixture reports ten GPU
  static tasks, and a synthetic passing static collection reports ten CPU Berry
  tasks;
- every artifact is confined to a new Stage 17 campaign run: command records are
  immutable under `attempts/`, execution data is create-once under `work/`, and
  collected science snapshots are hash-bound to their source endpoints;
- launch is impossible until the dedicated policy and current node admission
  checks pass;
- the actual campaign either reaches a terminal scientific state or publishes
  a typed operational stop with the measured wall-clock time and resume input.
