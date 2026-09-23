# R3 — Screen all 36 candidates

## Objective

Apply the sequence parent pre-screen → static → fixed-cell relax → selected polar DFPT and give every candidate a low-cost terminal or pending state.

## Status

`running` — R0 and R1 are available. The endpoint-gate run
`endpoint-gate-parents-78587656accad94e` is active; three force-inconclusive
endpoints are in their single allowed `restart-01` on g4.

## Entry criteria

R0 publishes 36 rows, R1 publishes response features, and the calculation-specific g4 execution profile passes its preflight/smoke gates.

## Inputs

Frozen candidate structures/scores, parent states, robust VASP input/output components, and explicit run manifests.

## Outputs

36/36 `pass`, `fail`, `inconclusive`, or `pending` states with gap, force, polarity-retention, restart, and reason fields.

## Reusable assets

`s16_candidate_release`, `s16_dft_campaign`, `s16_vasp_inputs`, `s16_vasp_outputs`, `s16_response_physics`, `s09_parent_generator`, `s09_postprocessor`.

## Minimal command

Continue the immutable Stage 17 endpoint-gate run using its recorded request, task manifests, execution policy, and no-clobber restart directories. The reusable general 36-row collector remains `implementation pending`; it may later wrap these records but must not rewrite them.

## Smidt-compatible fast path

`mp-aaacrlli` is eligible for the separate nonpolar-to-polar fast path. It uses
the relaxed accepted endpoints, ten coarse images, GPU static/gap calculations,
and CPU-only Berry calculations. It neither modifies nor waits for the live
Gamma-DFPT directories.

```bash
PY=/share/home/caiby/miniforge3/envs/fe_dft/bin/python
CLI=ferroelectric_pipeline/17_response_alignment_execution/scripts/run_r3_smidt_fast_path.py

$PY "$CLI" prepare --candidate mp-aaacrlli
$PY "$CLI" probe --campaign <smidt-fast-campaign>
$PY "$CLI" dry-run --campaign <smidt-fast-campaign> --stage static
$PY "$CLI" launch --campaign <smidt-fast-campaign> --stage static
$PY "$CLI" collect-static --campaign <smidt-fast-campaign>
$PY "$CLI" prepare-berry --campaign <smidt-fast-campaign>
$PY "$CLI" dry-run --campaign <smidt-fast-campaign> --stage berry
$PY "$CLI" launch --campaign <smidt-fast-campaign> --stage berry
$PY "$CLI" collect-berry --campaign <smidt-fast-campaign>
```

Every command creates a no-clobber attempt. `launch` ends in `DISPATCHED`, while
collection ends in `COMPLETED` or a typed `FAILED` record. A node/profile is not
admitted until a matching smoke record is hash-bound into
`configs/r3_smidt_fast_path_execution_policy.json`.

The public states are `smidt_fast_pass`, `needs_dense_path`,
`path_metallic_stop`, `path_inconclusive`, and `operational_inconclusive`.
`needs_dense_path` requires a 19-image `prepare-dense` follow-up. None of these
screening states is automatically a positive or negative training label.

## Checks

No duplicate job, one predetermined relax restart, explicit gap threshold, residual-force/polarity checks, and 36-row completeness. `gap < 0.01 eV` is `pbe_metallic_stop`, not a true negative.

## Stop/go criteria

Advance only insulating relaxed candidates with reliable small/medium-cell parents to polar DFPT. Missing parent is inconclusive, not failure.

## Duration and resource

Static 3–8 GPU-hours (about 0.5 day including queue/faults); insulating relax 30–80 GPU-hours and about 1–3 days; prioritized polar DFPT about 4–10 days on validated CPU slots. The g4 smoke gate has passed for the calculation-specific profile, and authorized work may use the available GPUs in batches of no more than 40 submitted tasks.

## Protected sources

The candidate release, pilot results, and Stage 16 cluster/contracts remain read-only. Never resume into a historical Stage 16 directory.

## Output location

Write attempts to `runs/R3_candidate_screening/`, candidate states to `results/R3_candidate_screening/`, and funnel reports to `reports/R3_candidate_screening/`.
