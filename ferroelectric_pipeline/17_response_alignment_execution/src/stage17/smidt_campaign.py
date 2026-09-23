"""Source binding and preparation primitives for the Stage 17 Smidt fast path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.ferroelectricity.polarization import Polarization

from dfc_dft.contracts import FunnelConfig
from dfc_dft.parents import validate_parent

from .smidt_fast_path import (
    FastPathDecision,
    REFINEMENT_IMAGE_COUNTS,
    StaticObservation,
    analyze_fast_path,
    classify_static,
    interpolate_mapped_half_path,
    polarization_quantum_lattice,
    unwrap_cartesian_branch,
)
from .smidt_outputs import ParsedBerry, ParsedStatic, parse_berry_output, parse_static_output


@dataclass(frozen=True)
class EndpointBinding:
    material_id: str
    parent: Structure
    polar: Structure
    parent_path: Path
    polar_path: Path
    parent_sha256: str
    polar_sha256: str
    parent_outcar_sha256: str
    polar_outcar_sha256: str
    gate_results_sha256: str


@dataclass(frozen=True)
class ReuseDecision:
    reusable: bool
    reason_code: str


@dataclass(frozen=True)
class PreparedPath:
    material_id: str
    campaign_dir: Path
    refinement_level: str
    structures: tuple[Structure, ...]
    polar_to_parent: tuple[int, ...]
    mapping_jimages: tuple[tuple[int, int, int], ...]
    mapping_accepted: bool
    source_manifest_sha256: str
    static_work_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class StaticCollection:
    observations: tuple[StaticObservation, ...]
    decision: FastPathDecision


@dataclass(frozen=True)
class BerryCollection:
    observations: tuple[ParsedBerry, ...]
    decision: FastPathDecision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _safe_source(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("endpoint source must be a nonempty relative path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
        raise ValueError("endpoint source escapes endpoint run")
    resolved = (root / Path(*relative.parts)).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("endpoint source escapes endpoint run")
    return resolved


def _require_hash(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    actual = _sha256(path)
    if expected != actual:
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def bind_candidate_endpoints(
    endpoint_run: Path, gate_results_path: Path, candidate_id: str
) -> EndpointBinding:
    """Bind one eligible candidate to the exact relaxed endpoint bytes."""
    endpoint_run = Path(endpoint_run)
    gate_results_path = Path(gate_results_path)
    try:
        gate = json.loads(gate_results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid endpoint gate results") from exc
    rows = [row for row in gate.get("results", []) if row.get("material_id") == candidate_id]
    if len(rows) != 1:
        raise ValueError("candidate must appear exactly one time in endpoint gate results")
    row = rows[0]
    if row.get("gate_decision") != "eligible_for_parent_and_polar_gamma_dfpt":
        raise ValueError("candidate is not eligible for the fast path")
    if row.get("strict_parent_validation", {}).get("accepted") is not True:
        raise ValueError("candidate strict parent validation is not accepted")
    phase_values: dict[str, tuple[Path, str, str]] = {}
    for phase in ("parent", "polar"):
        phase_row = row.get(phase)
        if not isinstance(phase_row, dict) or phase_row.get("force_gate_pass") is not True:
            raise ValueError(f"{phase} endpoint force gate is not accepted")
        source = _safe_source(endpoint_run, phase_row.get("source"))
        contcar = source / "CONTCAR"
        outcar = source / "OUTCAR"
        contcar_sha = _require_hash(contcar, phase_row.get("contcar_sha256"), f"{phase} CONTCAR")
        outcar_sha = _require_hash(outcar, phase_row.get("outcar_sha256"), f"{phase} OUTCAR")
        phase_values[phase] = (contcar, contcar_sha, outcar_sha)
    try:
        parent = Structure.from_file(phase_values["parent"][0])
        polar = Structure.from_file(phase_values["polar"][0])
    except Exception as exc:
        raise ValueError("could not parse relaxed endpoint CONTCAR") from exc
    return EndpointBinding(
        candidate_id,
        parent,
        polar,
        phase_values["parent"][0],
        phase_values["polar"][0],
        phase_values["parent"][1],
        phase_values["polar"][1],
        phase_values["parent"][2],
        phase_values["polar"][2],
        _sha256(gate_results_path),
    )


def canonical_structure_sha256(structure: Structure) -> str:
    payload = {
        "lattice": [[round(float(value), 12) for value in row] for row in structure.lattice.matrix],
        "species": [site.species_string for site in structure],
        "frac_coords": [
            [round(float(value % 1.0), 12) for value in row] for row in structure.frac_coords
        ],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def assess_endpoint_static_reuse(
    expected_structure: Structure, expected_manifest_sha256: str, existing_dir: Path
) -> ReuseDecision:
    existing_dir = Path(existing_dir)
    poscar = existing_dir / "POSCAR"
    manifest = existing_dir / "input_manifest.json"
    if not poscar.is_file():
        return ReuseDecision(False, "missing_poscar")
    try:
        existing = Structure.from_file(poscar)
    except Exception:
        return ReuseDecision(False, "invalid_poscar")
    if canonical_structure_sha256(existing) != canonical_structure_sha256(expected_structure):
        return ReuseDecision(False, "structure_hash_mismatch")
    if not manifest.is_file():
        return ReuseDecision(False, "missing_input_manifest")
    if _sha256(manifest) != expected_manifest_sha256:
        return ReuseDecision(False, "input_manifest_hash_mismatch")
    return ReuseDecision(True, "exact_structure_and_input_manifest_match")


def prepare_path_inputs(
    *,
    output_root: Path,
    endpoint_run: Path,
    gate_results_path: Path,
    candidate_id: str,
    config: FunnelConfig,
    refinement_level: str,
    static_writer: Callable[[Structure, Path, FunnelConfig], None],
) -> PreparedPath:
    """Validate relaxed endpoints and create one no-clobber path/input tree."""
    if not isinstance(config, FunnelConfig):
        raise TypeError("config must be a FunnelConfig")
    if refinement_level not in REFINEMENT_IMAGE_COUNTS:
        raise ValueError("refinement_level must be 'coarse' or 'dense'")
    endpoints = bind_candidate_endpoints(endpoint_run, gate_results_path, candidate_id)
    decision = validate_parent(endpoints.polar, endpoints.parent, config)
    if not decision.accepted:
        raise ValueError("relaxed endpoint parent validation failed: " + ",".join(decision.reasons))
    source_manifest = {
        "schema_version": 1,
        "material_id": candidate_id,
        "refinement_level": refinement_level,
        "image_count": REFINEMENT_IMAGE_COUNTS[refinement_level],
        "gate_results": {
            "path": str(Path(gate_results_path).resolve()),
            "sha256": endpoints.gate_results_sha256,
        },
        "parent": {"path": str(endpoints.parent_path), "sha256": endpoints.parent_sha256},
        "polar": {"path": str(endpoints.polar_path), "sha256": endpoints.polar_sha256},
        "config_sha256": config.sha256(),
        "polar_to_parent": list(decision.mapping),
        "mapping_jimages": [list(row) for row in decision.mapping_jimages],
        "mapping_rms_A": decision.mapping_rms_A,
        "mapping_max_A": decision.mapping_max_A,
    }
    manifest_bytes = _canonical_json(source_manifest)
    source_digest = hashlib.sha256(manifest_bytes).hexdigest()
    campaign_dir = Path(output_root) / f"smidt-fast-{source_digest[:16]}"
    path_root = campaign_dir / "paths" / refinement_level
    path_root.mkdir(parents=True, exist_ok=False)
    structures_root = path_root / "structures"
    structures_root.mkdir()
    static_root = campaign_dir / "work" / refinement_level / "static"
    static_root.mkdir(parents=True)
    (path_root / "source_manifest.json").write_bytes(manifest_bytes)
    structures = interpolate_mapped_half_path(
        endpoints.parent,
        endpoints.polar,
        decision.mapping,
        decision.mapping_jimages,
        REFINEMENT_IMAGE_COUNTS[refinement_level],
    )
    work_dirs: list[Path] = []
    for index, structure in enumerate(structures):
        structure_path = structures_root / f"image-{index:03d}.json"
        structure_path.write_bytes(_canonical_json(structure.as_dict()))
        destination = static_root / f"image-{index:03d}"
        static_writer(structure, destination, config)
        work_dirs.append(destination)
    return PreparedPath(
        candidate_id,
        campaign_dir,
        refinement_level,
        structures,
        tuple(decision.mapping),
        tuple(decision.mapping_jimages),
        decision.accepted,
        source_digest,
        tuple(work_dirs),
    )


def collect_static_path(
    structures: Sequence[Structure],
    work_dirs: Sequence[Path],
    refinement_level: str,
    *,
    parser: Callable[[Path, Structure], ParsedStatic] = parse_static_output,
) -> StaticCollection:
    observations: list[StaticObservation] = []
    for index, (structure, work_dir) in enumerate(zip(structures, work_dirs)):
        parsed = parser(Path(work_dir), structure)
        observations.append(
            StaticObservation(
                image_index=index,
                status="complete" if parsed.status == "complete" else "incomplete",
                energy_eV_atom=parsed.energy_eV_atom,
                gap_eV=parsed.gap_eV,
                reason_codes=parsed.reason_codes,
            )
        )
    rows = tuple(observations)
    return StaticCollection(rows, classify_static(rows, refinement_level))


def prepare_berry_inputs(
    output_root: Path,
    structures: Sequence[Structure],
    static_decision: FastPathDecision,
    *,
    config: FunnelConfig,
    berry_writer: Callable[[Structure, Path, FunnelConfig], None],
) -> tuple[Path, ...]:
    if static_decision.state != "static_pass":
        raise ValueError("Berry inputs require a static_pass decision")
    expected = int(static_decision.metrics.get("image_count", -1))
    if len(structures) != expected or expected not in REFINEMENT_IMAGE_COUNTS.values():
        raise ValueError("Berry structure count does not match static_pass")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    destinations: list[Path] = []
    for index, structure in enumerate(structures):
        destination = root / f"image-{index:03d}"
        berry_writer(structure, destination, config)
        destinations.append(destination)
    return tuple(destinations)


def collect_berry_path(
    structures: Sequence[Structure],
    work_dirs: Sequence[Path],
    static_collection: StaticCollection,
    refinement_level: str,
    *,
    parser: Callable[[Path, Structure], ParsedBerry] = parse_berry_output,
) -> BerryCollection:
    """Collect Berry vectors and run the full three-dimensional path analysis."""
    if refinement_level not in REFINEMENT_IMAGE_COUNTS:
        raise ValueError("refinement_level must be 'coarse' or 'dense'")
    if static_collection.decision.state != "static_pass":
        raise ValueError("Berry collection requires a static_pass decision")
    expected = REFINEMENT_IMAGE_COUNTS[refinement_level]
    if len(structures) != expected or len(work_dirs) != expected:
        return BerryCollection(
            (),
            FastPathDecision(
                "operational_inconclusive",
                ("berry_image_count_mismatch",),
                {
                    "expected_image_count": expected,
                    "structure_count": len(structures),
                    "work_dir_count": len(work_dirs),
                },
            ),
        )
    parsed = tuple(parser(Path(path), structure) for path, structure in zip(work_dirs, structures))
    incomplete = [
        reason
        for observation in parsed
        if observation.status != "complete"
        for reason in (observation.reason_codes or ("berry_output_incomplete",))
    ]
    if incomplete:
        return BerryCollection(
            parsed,
            FastPathDecision(
                "operational_inconclusive", tuple(dict.fromkeys(incomplete)), {"image_count": expected}
            ),
        )
    p_elecs = [observation.p_elec for observation in parsed]
    p_ions = [observation.p_ion for observation in parsed]
    try:
        polarization = Polarization(p_elecs, p_ions, structures)
        unit_directions = np.array(structures[-1].lattice.matrix, dtype=float, copy=True)
        unit_directions /= np.linalg.norm(unit_directions, axis=1)[:, None]
        raw_components = (
            np.asarray(polarization.p_elecs, dtype=float)
            + np.asarray(polarization.p_ions, dtype=float)
        ) * (-1602.1766 / float(structures[-1].volume))
        raw_cartesian = raw_components @ unit_directions
        pymatgen_components = np.asarray(
            polarization.get_same_branch_polarization_data(
                convert_to_muC_per_cm2=True, all_in_polar=True
            ),
            dtype=float,
        )
        pymatgen_cartesian = pymatgen_components @ unit_directions
        quantum = polarization_quantum_lattice(structures[-1])
        branch = unwrap_cartesian_branch(raw_cartesian, quantum)
        decision = analyze_fast_path(
            static_collection.observations,
            branch,
            pymatgen_cartesian,
            quantum,
            refinement_level=refinement_level,
        )
    except Exception:
        decision = FastPathDecision(
            "operational_inconclusive",
            ("polarization_analysis_failed",),
            {"image_count": expected},
        )
    return BerryCollection(parsed, decision)
