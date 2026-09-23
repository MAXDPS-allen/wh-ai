"""Pure Smidt-compatible half-path construction and screening decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.ferroelectricity.polarization import EnergyTrend
from scipy.interpolate import UnivariateSpline


GAP_STOP_EV = 0.01
PS_MIN_UC_CM2 = 0.1
POL_SMOOTH_MAX_UC_CM2 = 0.1
ENERGY_SMOOTH_MAX_EV_ATOM = 0.01
POLAR_LOWER_MIN_EV_ATOM = 0.001
REFINEMENT_IMAGE_COUNTS = MappingProxyType({"coarse": 10, "dense": 19})


@dataclass(frozen=True)
class StaticObservation:
    image_index: int
    status: str
    energy_eV_atom: float | None
    gap_eV: float | None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(str(value) for value in self.reason_codes))


@dataclass(frozen=True)
class FastPathDecision:
    state: str
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(str(value) for value in self.reason_codes))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class BranchResult:
    vectors_uC_cm2: np.ndarray
    branch_indices: tuple[tuple[int, int, int], ...]
    shortest_quantum_uC_cm2: float
    max_step_uC_cm2: float
    ambiguous: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors_uC_cm2, dtype=float).copy()
        if vectors.ndim != 2 or vectors.shape[1:] != (3,) or not np.all(np.isfinite(vectors)):
            raise ValueError("branch vectors must be a finite N x 3 array")
        vectors.setflags(write=False)
        object.__setattr__(self, "vectors_uC_cm2", vectors)
        object.__setattr__(
            self,
            "branch_indices",
            tuple(tuple(int(value) for value in row) for row in self.branch_indices),
        )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))


@dataclass(frozen=True)
class BranchComparison:
    accepted: bool
    max_residual_uC_cm2: float
    global_shift_indices: tuple[int, int, int]


def polarization_quantum_lattice(structure: Structure) -> np.ndarray:
    """Return the Cartesian polarization-lattice basis in microCoulomb/cm^2."""
    if not isinstance(structure, Structure) or len(structure) == 0:
        raise TypeError("structure must be a nonempty pymatgen Structure")
    result = 1602.176634 * np.asarray(structure.lattice.matrix, dtype=float) / structure.volume
    result.setflags(write=False)
    return result


def _quantum_lattice(quantum_basis: np.ndarray) -> tuple[np.ndarray, Lattice, float]:
    basis = np.asarray(quantum_basis, dtype=float)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("quantum_basis must be a finite 3 x 3 matrix")
    if abs(float(np.linalg.det(basis))) <= 1e-12:
        raise ValueError("quantum_basis must be nonsingular")
    lattice = Lattice(basis)
    radius = float(np.min(np.linalg.norm(basis, axis=1)))
    points = lattice.get_points_in_sphere([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], radius + 1e-9)
    nonzero = [float(distance) for _, distance, _, _ in points if float(distance) > 1e-10]
    if not nonzero:
        raise ValueError("could not determine shortest polarization quantum")
    return basis, lattice, min(nonzero)


def _nearest_quantum_images(
    delta_cart: np.ndarray, basis: np.ndarray, lattice: Lattice, shortest: float
) -> list[tuple[float, tuple[int, int, int]]]:
    fractional = np.asarray(delta_cart, dtype=float) @ np.linalg.inv(basis)
    best_distance, _ = lattice.get_distance_and_image([0.0, 0.0, 0.0], fractional)
    points = lattice.get_points_in_sphere(
        [fractional],
        [0.0, 0.0, 0.0],
        float(best_distance) + shortest + 1e-8,
    )
    candidates: dict[tuple[int, int, int], float] = {}
    for _, distance, _, image in points:
        key = tuple(int(value) for value in image)
        candidates[key] = min(float(distance), candidates.get(key, float("inf")))
    return sorted((distance, image) for image, distance in candidates.items())


def unwrap_cartesian_branch(
    raw_cartesian_uC_cm2: np.ndarray, quantum_basis: np.ndarray
) -> BranchResult:
    """Choose successive nearest representatives in the full polarization lattice."""
    raw = np.asarray(raw_cartesian_uC_cm2, dtype=float)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("raw polarization must be a finite nonempty N x 3 array")
    basis, lattice, shortest = _quantum_lattice(quantum_basis)
    adjusted: list[np.ndarray] = []
    indices: list[tuple[int, int, int]] = []
    reasons: list[str] = []
    steps: list[float] = []
    previous = np.zeros(3, dtype=float)
    for row in raw:
        candidates = _nearest_quantum_images(row - previous, basis, lattice, shortest)
        if len(candidates) < 2:
            raise ValueError("polarization lattice search returned fewer than two images")
        best_distance, image = candidates[0]
        if abs(candidates[1][0] - best_distance) <= 1e-6:
            reasons.append("branch_nearest_image_tie")
        vector = row + np.asarray(image, dtype=float) @ basis
        step = float(np.linalg.norm(vector - previous))
        if step >= 0.5 * shortest - 1e-12:
            reasons.append("branch_step_reaches_half_shortest_quantum")
        adjusted.append(vector)
        indices.append(image)
        steps.append(step)
        previous = vector
    return BranchResult(
        np.asarray(adjusted),
        tuple(indices),
        shortest,
        max(steps, default=0.0),
        bool(reasons),
        tuple(dict.fromkeys(reasons)),
    )


def compare_branch_with_pymatgen(
    authoritative_cartesian_uC_cm2: np.ndarray,
    pymatgen_cartesian_uC_cm2: np.ndarray,
    quantum_basis: np.ndarray,
    tolerance_uC_cm2: float = 1e-5,
) -> BranchComparison:
    """Compare paths after fitting one global polarization-quantum shift."""
    authoritative = np.asarray(authoritative_cartesian_uC_cm2, dtype=float)
    comparison = np.asarray(pymatgen_cartesian_uC_cm2, dtype=float)
    if (
        authoritative.shape != comparison.shape
        or authoritative.ndim != 2
        or authoritative.shape[1:] != (3,)
        or len(authoritative) == 0
        or not np.all(np.isfinite(authoritative))
        or not np.all(np.isfinite(comparison))
    ):
        raise ValueError("branch comparison requires matching finite N x 3 arrays")
    if not math.isfinite(tolerance_uC_cm2) or tolerance_uC_cm2 < 0.0:
        raise ValueError("cross-check tolerance must be finite and nonnegative")
    basis, lattice, shortest = _quantum_lattice(quantum_basis)
    candidates = _nearest_quantum_images(
        comparison[0] - authoritative[0], basis, lattice, shortest
    )
    shift = candidates[0][1]
    aligned = comparison + np.asarray(shift, dtype=float) @ basis
    residual = np.linalg.norm(aligned - authoritative, axis=1)
    maximum = float(np.max(residual))
    return BranchComparison(maximum <= tolerance_uC_cm2, maximum, shift)


def classify_path_metrics(
    *,
    refinement_level: str,
    static_state: str,
    static_reason_codes: Sequence[str],
    branch_ambiguous: bool,
    crosscheck_residual_uC_cm2: float,
    spontaneous_polarization_uC_cm2: float,
    polarization_smoothness_uC_cm2: float,
    energy_smoothness_eV_atom: float,
    parent_minus_polar_eV_atom: float,
) -> FastPathDecision:
    """Apply the frozen state precedence to already computed path metrics."""
    if refinement_level not in REFINEMENT_IMAGE_COUNTS:
        raise ValueError("refinement_level must be 'coarse' or 'dense'")
    static_reasons = tuple(str(value) for value in static_reason_codes)
    metrics = {
        "refinement_level": refinement_level,
        "spontaneous_polarization_uC_cm2": spontaneous_polarization_uC_cm2,
        "polarization_smoothness_uC_cm2": polarization_smoothness_uC_cm2,
        "energy_smoothness_eV_atom": energy_smoothness_eV_atom,
        "parent_minus_polar_eV_atom": parent_minus_polar_eV_atom,
        "polarization_branch_crosscheck_residual_uC_cm2": crosscheck_residual_uC_cm2,
    }
    if static_state == "operational_inconclusive":
        return FastPathDecision(static_state, static_reasons, metrics)
    if static_state == "path_metallic_stop":
        return FastPathDecision(static_state, static_reasons, metrics)
    if static_state != "static_pass":
        return FastPathDecision(
            "operational_inconclusive", ("invalid_static_gate_state",), metrics
        )
    numeric = (
        crosscheck_residual_uC_cm2,
        spontaneous_polarization_uC_cm2,
        polarization_smoothness_uC_cm2,
        energy_smoothness_eV_atom,
        parent_minus_polar_eV_atom,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        return FastPathDecision(
            "operational_inconclusive", ("path_metric_nonfinite",), metrics
        )
    if crosscheck_residual_uC_cm2 > 1e-5:
        return FastPathDecision(
            "operational_inconclusive",
            ("polarization_branch_crosscheck_mismatch",),
            metrics,
        )
    dense_reasons: list[str] = []
    if branch_ambiguous:
        dense_reasons.append(
            "polarization_branch_ambiguous"
            if refinement_level == "coarse"
            else "dense_path_still_ambiguous"
        )
    if polarization_smoothness_uC_cm2 >= POL_SMOOTH_MAX_UC_CM2:
        dense_reasons.append(
            "polarization_path_non_smooth"
            if refinement_level == "coarse"
            else "dense_path_still_non_smooth"
        )
    if energy_smoothness_eV_atom >= ENERGY_SMOOTH_MAX_EV_ATOM:
        dense_reasons.append(
            "energy_path_non_smooth"
            if refinement_level == "coarse"
            else "dense_path_still_non_smooth"
        )
    if dense_reasons:
        state = "needs_dense_path" if refinement_level == "coarse" else "path_inconclusive"
        return FastPathDecision(state, tuple(dict.fromkeys(dense_reasons)), metrics)
    inconclusive: list[str] = []
    if spontaneous_polarization_uC_cm2 <= PS_MIN_UC_CM2:
        inconclusive.append("polarization_below_resolution")
    if parent_minus_polar_eV_atom < POLAR_LOWER_MIN_EV_ATOM:
        inconclusive.append("polar_not_lower_than_parent")
    if inconclusive:
        return FastPathDecision("path_inconclusive", tuple(inconclusive), metrics)
    return FastPathDecision("smidt_fast_pass", (), metrics)


def analyze_fast_path(
    static_observations: Sequence[StaticObservation],
    branch: BranchResult,
    pymatgen_branch_cartesian_uC_cm2: np.ndarray,
    quantum_basis: np.ndarray,
    *,
    refinement_level: str,
) -> FastPathDecision:
    """Combine static, branch, spline, and endpoint evidence into one screen state."""
    static = classify_static(static_observations, refinement_level)
    if static.state != "static_pass":
        return static
    expected = REFINEMENT_IMAGE_COUNTS[refinement_level]
    comparison = np.asarray(pymatgen_branch_cartesian_uC_cm2, dtype=float)
    if len(branch.vectors_uC_cm2) != expected or comparison.shape != (expected, 3):
        return FastPathDecision(
            "operational_inconclusive",
            ("polarization_image_count_mismatch",),
            {"expected_image_count": expected, "branch_image_count": len(branch.vectors_uC_cm2)},
        )
    try:
        checked = compare_branch_with_pymatgen(
            branch.vectors_uC_cm2, comparison, quantum_basis
        )
        basis = np.asarray(quantum_basis, dtype=float)
        unit_directions = basis / np.linalg.norm(basis, axis=1)[:, None]
        components = branch.vectors_uC_cm2 @ np.linalg.inv(unit_directions)
        x_values = np.arange(expected, dtype=float)
        component_smoothness = []
        for column in range(3):
            spline = UnivariateSpline(x_values, components[:, column])
            residual = spline(x_values) - components[:, column]
            component_smoothness.append(float(np.sqrt(np.mean(np.square(residual)))))
        polarization_smoothness = max(component_smoothness)
        rows = tuple(static_observations)
        energies = np.asarray([float(row.energy_eV_atom) for row in rows], dtype=float)
        gaps = np.asarray([float(row.gap_eV) for row in rows], dtype=float)
        energy_smoothness = float(EnergyTrend(list(energies)).smoothness())
    except Exception:
        return FastPathDecision(
            "operational_inconclusive", ("path_metric_analysis_failed",), {}
        )
    spontaneous = float(
        np.linalg.norm(branch.vectors_uC_cm2[-1] - branch.vectors_uC_cm2[0])
    )
    parent_minus_polar = float(energies[0] - energies[-1])
    path_maximum = float((np.max(energies) - energies[-1]) * 1000.0)
    decision = classify_path_metrics(
        refinement_level=refinement_level,
        static_state=static.state,
        static_reason_codes=static.reason_codes,
        branch_ambiguous=branch.ambiguous,
        crosscheck_residual_uC_cm2=checked.max_residual_uC_cm2,
        spontaneous_polarization_uC_cm2=spontaneous,
        polarization_smoothness_uC_cm2=polarization_smoothness,
        energy_smoothness_eV_atom=energy_smoothness,
        parent_minus_polar_eV_atom=parent_minus_polar,
    )
    metrics = dict(decision.metrics)
    metrics.update(
        {
            "image_count": expected,
            "gap_min_eV": float(np.min(gaps)),
            "path_maximum_meV_atom": path_maximum,
            "polarization_component_smoothness_uC_cm2": component_smoothness,
            "polarization_quantum_basis_uC_cm2": basis.tolist(),
            "branch_indices": [list(row) for row in branch.branch_indices],
            "branch_max_step_uC_cm2": branch.max_step_uC_cm2,
            "branch_reason_codes": list(branch.reason_codes),
            "pymatgen_global_shift_indices": list(checked.global_shift_indices),
        }
    )
    return FastPathDecision(decision.state, decision.reason_codes, metrics)


def _integer_triplet(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-vector")
    if not all(type(item) is int for item in value):
        raise ValueError(f"{name} must contain integer values")
    return int(value[0]), int(value[1]), int(value[2])


def interpolate_mapped_half_path(
    parent: Structure,
    polar: Structure,
    polar_to_parent: Sequence[int],
    mapping_jimages: Sequence[Sequence[int]],
    image_count: int,
) -> tuple[Structure, ...]:
    """Interpolate a frozen mapped parent-to-polar path without remapping sites."""
    if not isinstance(parent, Structure) or not isinstance(polar, Structure):
        raise TypeError("parent and polar must be pymatgen Structure objects")
    if len(parent) == 0 or len(parent) != len(polar):
        raise ValueError("endpoint atom counts differ")
    if image_count not in REFINEMENT_IMAGE_COUNTS.values():
        raise ValueError("image_count must be 10 or 19")
    if not np.array_equal(np.asarray(parent.lattice.matrix), np.asarray(polar.lattice.matrix)):
        raise ValueError("endpoint lattice or basis differs")
    if len(polar_to_parent) != len(polar):
        raise ValueError("mapping length differs from endpoint atom count")
    if not all(type(value) is int for value in polar_to_parent):
        raise ValueError("mapping must contain integer indices")
    mapping = tuple(int(value) for value in polar_to_parent)
    if set(mapping) != set(range(len(parent))):
        raise ValueError("mapping must be a bijection")
    if len(mapping_jimages) != len(polar):
        raise ValueError("image-shift length differs from endpoint atom count")
    shifts = np.asarray(
        [_integer_triplet(value, "mapping image shift") for value in mapping_jimages],
        dtype=float,
    )
    for polar_index, parent_index in enumerate(mapping):
        if parent[parent_index].species != polar[polar_index].species:
            raise ValueError("mapped endpoint species differ")

    parent_coords = np.asarray(
        [parent.frac_coords[parent_index] for parent_index in mapping], dtype=float
    )
    polar_coords = np.asarray(polar.frac_coords, dtype=float) + shifts
    delta = polar_coords - parent_coords
    species = [site.species for site in polar]
    images = []
    for image_index in range(image_count):
        fraction = image_index / (image_count - 1)
        images.append(
            Structure(
                parent.lattice,
                species,
                parent_coords + fraction * delta,
                coords_are_cartesian=False,
                to_unit_cell=False,
            )
        )
    return tuple(images)


def classify_static(
    observations: Sequence[StaticObservation], refinement_level: str
) -> FastPathDecision:
    """Apply completeness before the frozen metallic-path threshold."""
    if refinement_level not in REFINEMENT_IMAGE_COUNTS:
        raise ValueError("refinement_level must be 'coarse' or 'dense'")
    expected = REFINEMENT_IMAGE_COUNTS[refinement_level]
    rows = tuple(observations)
    if len(rows) != expected:
        return FastPathDecision(
            "operational_inconclusive",
            ("static_image_count_mismatch",),
            {"image_count": len(rows), "expected_image_count": expected},
        )
    if any(not isinstance(row, StaticObservation) for row in rows):
        return FastPathDecision(
            "operational_inconclusive",
            ("invalid_static_observation",),
            {"image_count": len(rows)},
        )
    if tuple(row.image_index for row in rows) != tuple(range(expected)):
        return FastPathDecision(
            "operational_inconclusive",
            ("static_image_indices_invalid",),
            {"image_count": len(rows)},
        )
    incomplete_reasons: list[str] = []
    for row in rows:
        if row.status != "complete":
            incomplete_reasons.extend(row.reason_codes or (f"static_{row.status}",))
    if incomplete_reasons:
        return FastPathDecision(
            "operational_inconclusive",
            tuple(dict.fromkeys(incomplete_reasons)),
            {"image_count": len(rows)},
        )
    values = [(row.energy_eV_atom, row.gap_eV) for row in rows]
    if any(
        energy is None
        or gap is None
        or not math.isfinite(float(energy))
        or not math.isfinite(float(gap))
        for energy, gap in values
    ):
        return FastPathDecision(
            "operational_inconclusive",
            ("static_value_nonfinite",),
            {"image_count": len(rows)},
        )
    gaps = tuple(float(row.gap_eV) for row in rows if row.gap_eV is not None)
    if any(gap < 0.0 for gap in gaps):
        return FastPathDecision(
            "operational_inconclusive",
            ("static_gap_invalid",),
            {"image_count": len(rows)},
        )
    gap_min = min(gaps)
    metrics = {
        "image_count": len(rows),
        "refinement_level": refinement_level,
        "gap_min_eV": gap_min,
    }
    if gap_min < GAP_STOP_EV:
        return FastPathDecision(
            "path_metallic_stop", ("path_gap_below_0p01_eV",), metrics
        )
    return FastPathDecision("static_pass", (), metrics)
