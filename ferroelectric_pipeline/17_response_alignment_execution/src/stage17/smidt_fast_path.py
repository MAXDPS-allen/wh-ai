"""Pure Smidt-compatible half-path construction and screening decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from pymatgen.core import Structure


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
