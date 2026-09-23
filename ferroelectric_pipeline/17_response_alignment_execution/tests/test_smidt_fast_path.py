from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


STAGE17_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE17_ROOT / "src"))

from stage17.smidt_fast_path import (  # noqa: E402
    StaticObservation,
    classify_static,
    interpolate_mapped_half_path,
)


def _endpoints() -> tuple[Structure, Structure]:
    lattice = Lattice(
        [
            [4.0, 0.0, 0.0],
            [1.0, 4.0, 0.0],
            [0.2, 0.3, 5.0],
        ]
    )
    parent = Structure(
        lattice,
        ["Na", "Cl"],
        [[0.95, 0.20, 0.30], [0.25, 0.40, 0.50]],
        to_unit_cell=False,
    )
    polar = Structure(
        lattice,
        ["Cl", "Na"],
        [[0.30, 0.45, 0.50], [0.05, 0.25, 0.35]],
        to_unit_cell=False,
    )
    return parent, polar


@pytest.mark.parametrize("image_count", [10, 19])
def test_interpolation_uses_frozen_mapping_and_recovers_unwrapped_endpoints(
    image_count: int,
) -> None:
    parent, polar = _endpoints()
    images = interpolate_mapped_half_path(
        parent,
        polar,
        polar_to_parent=(1, 0),
        mapping_jimages=((0, 0, 0), (1, 0, 0)),
        image_count=image_count,
    )

    assert len(images) == image_count
    assert [site.species_string for site in images[0]] == ["Cl", "Na"]
    assert np.array_equal(images[0].lattice.matrix, parent.lattice.matrix)
    assert np.array_equal(images[-1].lattice.matrix, parent.lattice.matrix)
    np.testing.assert_allclose(
        images[0].frac_coords,
        np.asarray([parent.frac_coords[1], parent.frac_coords[0]]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        images[-1].frac_coords,
        np.asarray([polar.frac_coords[0], polar.frac_coords[1] + [1, 0, 0]]),
        atol=1e-15,
        rtol=0.0,
    )
    midpoint = images[(image_count - 1) // 2]
    expected_lambda = ((image_count - 1) // 2) / (image_count - 1)
    expected = images[0].frac_coords + expected_lambda * (
        images[-1].frac_coords - images[0].frac_coords
    )
    np.testing.assert_allclose(midpoint.frac_coords, expected, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize(
    ("mapping", "jimages", "image_count", "message"),
    [
        ((0, 0), ((0, 0, 0), (0, 0, 0)), 10, "bijection"),
        ((1,), ((0, 0, 0),), 10, "mapping length"),
        ((1, 0), ((0, 0, 0),), 10, "image-shift length"),
        ((1, 0), ((0, 0, 0), (1.5, 0, 0)), 10, "integer"),
        ((1, 0), ((0, 0, 0), (1, 0, 0)), 11, "10 or 19"),
    ],
)
def test_interpolation_rejects_invalid_mapping_contract(
    mapping: tuple[int, ...],
    jimages: tuple[tuple[float, float, float], ...],
    image_count: int,
    message: str,
) -> None:
    parent, polar = _endpoints()
    with pytest.raises(ValueError, match=message):
        interpolate_mapped_half_path(
            parent,
            polar,
            polar_to_parent=mapping,
            mapping_jimages=jimages,
            image_count=image_count,
        )


def test_interpolation_rejects_lattice_and_species_drift() -> None:
    parent, polar = _endpoints()
    drifted_lattice = polar.copy()
    drifted_lattice.lattice = Lattice(np.asarray(polar.lattice.matrix) * 1.000001)
    with pytest.raises(ValueError, match="lattice"):
        interpolate_mapped_half_path(
            parent, drifted_lattice, (1, 0), ((0, 0, 0), (1, 0, 0)), 10
        )
    drifted_species = Structure(
        parent.lattice,
        ["K", "Na"],
        polar.frac_coords,
        to_unit_cell=False,
    )
    with pytest.raises(ValueError, match="species"):
        interpolate_mapped_half_path(
            parent, drifted_species, (1, 0), ((0, 0, 0), (1, 0, 0)), 10
        )


def _complete(count: int, gap: float = 0.5) -> list[StaticObservation]:
    return [
        StaticObservation(
            image_index=index,
            status="complete",
            energy_eV_atom=-5.0 + index * 1e-3,
            gap_eV=gap,
            reason_codes=(),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(("level", "count"), [("coarse", 10), ("dense", 19)])
def test_static_gate_accepts_complete_insulating_paths(level: str, count: int) -> None:
    decision = classify_static(_complete(count, gap=0.01), level)
    assert decision.state == "static_pass"
    assert decision.reason_codes == ()
    assert decision.metrics["image_count"] == count
    assert decision.metrics["gap_min_eV"] == pytest.approx(0.01)


def test_static_gate_stops_strictly_below_gap_boundary() -> None:
    observations = _complete(10)
    observations[4] = replace(observations[4], gap_eV=0.0099)
    decision = classify_static(observations, "coarse")
    assert decision.state == "path_metallic_stop"
    assert decision.reason_codes == ("path_gap_below_0p01_eV",)


def test_static_gate_rejects_count_status_duplicate_and_nonfinite_data() -> None:
    missing = classify_static(_complete(18), "dense")
    assert missing.state == "operational_inconclusive"
    assert "static_image_count_mismatch" in missing.reason_codes

    bad_status_rows = _complete(10)
    bad_status_rows[2] = replace(
        bad_status_rows[2],
        status="truncated",
        energy_eV_atom=None,
        gap_eV=None,
        reason_codes=("truncated_output",),
    )
    bad_status = classify_static(bad_status_rows, "coarse")
    assert bad_status.state == "operational_inconclusive"
    assert "truncated_output" in bad_status.reason_codes

    duplicate_rows = _complete(10)
    duplicate_rows[-1] = replace(duplicate_rows[-1], image_index=8)
    duplicate = classify_static(duplicate_rows, "coarse")
    assert duplicate.state == "operational_inconclusive"
    assert "static_image_indices_invalid" in duplicate.reason_codes

    nonfinite_rows = _complete(10)
    nonfinite_rows[3] = replace(nonfinite_rows[3], energy_eV_atom=float("nan"))
    nonfinite = classify_static(nonfinite_rows, "coarse")
    assert nonfinite.state == "operational_inconclusive"
    assert "static_value_nonfinite" in nonfinite.reason_codes


def test_static_gate_rejects_refinement_level_mismatch() -> None:
    decision = classify_static(_complete(19), "coarse")
    assert decision.state == "operational_inconclusive"
    assert decision.reason_codes == ("static_image_count_mismatch",)
    with pytest.raises(ValueError, match="refinement_level"):
        classify_static(_complete(10), "unknown")
