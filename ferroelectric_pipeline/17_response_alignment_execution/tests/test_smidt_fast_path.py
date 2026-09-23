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
    analyze_fast_path,
    classify_path_metrics,
    classify_static,
    compare_branch_with_pymatgen,
    interpolate_mapped_half_path,
    polarization_quantum_lattice,
    unwrap_cartesian_branch,
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


def _skew_structure() -> Structure:
    return Structure(
        Lattice(
            [
                [4.0, 0.0, 0.0],
                [1.7, 3.6, 0.0],
                [0.4, 0.8, 5.2],
            ]
        ),
        ["Na"],
        [[0.0, 0.0, 0.0]],
    )


def test_polarization_quantum_uses_full_nonorthogonal_lattice() -> None:
    structure = _skew_structure()
    expected = 1602.176634 * np.asarray(structure.lattice.matrix) / structure.volume
    actual = polarization_quantum_lattice(structure)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert abs(float(actual[0] @ actual[1])) > 1.0


def test_branch_unwraps_periodic_crossings_in_cartesian_space() -> None:
    quantum = polarization_quantum_lattice(_skew_structure())
    direction = np.asarray([0.7, -0.2, 0.4])
    continuous = np.asarray([index * direction for index in range(10)])
    raw = continuous.copy()
    raw[3:7] -= quantum[0]
    raw[7:] -= quantum[0] + quantum[1]

    result = unwrap_cartesian_branch(raw, quantum)

    np.testing.assert_allclose(result.vectors_uC_cm2, continuous, atol=1e-9, rtol=0.0)
    assert result.ambiguous is False
    assert result.branch_indices[0] == (0, 0, 0)
    assert result.branch_indices[-1] == (1, 1, 0)


def test_branch_flags_exact_half_quantum_as_ambiguous() -> None:
    quantum = polarization_quantum_lattice(_skew_structure())
    raw = np.asarray([[0.0, 0.0, 0.0], 0.5 * quantum[0]])
    result = unwrap_cartesian_branch(raw, quantum)
    assert result.ambiguous is True
    assert "branch_nearest_image_tie" in result.reason_codes
    assert "branch_step_reaches_half_shortest_quantum" in result.reason_codes


def test_pymatgen_crosscheck_accepts_one_global_quantum_shift() -> None:
    quantum = polarization_quantum_lattice(_skew_structure())
    authoritative = np.asarray([[index * 0.2, index * -0.1, index * 0.05] for index in range(10)])
    comparison = authoritative + quantum[1]
    checked = compare_branch_with_pymatgen(authoritative, comparison, quantum)
    assert checked.accepted is True
    assert checked.max_residual_uC_cm2 <= 1e-10
    assert checked.global_shift_indices == (0, -1, 0)


def test_pymatgen_crosscheck_rejects_non_global_disagreement() -> None:
    quantum = polarization_quantum_lattice(_skew_structure())
    authoritative = np.zeros((10, 3))
    comparison = authoritative + quantum[1]
    comparison[5, 0] += 2e-5
    checked = compare_branch_with_pymatgen(authoritative, comparison, quantum)
    assert checked.accepted is False
    assert checked.max_residual_uC_cm2 > 1e-5


def _metric_decision(**updates: object):
    values: dict[str, object] = {
        "refinement_level": "coarse",
        "static_state": "static_pass",
        "static_reason_codes": (),
        "branch_ambiguous": False,
        "crosscheck_residual_uC_cm2": 0.0,
        "spontaneous_polarization_uC_cm2": 5.0,
        "polarization_smoothness_uC_cm2": 0.01,
        "energy_smoothness_eV_atom": 0.001,
        "parent_minus_polar_eV_atom": 0.01,
    }
    values.update(updates)
    return classify_path_metrics(**values)


def test_path_metric_boundaries_and_state_precedence() -> None:
    assert _metric_decision().state == "smidt_fast_pass"
    assert _metric_decision(spontaneous_polarization_uC_cm2=0.1).state == "path_inconclusive"
    assert _metric_decision(polarization_smoothness_uC_cm2=0.1).state == "needs_dense_path"
    assert _metric_decision(energy_smoothness_eV_atom=0.01).state == "needs_dense_path"
    assert _metric_decision(parent_minus_polar_eV_atom=0.001).state == "smidt_fast_pass"
    assert _metric_decision(parent_minus_polar_eV_atom=0.000999).state == "path_inconclusive"

    dense = _metric_decision(
        refinement_level="dense", polarization_smoothness_uC_cm2=0.1
    )
    assert dense.state == "path_inconclusive"
    assert "dense_path_still_non_smooth" in dense.reason_codes

    metallic = _metric_decision(
        static_state="path_metallic_stop",
        static_reason_codes=("path_gap_below_0p01_eV",),
        branch_ambiguous=True,
    )
    assert metallic.state == "path_metallic_stop"

    mismatch = _metric_decision(
        crosscheck_residual_uC_cm2=1.00001e-5,
        branch_ambiguous=True,
        spontaneous_polarization_uC_cm2=0.0,
    )
    assert mismatch.state == "operational_inconclusive"
    assert mismatch.reason_codes == ("polarization_branch_crosscheck_mismatch",)


def test_public_path_decision_never_emits_training_or_confirmed_boolean() -> None:
    payload = _metric_decision().to_dict()
    forbidden = {"is_ferroelectric", "confirmed_positive", "scientific_label", "training_label"}
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["metrics"])


def test_analyze_fast_path_computes_article_style_metrics() -> None:
    structure = _skew_structure()
    quantum = polarization_quantum_lattice(structure)
    lambdas = np.linspace(0.0, 1.0, 10)
    continuous = np.outer(lambdas, np.asarray([5.0, 1.0, -0.5]))
    raw = continuous.copy()
    raw[6:] -= quantum[0]
    branch = unwrap_cartesian_branch(raw, quantum)
    pymatgen_branch = continuous + quantum[1]
    observations = [
        StaticObservation(
            image_index=index,
            status="complete",
            energy_eV_atom=-5.0 + 0.01 * (1.0 - fraction) ** 2,
            gap_eV=0.5 + 0.1 * fraction,
        )
        for index, fraction in enumerate(lambdas)
    ]

    decision = analyze_fast_path(
        observations,
        branch,
        pymatgen_branch,
        quantum,
        refinement_level="coarse",
    )

    assert decision.state == "smidt_fast_pass"
    assert decision.metrics["spontaneous_polarization_uC_cm2"] == pytest.approx(
        np.linalg.norm(continuous[-1] - continuous[0])
    )
    assert decision.metrics["parent_minus_polar_eV_atom"] == pytest.approx(0.01)
    assert decision.metrics["path_maximum_meV_atom"] == pytest.approx(10.0)
    assert decision.metrics["gap_min_eV"] == pytest.approx(0.5)
    assert decision.metrics["polarization_smoothness_uC_cm2"] < 0.1
    assert decision.metrics["energy_smoothness_eV_atom"] < 0.01


def test_analyze_fast_path_rejects_branch_image_count_mismatch() -> None:
    quantum = polarization_quantum_lattice(_skew_structure())
    branch = unwrap_cartesian_branch(np.zeros((9, 3)), quantum)
    decision = analyze_fast_path(
        _complete(10),
        branch,
        np.zeros((9, 3)),
        quantum,
        refinement_level="coarse",
    )
    assert decision.state == "operational_inconclusive"
    assert decision.reason_codes == ("polarization_image_count_mismatch",)
