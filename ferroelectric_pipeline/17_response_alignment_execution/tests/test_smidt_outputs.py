from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.outputs import Vasprun


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
STAGE17_ROOT = Path(__file__).resolve().parents[1]
STAGE16_SRC = PIPELINE_ROOT / "16_method_validation_and_completion/src"
sys.path.insert(0, str(STAGE17_ROOT / "src"))
sys.path.insert(0, str(STAGE16_SRC))

from stage17.smidt_outputs import parse_berry_output, parse_static_output  # noqa: E402


STATIC_FIXTURE = PIPELINE_ROOT / "test_run/GeTe_mp-938/vasp/static_image_00"
BERRY_FIXTURE = PIPELINE_ROOT / "test_run/SrAlGeH_mp-980057/vasp/polar_image_00"
UNCONVERGED_FIXTURE = (
    PIPELINE_ROOT
    / "12_latent_fe_discovery/tier1_screen/dft_173_no_hse/KErTe2_mp-9263/dfpt_gamma"
)


def _final_structure(path: Path) -> Structure:
    return Vasprun(
        str(path / "vasprun.xml"),
        parse_dos=False,
        parse_eigen=True,
        parse_projected_eigen=False,
        exception_on_bad_xml=True,
    ).final_structure


def test_real_completed_static_output_is_parsed_strictly() -> None:
    parsed = parse_static_output(STATIC_FIXTURE, _final_structure(STATIC_FIXTURE))
    assert parsed.status == "complete"
    assert parsed.reason_codes == ()
    assert parsed.energy_eV_atom == pytest.approx(-7.86967783 / 2)
    assert parsed.gap_eV == pytest.approx(0.0024)
    assert parsed.final_structure is not None


def test_missing_truncated_and_failure_marker_are_distinct(tmp_path: Path) -> None:
    missing = parse_static_output(tmp_path / "absent", _final_structure(STATIC_FIXTURE))
    assert missing.reason_codes == ("missing_outcar",)

    truncated_dir = tmp_path / "truncated"
    truncated_dir.mkdir()
    (truncated_dir / "OUTCAR").write_text("VASP calculation still running\n", encoding="utf-8")
    truncated = parse_static_output(truncated_dir, _final_structure(STATIC_FIXTURE))
    assert truncated.reason_codes == ("truncated_output",)

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    (failed_dir / "OUTCAR").write_text("VERY BAD NEWS: test failure\n", encoding="utf-8")
    failed = parse_static_output(failed_dir, _final_structure(STATIC_FIXTURE))
    assert failed.reason_codes == ("vasp_failure_marker",)


def test_real_normally_terminated_but_unconverged_output_is_not_complete() -> None:
    parsed = parse_static_output(UNCONVERGED_FIXTURE, _final_structure(UNCONVERGED_FIXTURE))
    assert parsed.status == "operational_inconclusive"
    assert parsed.reason_codes == ("electronic_not_converged",)


@pytest.mark.parametrize("drift", ["species", "lattice", "coordinates"])
def test_static_output_rejects_final_structure_identity_drift(drift: str) -> None:
    expected = _final_structure(STATIC_FIXTURE).copy()
    if drift == "species":
        expected.replace(0, "Sn")
        reason = "final_structure_species_mismatch"
    elif drift == "lattice":
        expected = Structure(
            Lattice(np.asarray(expected.lattice.matrix) * 1.000001),
            [site.species for site in expected],
            expected.frac_coords,
            to_unit_cell=False,
        )
        reason = "final_structure_lattice_mismatch"
    else:
        expected.translate_sites([0], [2e-6, 0.0, 0.0], frac_coords=True, to_unit_cell=False)
        reason = "final_structure_coordinate_mismatch"
    parsed = parse_static_output(STATIC_FIXTURE, expected)
    assert parsed.status == "operational_inconclusive"
    assert parsed.reason_codes == (reason,)


def test_real_completed_berry_output_contains_both_vectors() -> None:
    parsed = parse_berry_output(BERRY_FIXTURE, _final_structure(BERRY_FIXTURE))
    assert parsed.status == "complete"
    assert parsed.reason_codes == ()
    assert parsed.p_elec is not None and len(parsed.p_elec) == 3
    assert parsed.p_ion is not None and len(parsed.p_ion) == 3
    assert np.all(np.isfinite(parsed.p_elec))
    assert np.all(np.isfinite(parsed.p_ion))


@pytest.mark.parametrize(
    ("removed_token", "reason"),
    [
        ("p[elc]", "missing_p_elec"),
        ("p[ion]", "missing_p_ion"),
    ],
)
def test_berry_output_distinguishes_missing_polarization_terms(
    tmp_path: Path, removed_token: str, reason: str
) -> None:
    work = tmp_path / removed_token.replace("[", "_").replace("]", "")
    work.mkdir()
    shutil.copy2(BERRY_FIXTURE / "vasprun.xml", work / "vasprun.xml")
    lines = (BERRY_FIXTURE / "OUTCAR").read_text(errors="replace").splitlines()
    (work / "OUTCAR").write_text(
        "\n".join(line for line in lines if removed_token not in line) + "\n",
        encoding="utf-8",
    )
    parsed = parse_berry_output(work, _final_structure(BERRY_FIXTURE))
    assert parsed.status == "operational_inconclusive"
    assert parsed.reason_codes == (reason,)
