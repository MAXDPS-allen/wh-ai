"""Strict adapters for static and Berry VASP outputs used by the fast path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from dfc_dft.vasp_outputs import parse_completion


@dataclass(frozen=True)
class ParsedStatic:
    status: str
    reason_codes: tuple[str, ...]
    energy_eV_atom: float | None = None
    gap_eV: float | None = None
    final_structure: Structure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))


@dataclass(frozen=True)
class ParsedBerry:
    status: str
    reason_codes: tuple[str, ...]
    p_elec: tuple[float, float, float] | None = None
    p_ion: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))


def _inconclusive_static(reason: str) -> ParsedStatic:
    return ParsedStatic("operational_inconclusive", (reason,))


def _inconclusive_berry(reason: str) -> ParsedBerry:
    return ParsedBerry("operational_inconclusive", (reason,))


def _completion_reason(work_dir: Path) -> str | None:
    outcar = work_dir / "OUTCAR"
    completion = parse_completion(outcar)
    if completion.status == "missing":
        return "missing_outcar"
    if completion.status == "running":
        return "truncated_output"
    if completion.status == "failed":
        return "vasp_failure_marker"
    if not completion.normally_terminated:
        return "abnormal_termination"
    return None


def _structure_identity_reason(actual: Structure, expected: Structure) -> str | None:
    if len(actual) != len(expected):
        return "final_structure_atom_count_mismatch"
    if tuple(site.species for site in actual) != tuple(site.species for site in expected):
        return "final_structure_species_mismatch"
    if not np.allclose(
        np.asarray(actual.lattice.matrix),
        np.asarray(expected.lattice.matrix),
        atol=1e-8,
        rtol=0.0,
    ):
        return "final_structure_lattice_mismatch"
    delta = np.asarray(actual.frac_coords) - np.asarray(expected.frac_coords)
    delta -= np.rint(delta)
    if float(np.max(np.abs(delta), initial=0.0)) > 1e-6:
        return "final_structure_coordinate_mismatch"
    return None


def _load_vasprun(path: Path) -> Vasprun:
    return Vasprun(
        str(path),
        parse_dos=False,
        parse_eigen=True,
        parse_projected_eigen=False,
        exception_on_bad_xml=True,
    )


def parse_static_output(work_dir: Path, expected_structure: Structure) -> ParsedStatic:
    """Parse a completed static and reject convergence or identity drift."""
    work_dir = Path(work_dir)
    if not isinstance(expected_structure, Structure) or len(expected_structure) == 0:
        raise TypeError("expected_structure must be a nonempty pymatgen Structure")
    completion_reason = _completion_reason(work_dir)
    if completion_reason is not None:
        return _inconclusive_static(completion_reason)
    vasprun_path = work_dir / "vasprun.xml"
    if not vasprun_path.is_file():
        return _inconclusive_static("missing_vasprun_xml")
    try:
        run = _load_vasprun(vasprun_path)
    except Exception:
        return _inconclusive_static("vasprun_parse_failed")
    if not bool(run.converged_electronic):
        return _inconclusive_static("electronic_not_converged")
    final_structure = run.final_structure
    if final_structure is None or len(final_structure) == 0:
        return _inconclusive_static("missing_final_structure")
    identity_reason = _structure_identity_reason(final_structure, expected_structure)
    if identity_reason is not None:
        return _inconclusive_static(identity_reason)
    try:
        energy = float(run.final_energy) / len(final_structure)
        gap = float(run.eigenvalue_band_properties[0])
    except Exception:
        return _inconclusive_static("static_value_parse_failed")
    if not math.isfinite(energy) or not math.isfinite(gap):
        return _inconclusive_static("static_value_nonfinite")
    if gap < 0.0:
        return _inconclusive_static("static_gap_invalid")
    return ParsedStatic("complete", (), energy, gap, final_structure)


def _finite_vector(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        vector = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
        return None
    return vector  # type: ignore[return-value]


def parse_berry_output(work_dir: Path, expected_structure: Structure) -> ParsedBerry:
    """Parse a normally terminated, electronically converged LCALCPOL output."""
    static = parse_static_output(work_dir, expected_structure)
    if static.status != "complete":
        return ParsedBerry(static.status, static.reason_codes)
    outcar_path = Path(work_dir) / "OUTCAR"
    try:
        text = outcar_path.read_text(errors="replace")
    except OSError:
        return _inconclusive_berry("outcar_parse_failed")
    if "p[elc]" not in text:
        return _inconclusive_berry("missing_p_elec")
    if "p[ion]" not in text:
        return _inconclusive_berry("missing_p_ion")
    try:
        outcar = Outcar(str(outcar_path))
    except Exception:
        return _inconclusive_berry("outcar_parse_failed")
    p_elec = _finite_vector(getattr(outcar, "p_elec", None))
    if p_elec is None:
        return _inconclusive_berry("missing_p_elec")
    p_ion = _finite_vector(getattr(outcar, "p_ion", None))
    if p_ion is None:
        return _inconclusive_berry("missing_p_ion")
    return ParsedBerry("complete", (), p_elec, p_ion)
