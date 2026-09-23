from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
STAGE17_ROOT = Path(__file__).resolve().parents[1]
STAGE16_ROOT = PIPELINE_ROOT / "16_method_validation_and_completion"
sys.path.insert(0, str(STAGE17_ROOT / "src"))
sys.path.insert(0, str(STAGE16_ROOT / "src"))

from dfc_dft.contracts import load_config  # noqa: E402
from stage17.smidt_campaign import (  # noqa: E402
    assess_endpoint_static_reuse,
    bind_candidate_endpoints,
    collect_berry_path,
    collect_static_path,
    prepare_berry_inputs,
    prepare_path_inputs,
)
from stage17.smidt_outputs import ParsedBerry, ParsedStatic  # noqa: E402


REAL_ENDPOINT_RUN = (
    STAGE17_ROOT
    / "runs/R3_candidate_screening/endpoint-gate-parents-78587656accad94e"
)
REAL_GATE_RESULTS = REAL_ENDPOINT_RUN / "endpoint_relax_gate_results.json"
REAL_CONFIG = STAGE16_ROOT / "configs/born_assisted_dft_funnel.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_endpoints() -> tuple[Structure, Structure]:
    lattice = Lattice.cubic(4.0)
    parent = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    polar = Structure(lattice, ["Na", "Cl"], [[0.02, 0, 0], [0.48, 0.5, 0.5]])
    return parent, polar


def _write_endpoint_fixture(root: Path) -> tuple[Path, Path]:
    parent, polar = _toy_endpoints()
    calculations = root / "calculations"
    rows = {}
    for phase, structure in (("parent", parent), ("polar", polar)):
        directory = calculations / phase
        directory.mkdir(parents=True)
        Poscar(structure).write_file(directory / "CONTCAR")
        (directory / "OUTCAR").write_text(f"complete {phase}\n", encoding="utf-8")
        rows[phase] = {
            "source": f"calculations/{phase}",
            "contcar_sha256": _sha(directory / "CONTCAR"),
            "outcar_sha256": _sha(directory / "OUTCAR"),
            "force_gate_pass": True,
            "max_force_eV_A": 0.001,
        }
    gate = {
        "schema_version": 1,
        "results": [
            {
                "material_id": "mp-test",
                "gate_decision": "eligible_for_parent_and_polar_gamma_dfpt",
                "parent": rows["parent"],
                "polar": rows["polar"],
                "strict_parent_validation": {"accepted": True},
            }
        ],
    }
    gate_path = root / "endpoint_relax_gate_results.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n", encoding="utf-8")
    return root, gate_path


def test_endpoint_binding_requires_recorded_hashes_and_eligible_gate(tmp_path: Path) -> None:
    endpoint_run, gate_path = _write_endpoint_fixture(tmp_path / "endpoint")
    bound = bind_candidate_endpoints(endpoint_run, gate_path, "mp-test")
    assert bound.material_id == "mp-test"
    assert bound.parent_sha256 == _sha(endpoint_run / "calculations/parent/CONTCAR")
    assert bound.polar_sha256 == _sha(endpoint_run / "calculations/polar/CONTCAR")

    with (endpoint_run / "calculations/polar/CONTCAR").open("a", encoding="utf-8") as handle:
        handle.write("drift\n")
    with pytest.raises(ValueError, match="polar CONTCAR SHA-256 mismatch"):
        bind_candidate_endpoints(endpoint_run, gate_path, "mp-test")


def test_endpoint_binding_rejects_ineligible_or_duplicate_candidate(tmp_path: Path) -> None:
    endpoint_run, gate_path = _write_endpoint_fixture(tmp_path / "endpoint")
    gate = json.loads(gate_path.read_text())
    gate["results"][0]["gate_decision"] = "relaxation_inconclusive_no_second_restart"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        bind_candidate_endpoints(endpoint_run, gate_path, "mp-test")
    gate["results"].append(gate["results"][0])
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        bind_candidate_endpoints(endpoint_run, gate_path, "mp-test")


def test_static_reuse_requires_structure_and_manifest_hash_match(tmp_path: Path) -> None:
    parent, _ = _toy_endpoints()
    existing = tmp_path / "existing"
    existing.mkdir()
    Poscar(parent).write_file(existing / "POSCAR")
    (existing / "input_manifest.json").write_text('{"calc_type":"static"}\n', encoding="utf-8")
    expected_manifest_sha = _sha(existing / "input_manifest.json")

    accepted = assess_endpoint_static_reuse(parent, expected_manifest_sha, existing)
    assert accepted.reusable is True
    assert accepted.reason_code == "exact_structure_and_input_manifest_match"

    drifted = parent.copy()
    drifted.translate_sites([0], [1e-4, 0, 0], frac_coords=True)
    rejected = assess_endpoint_static_reuse(drifted, expected_manifest_sha, existing)
    assert rejected.reusable is False
    assert rejected.reason_code == "structure_hash_mismatch"

    rejected_manifest = assess_endpoint_static_reuse(parent, "0" * 64, existing)
    assert rejected_manifest.reusable is False
    assert rejected_manifest.reason_code == "input_manifest_hash_mismatch"


@pytest.mark.parametrize(("level", "count"), [("coarse", 10), ("dense", 19)])
def test_prepare_real_candidate_freezes_mapping_and_separates_refinement_paths(
    tmp_path: Path, level: str, count: int
) -> None:
    written: list[Path] = []

    def fake_writer(structure: Structure, destination: Path, config) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        Poscar(structure).write_file(destination / "POSCAR")
        (destination / "input_manifest.json").write_text(
            json.dumps({"calc_type": "static", "sites": len(structure)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)

    prepared = prepare_path_inputs(
        output_root=tmp_path,
        endpoint_run=REAL_ENDPOINT_RUN,
        gate_results_path=REAL_GATE_RESULTS,
        candidate_id="mp-aaacrlli",
        config=load_config(REAL_CONFIG),
        refinement_level=level,
        static_writer=fake_writer,
    )

    assert prepared.refinement_level == level
    assert len(prepared.structures) == count
    assert len(written) == count
    assert prepared.mapping_accepted is True
    assert len(prepared.polar_to_parent) == 72
    assert all(f"/{level}/" in str(path) for path in written)
    assert (prepared.campaign_dir / f"paths/{level}/source_manifest.json").is_file()


def test_coarse_and_dense_share_campaign_identity_but_never_paths(tmp_path: Path) -> None:
    def fake_writer(structure: Structure, destination: Path, config) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        Poscar(structure).write_file(destination / "POSCAR")
        (destination / "input_manifest.json").write_text("{}\n", encoding="utf-8")

    common = {
        "output_root": tmp_path,
        "endpoint_run": REAL_ENDPOINT_RUN,
        "gate_results_path": REAL_GATE_RESULTS,
        "candidate_id": "mp-aaacrlli",
        "config": load_config(REAL_CONFIG),
        "static_writer": fake_writer,
    }
    coarse = prepare_path_inputs(refinement_level="coarse", **common)
    dense = prepare_path_inputs(refinement_level="dense", **common)
    assert coarse.campaign_dir == dense.campaign_dir
    assert (coarse.campaign_dir / "paths/coarse/structures/image-009.json").is_file()
    assert (dense.campaign_dir / "paths/dense/structures/image-018.json").is_file()
    assert coarse.static_work_dirs[1] != dense.static_work_dirs[1]


def _parsed_static(index: int, gap: float = 0.5) -> ParsedStatic:
    return ParsedStatic(
        status="complete",
        reason_codes=(),
        energy_eV_atom=-5.0 + index * 1e-3,
        gap_eV=gap,
        final_structure=_toy_endpoints()[0],
    )


def test_static_collection_blocks_berry_until_complete_and_insulating(tmp_path: Path) -> None:
    structures = tuple(_toy_endpoints()[0] for _ in range(10))
    work_dirs = tuple(tmp_path / f"image-{index:03d}" for index in range(10))

    incomplete = collect_static_path(
        structures,
        work_dirs[:-1],
        "coarse",
        parser=lambda path, structure: _parsed_static(int(path.name[-3:])),
    )
    assert incomplete.decision.state == "operational_inconclusive"
    with pytest.raises(ValueError, match="static_pass"):
        prepare_berry_inputs(
            tmp_path / "berry-incomplete",
            structures,
            incomplete.decision,
            config=object(),
            berry_writer=lambda *args: None,
        )

    metallic = collect_static_path(
        structures,
        work_dirs,
        "coarse",
        parser=lambda path, structure: _parsed_static(
            int(path.name[-3:]), gap=0.009 if path.name == "image-004" else 0.5
        ),
    )
    assert metallic.decision.state == "path_metallic_stop"

    complete = collect_static_path(
        structures,
        work_dirs,
        "coarse",
        parser=lambda path, structure: _parsed_static(int(path.name[-3:])),
    )
    written: list[Path] = []

    def fake_berry_writer(structure, destination, config) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        written.append(destination)

    prepared_dirs = prepare_berry_inputs(
        tmp_path / "berry-complete",
        structures,
        complete.decision,
        config=object(),
        berry_writer=fake_berry_writer,
    )
    assert len(prepared_dirs) == 10
    assert written == list(prepared_dirs)


def _passing_static_collection(count: int, level: str):
    structures = tuple(_toy_endpoints()[0] for _ in range(count))
    rows = [
        ParsedStatic(
            status="complete",
            reason_codes=(),
            energy_eV_atom=-5.0 + 0.01 * (1.0 - index / (count - 1)) ** 2,
            gap_eV=0.5,
            final_structure=structures[index],
        )
        for index in range(count)
    ]
    collection = collect_static_path(
        structures,
        tuple(Path(f"image-{index:03d}") for index in range(count)),
        level,
        parser=lambda path, structure: rows[int(path.name[-3:])],
    )
    assert collection.decision.state == "static_pass"
    return structures, collection


@pytest.mark.parametrize(("level", "count"), [("coarse", 10), ("dense", 19)])
def test_berry_collection_produces_terminal_scientific_decision(
    level: str, count: int
) -> None:
    structures, static = _passing_static_collection(count, level)
    work_dirs = tuple(Path(f"image-{index:03d}") for index in range(count))

    def parser(path: Path, structure: Structure) -> ParsedBerry:
        fraction = int(path.name[-3:]) / (count - 1)
        return ParsedBerry(
            status="complete",
            reason_codes=(),
            p_elec=(-0.2 * fraction, 0.0, 0.0),
            p_ion=(0.0, 0.0, 0.0),
        )

    result = collect_berry_path(
        structures,
        work_dirs,
        static,
        level,
        parser=parser,
    )
    assert result.decision.state == "smidt_fast_pass"
    assert result.decision.metrics["image_count"] == count
    assert result.decision.metrics["spontaneous_polarization_uC_cm2"] > 0.1


def test_berry_collection_keeps_parser_failure_operational() -> None:
    structures, static = _passing_static_collection(10, "coarse")
    work_dirs = tuple(Path(f"image-{index:03d}") for index in range(10))

    def parser(path: Path, structure: Structure) -> ParsedBerry:
        if path.name == "image-004":
            return ParsedBerry("operational_inconclusive", ("missing_p_ion",))
        return ParsedBerry("complete", (), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    result = collect_berry_path(
        structures,
        work_dirs,
        static,
        "coarse",
        parser=parser,
    )
    assert result.decision.state == "operational_inconclusive"
    assert result.decision.reason_codes == ("missing_p_ion",)


def _load_cli():
    path = STAGE17_ROOT / "scripts/run_r3_smidt_fast_path.py"
    spec = importlib.util.spec_from_file_location("run_r3_smidt_fast_path_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_prepare_is_no_clobber_and_hash_chains_attempts(tmp_path: Path, monkeypatch) -> None:
    cli = _load_cli()
    source_bytes = b'{"material_id":"mp-aaacrlli"}\n'

    def fake_prepare_path_inputs(**kwargs):
        campaign = Path(kwargs["output_root"]) / "smidt-fast-test"
        path_root = campaign / "paths/coarse"
        path_root.mkdir(parents=True, exist_ok=True)
        source = path_root / "source_manifest.json"
        if not source.exists():
            source.write_bytes(source_bytes)
        structures = tuple(_toy_endpoints()[0] for _ in range(10))
        work = tuple(
            campaign / f"work/coarse/static/image-{index:03d}" for index in range(10)
        )
        for directory in work:
            directory.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            material_id="mp-aaacrlli",
            campaign_dir=campaign,
            refinement_level="coarse",
            structures=structures,
            static_work_dirs=work,
            source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )

    monkeypatch.setattr(cli, "prepare_path_inputs", fake_prepare_path_inputs)
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(cli, "write_switching_static", lambda *args: None)
    argv = [
        "prepare",
        "--candidate",
        "mp-aaacrlli",
        "--output-root",
        str(tmp_path),
    ]
    assert cli.main(argv, stage17_root=STAGE17_ROOT) == 0
    assert cli.main(argv, stage17_root=STAGE17_ROOT) == 0
    campaign = tmp_path / "smidt-fast-test"
    attempts = sorted((campaign / "attempts").iterdir())
    assert [path.name for path in attempts] == ["attempt-01", "attempt-02"]
    assert (campaign / "paths/coarse/source_manifest.json").read_bytes() == source_bytes
    for attempt in attempts:
        assert (attempt / "request.json").is_file()
        assert (attempt / "source_manifest.json").is_file()
        assert (attempt / "stdout.txt").is_file()
        assert (attempt / "stderr.txt").is_file()
        assert (attempt / "result.json").is_file()
        assert (attempt / "COMPLETED").is_file()
    second_request = json.loads((attempts[1] / "request.json").read_text())
    assert second_request["resume_from"]["attempt"] == "attempt-01"
    assert len(second_request["resume_from"]["terminal_record_sha256"]) == 64


def test_cli_dry_run_never_dispatches_and_failed_launch_gets_failed_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    cli = _load_cli()
    campaign = tmp_path / "smidt-fast-test"
    source = campaign / "paths/coarse/source_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"material_id":"mp-aaacrlli"}\n', encoding="utf-8")
    for index in range(10):
        (campaign / f"work/coarse/static/image-{index:03d}").mkdir(parents=True)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"nodes": {"g4": {}}, "probeable_nodes": ["g4"]}), encoding="utf-8")
    policy = {
        "nodes": {"g4": {}},
        "probeable_nodes": ["g4"],
        "max_tasks_per_submission": 40,
    }
    monkeypatch.setattr(cli, "load_execution_policy", lambda path: policy)
    monkeypatch.setattr(cli, "collect_live_probe", lambda *args, **kwargs: {"node": "g4"})
    monkeypatch.setattr(cli, "load_smoke_records", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli,
        "plan_submission",
        lambda *args, **kwargs: SimpleNamespace(
            stage="static", profile="gpu", assignments=tuple(range(10))
        ),
    )
    monkeypatch.setattr(
        cli,
        "dispatch_submission",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    dry = [
        "dry-run",
        "--campaign",
        str(campaign),
        "--stage",
        "static",
        "--policy",
        str(policy_path),
    ]
    assert cli.main(dry, stage17_root=STAGE17_ROOT) == 0

    monkeypatch.setattr(
        cli,
        "plan_submission",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no admitted smoke")),
    )
    launch = [
        "launch",
        "--campaign",
        str(campaign),
        "--stage",
        "static",
        "--policy",
        str(policy_path),
    ]
    assert cli.main(launch, stage17_root=STAGE17_ROOT) == 2
    latest = sorted((campaign / "attempts").iterdir())[-1]
    assert (latest / "failure.json").is_file()
    assert (latest / "FAILED").is_file()
