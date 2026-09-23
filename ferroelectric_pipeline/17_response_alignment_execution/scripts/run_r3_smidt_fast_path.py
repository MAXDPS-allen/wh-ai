#!/usr/bin/env python3
"""Prepare, inspect, launch, and collect the Stage 17 Smidt fast path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence


STAGE17_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = STAGE17_ROOT.parent
STAGE16_ROOT = PIPELINE_ROOT / "16_method_validation_and_completion"
for source in (STAGE17_ROOT / "src", STAGE16_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from dfc_dft.contracts import load_config
from dfc_dft.vasp_inputs import write_switching_berry, write_switching_static
from dfc_dft.vasp_outputs import parse_completion
from stage17.run_contract import next_attempt_path
from stage17.smidt_campaign import (
    StaticCollection,
    collect_berry_path,
    collect_static_path,
    prepare_berry_inputs,
    prepare_path_inputs,
)
from stage17.smidt_cluster import (
    TaskSpec,
    collect_live_probe,
    dispatch_submission,
    load_execution_policy,
    plan_submission,
    render_environment_exports,
)
from stage17.smidt_fast_path import FastPathDecision, StaticObservation


DEFAULT_ENDPOINT_RUN = (
    STAGE17_ROOT
    / "runs/R3_candidate_screening/endpoint-gate-parents-78587656accad94e"
)
DEFAULT_GATE_RESULTS = DEFAULT_ENDPOINT_RUN / "endpoint_relax_gate_results.json"
DEFAULT_CONFIG = STAGE16_ROOT / "configs/born_assisted_dft_funnel.json"
DEFAULT_POLICY = STAGE17_ROOT / "configs/r3_smidt_fast_path_execution_policy.json"
DEFAULT_OUTPUT_ROOT = STAGE17_ROOT / "runs/R3_candidate_screening"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _find_campaign(output_root: Path, candidate: str, level: str = "coarse") -> Path | None:
    for campaign in sorted(Path(output_root).glob("smidt-fast-*")):
        manifest = campaign / f"paths/{level}/source_manifest.json"
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("material_id") == candidate and value.get("refinement_level") == level:
            return campaign
    return None


def _load_structures(campaign: Path, level: str) -> tuple:
    from pymatgen.core import Structure

    paths = sorted((campaign / f"paths/{level}/structures").glob("image-*.json"))
    return tuple(Structure.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths)


def _load_prepared(campaign: Path, level: str) -> SimpleNamespace:
    structures = _load_structures(campaign, level)
    static_dirs = tuple(
        campaign / f"work/{level}/static/image-{index:03d}" for index in range(len(structures))
    )
    manifest = campaign / f"paths/{level}/source_manifest.json"
    return SimpleNamespace(
        material_id=json.loads(manifest.read_text(encoding="utf-8"))["material_id"],
        campaign_dir=campaign,
        refinement_level=level,
        structures=structures,
        static_work_dirs=static_dirs,
        source_manifest_sha256=_sha256(manifest),
    )


def _terminal_record(attempt: Path) -> Path | None:
    for marker, record in (
        ("DISPATCHED", "dispatch.json"),
        ("FAILED", "failure.json"),
        ("COMPLETED", "result.json"),
        ("SUCCESS", "result.json"),
    ):
        if (attempt / marker).is_file() and (attempt / record).is_file():
            return attempt / record
    return None


def _output_root_for_request(campaign: Path, stage17_root: Path) -> str:
    try:
        relative = campaign.resolve().relative_to(stage17_root.resolve())
        text = relative.as_posix()
        if text.startswith("runs/R3_candidate_screening/"):
            return text
    except ValueError:
        pass
    return "runs/R3_candidate_screening/smidt-fast-test"


def _begin_attempt(
    campaign: Path, command: Sequence[str], stage17_root: Path, *, source_level: str = "coarse"
) -> Path:
    attempts_root = campaign / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt_path(attempts_root)
    attempt.mkdir(exist_ok=False)
    source = campaign / f"paths/{source_level}/source_manifest.json"
    if not source.is_file():
        source = campaign / "paths/coarse/source_manifest.json"
    source_sha = _sha256(source)
    shutil.copyfile(source, attempt / "source_manifest.json")
    previous = sorted(path for path in attempts_root.glob("attempt-*") if path != attempt)
    resume = None
    if previous:
        record = _terminal_record(previous[-1])
        if record is not None:
            resume = {
                "attempt": previous[-1].name,
                "terminal_record": record.name,
                "terminal_record_sha256": _sha256(record),
            }
    now = _utc_now().isoformat()
    request = {
        "schema_version": 1,
        "request_id": f"{attempt.name}-{hashlib.sha256(' '.join(command).encode()).hexdigest()[:12]}",
        "created_utc": now,
        "inputs": {
            "source_manifest": {
                "path": str(source),
                "sha256": source_sha,
            }
        },
        "command": list(command),
        "environment": "fe_dft",
        "node": "local",
        "resources": {"cpu_cores": 1, "ram_gib": 1, "gpu_count": 0},
        "output_roots": [_output_root_for_request(campaign, stage17_root)],
    }
    if resume is not None:
        request["resume_from"] = resume
    _write_new(attempt / "request.json", _canonical_json(request))
    _write_new(attempt / "stdout.txt", b"")
    _write_new(attempt / "stderr.txt", b"")
    return attempt


def _finish_attempt(attempt: Path, status: str, summary: dict) -> None:
    request_sha = _sha256(attempt / "request.json")
    now = _utc_now().isoformat()
    if status == "dispatched":
        record_name = "dispatch.json"
        marker = "DISPATCHED"
        record = {
            "schema_version": 1,
            "request_sha256": request_sha,
            "status": "dispatched",
            "work_roots": summary.pop("work_roots"),
            "dispatched_utc": now,
            "reason_codes": [],
            "summary": summary,
        }
    else:
        failed = status == "failed"
        record_name = "failure.json" if failed else "result.json"
        marker = "FAILED" if failed else "COMPLETED"
        record = {
            "schema_version": 1,
            "request_sha256": request_sha,
            "status": "failed" if failed else "success",
            "outputs": {},
            "started_utc": json.loads((attempt / "request.json").read_text())["created_utc"],
            "finished_utc": now,
            "exit_code": 1 if failed else 0,
            "reason_codes": list(summary.get("reason_codes", [])),
            "summary": summary,
        }
    _write_new(attempt / record_name, _canonical_json(record))
    _write_new(attempt / marker, b"\n")


def _latest_snapshot(campaign: Path, level: str, stage: str) -> Path:
    paths = sorted((campaign / f"snapshots/{level}").glob(f"{stage}-*"))
    if not paths:
        raise ValueError(f"no {level} {stage} snapshot exists")
    return paths[-1]


def _new_snapshot(campaign: Path, level: str, stage: str) -> Path:
    root = campaign / f"snapshots/{level}"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.rsplit("-", 1)[-1]) for path in root.glob(f"{stage}-[0-9][0-9][0-9]")]
    path = root / f"{stage}-{max(numbers, default=0) + 1:03d}"
    path.mkdir(exist_ok=False)
    return path


def _publish_static_snapshot(campaign: Path, level: str, collection: StaticCollection) -> Path:
    snapshot = _new_snapshot(campaign, level, "static")
    value = {
        "schema_version": 1,
        "refinement_level": level,
        "observations": [
            {
                "image_index": row.image_index,
                "status": row.status,
                "energy_eV_atom": row.energy_eV_atom,
                "gap_eV": row.gap_eV,
                "reason_codes": list(row.reason_codes),
            }
            for row in collection.observations
        ],
        "decision": collection.decision.to_dict(),
    }
    _write_new(snapshot / "result.json", _canonical_json(value))
    _write_new(snapshot / "manifest.sha256", f"{_sha256(snapshot / 'result.json')}  result.json\n".encode())
    return snapshot


def _load_static_snapshot(campaign: Path, level: str) -> StaticCollection:
    value = json.loads((_latest_snapshot(campaign, level, "static") / "result.json").read_text())
    observations = tuple(
        StaticObservation(
            int(row["image_index"]),
            str(row["status"]),
            row["energy_eV_atom"],
            row["gap_eV"],
            tuple(row["reason_codes"]),
        )
        for row in value["observations"]
    )
    decision_value = value["decision"]
    decision = FastPathDecision(
        decision_value["state"], tuple(decision_value["reason_codes"]), decision_value["metrics"]
    )
    return StaticCollection(observations, decision)


def _publish_berry_snapshot(campaign: Path, level: str, collection) -> Path:
    snapshot = _new_snapshot(campaign, level, "berry")
    value = {
        "schema_version": 1,
        "refinement_level": level,
        "decision": collection.decision.to_dict(),
        "berry_status": [row.status for row in collection.observations],
    }
    _write_new(snapshot / "result.json", _canonical_json(value))
    _write_new(snapshot / "manifest.sha256", f"{_sha256(snapshot / 'result.json')}  result.json\n".encode())
    return snapshot


def load_smoke_records(policy: dict, stage17_root: Path) -> dict:
    records = {}
    for node, node_policy in policy.get("nodes", {}).items():
        for profile, profile_policy in node_policy.items():
            raw_path = profile_policy.get("smoke_record_path")
            expected = profile_policy.get("smoke_record_sha256")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = stage17_root / path
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"{node} {profile} smoke record hash mismatch")
            records[(node, profile)] = json.loads(path.read_text(encoding="utf-8"))
    return records


def _task_specs(campaign: Path, level: str, stage: str, stage17_root: Path) -> tuple[TaskSpec, ...]:
    directories = sorted((campaign / f"work/{level}/{stage}").glob("image-*"))
    tasks = []
    for directory in directories:
        try:
            relative = directory.resolve().relative_to(stage17_root.resolve()).as_posix()
        except ValueError:
            relative = f"runs/R3_candidate_screening/{campaign.name}/work/{level}/{stage}/{directory.name}"
        tasks.append(TaskSpec(directory.name, stage, relative))
    return tuple(tasks)


def _probe_for_plan(policy: dict, runner, now: datetime) -> dict:
    probes = {}
    for node in sorted(policy.get("nodes", {})):
        probes[node] = collect_live_probe(node, policy, runner=runner, now=now)
    return probes


def _prepare_command(args, stage17_root: Path, level: str) -> tuple[Path, dict]:
    output_root = Path(args.output_root)
    existing = _find_campaign(output_root, args.candidate, level)
    if existing is not None:
        prepared = _load_prepared(existing, level)
        reused = True
    else:
        prepared = prepare_path_inputs(
            output_root=output_root,
            endpoint_run=Path(args.endpoint_run),
            gate_results_path=Path(args.gate_results),
            candidate_id=args.candidate,
            config=load_config(Path(args.config)),
            refinement_level=level,
            static_writer=write_switching_static,
        )
        reused = False
    return prepared.campaign_dir, {
        "candidate": args.candidate,
        "refinement_level": level,
        "image_count": len(prepared.structures),
        "campaign": str(prepared.campaign_dir),
        "reused_preparation": reused,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare", "prepare-dense"):
        child = sub.add_parser(name)
        child.add_argument("--candidate", required=True)
        child.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        child.add_argument("--endpoint-run", default=str(DEFAULT_ENDPOINT_RUN))
        child.add_argument("--gate-results", default=str(DEFAULT_GATE_RESULTS))
        child.add_argument("--config", default=str(DEFAULT_CONFIG))
        if name == "prepare-dense":
            child.add_argument("--resume-from", required=True)

    for name in ("dry-run", "launch"):
        child = sub.add_parser(name)
        child.add_argument("--campaign", type=Path, required=True)
        child.add_argument("--stage", choices=("static", "berry"), required=True)
        child.add_argument("--refinement-level", choices=("coarse", "dense"), default="coarse")
        child.add_argument("--policy", type=Path, default=DEFAULT_POLICY)

    for name in ("collect-static", "prepare-berry", "collect-berry"):
        child = sub.add_parser(name)
        child.add_argument("--campaign", type=Path, required=True)
        child.add_argument("--refinement-level", choices=("coarse", "dense"), default="coarse")
        child.add_argument("--config", default=str(DEFAULT_CONFIG))

    probe = sub.add_parser("probe")
    probe.add_argument("--campaign", type=Path, required=True)
    probe.add_argument("--policy", type=Path, default=DEFAULT_POLICY)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--campaign", type=Path, required=True)
    smoke.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    smoke.add_argument("--node", required=True)
    smoke.add_argument("--profile", choices=("gpu", "cpu"), required=True)
    smoke.add_argument("--input-dir", type=Path, required=True)
    return parser


def _run_smoke(args, policy: dict, runner, attempt: Path) -> dict:
    input_dir = args.input_dir.resolve()
    required = ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "input_manifest.json")
    if any(not (input_dir / name).is_file() for name in required):
        raise ValueError("smoke input directory is incomplete")
    work = attempt / "smoke-work"
    work.mkdir()
    for name in required:
        shutil.copy2(input_dir / name, work / name)
    executable = policy["executables"][args.profile]
    launcher = policy.get("launchers", {}).get(args.profile, ["mpirun", "-np", "1"])
    command = " ".join([*(str(value) for value in launcher), str(executable)])
    gpu = "CUDA_VISIBLE_DEVICES=0 " if args.profile == "gpu" else ""
    setup = " && ".join(render_environment_exports(args.profile, policy))
    remote = f"cd {shlex.quote(str(work))} && {setup} && {gpu}{command} > vasp.out 2>&1"
    started = _utc_now()
    completed = runner(
        ["ssh", "-o", "BatchMode=yes", args.node, remote],
        check=False,
        capture_output=True,
        text=True,
        timeout=int(policy["timeouts_seconds"]["static"]),
    )
    completion = parse_completion(work / "OUTCAR")
    record = {
        "schema_version": 1,
        "node": args.node,
        "profile": args.profile,
        "created_utc": started.isoformat(),
        "input_manifest_sha256": _sha256(work / "input_manifest.json"),
        "executable": executable,
        "launcher": " ".join(str(value) for value in launcher),
        "environment": "fe_gpu" if args.profile == "gpu" else "fe_dft",
        "exit_code": completed.returncode,
        "normally_terminated": completion.normally_terminated,
        "duration_seconds": (_utc_now() - started).total_seconds(),
        "output_sha256": _sha256(work / "OUTCAR") if (work / "OUTCAR").is_file() else None,
    }
    record_path = args.campaign / "smokes" / f"{args.node}-{args.profile}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_new(record_path, _canonical_json(record))
    if completed.returncode != 0 or not completion.normally_terminated:
        raise ValueError("smoke calculation did not terminate normally")
    return {"smoke_record": str(record_path), "smoke_record_sha256": _sha256(record_path)}


def main(argv: Sequence[str] | None = None, *, stage17_root: Path = STAGE17_ROOT, runner=subprocess.run) -> int:
    args = _parser().parse_args(argv)
    command = list(argv if argv is not None else sys.argv[1:])
    attempt: Path | None = None
    try:
        if args.command in {"prepare", "prepare-dense"}:
            level = "coarse" if args.command == "prepare" else "dense"
            if level == "dense":
                coarse_result = Path(args.resume_from)
                value = json.loads(coarse_result.read_text(encoding="utf-8"))
                state = value.get("decision", value).get("state")
                if state != "needs_dense_path":
                    raise ValueError("prepare-dense requires a coarse needs_dense_path result")
            campaign, summary = _prepare_command(args, stage17_root, level)
            attempt = _begin_attempt(campaign, command, stage17_root, source_level=level)
            _finish_attempt(attempt, "success", summary)
        elif args.command in {"dry-run", "launch"}:
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root, source_level=args.refinement_level)
            policy = load_execution_policy(args.policy)
            now = _utc_now()
            probes = _probe_for_plan(policy, runner, now)
            smokes = load_smoke_records(policy, stage17_root)
            tasks = _task_specs(campaign, args.refinement_level, args.stage, stage17_root)
            plan = plan_submission(args.stage, tasks, policy, probes, smokes, now=now)
            summary = {
                "stage": args.stage,
                "profile": plan.profile,
                "task_count": len(plan.assignments),
                "refinement_level": args.refinement_level,
            }
            if args.command == "dry-run":
                _finish_attempt(attempt, "success", summary)
            else:
                dispatch = dispatch_submission(plan, attempt, policy, runner=runner)
                output_root = _output_root_for_request(campaign, stage17_root)
                dispatch["work_roots"] = [
                    f"{output_root}/work/{args.refinement_level}/{args.stage}"
                ]
                _finish_attempt(attempt, "dispatched", dispatch)
                summary = dispatch
        elif args.command == "collect-static":
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root, source_level=args.refinement_level)
            structures = _load_structures(campaign, args.refinement_level)
            work = tuple(
                campaign / f"work/{args.refinement_level}/static/image-{index:03d}"
                for index in range(len(structures))
            )
            collection = collect_static_path(structures, work, args.refinement_level)
            snapshot = _publish_static_snapshot(campaign, args.refinement_level, collection)
            summary = {"snapshot": str(snapshot), **collection.decision.to_dict()}
            _finish_attempt(attempt, "success", summary)
        elif args.command == "prepare-berry":
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root, source_level=args.refinement_level)
            structures = _load_structures(campaign, args.refinement_level)
            static = _load_static_snapshot(campaign, args.refinement_level)
            root = campaign / f"work/{args.refinement_level}/berry"
            if root.exists():
                directories = tuple(sorted(root.glob("image-*")))
            else:
                directories = prepare_berry_inputs(
                    root,
                    structures,
                    static.decision,
                    config=load_config(Path(args.config)),
                    berry_writer=write_switching_berry,
                )
            summary = {"state": "berry_prepared", "task_count": len(directories)}
            _finish_attempt(attempt, "success", summary)
        elif args.command == "collect-berry":
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root, source_level=args.refinement_level)
            structures = _load_structures(campaign, args.refinement_level)
            static = _load_static_snapshot(campaign, args.refinement_level)
            work = tuple(
                campaign / f"work/{args.refinement_level}/berry/image-{index:03d}"
                for index in range(len(structures))
            )
            collection = collect_berry_path(
                structures, work, static, args.refinement_level
            )
            snapshot = _publish_berry_snapshot(campaign, args.refinement_level, collection)
            summary = {"snapshot": str(snapshot), **collection.decision.to_dict()}
            _finish_attempt(attempt, "success", summary)
        elif args.command == "probe":
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root)
            policy = load_execution_policy(args.policy)
            now = _utc_now()
            probes = {
                node: collect_live_probe(node, policy, runner=runner, now=now)
                for node in policy.get("probeable_nodes", [])
            }
            root = campaign / "probes" / now.strftime("%Y%m%dT%H%M%SZ")
            root.mkdir(parents=True, exist_ok=False)
            for node, value in probes.items():
                _write_new(root / f"{node}.json", _canonical_json(value))
            summary = {"probe_root": str(root), "nodes": sorted(probes)}
            _finish_attempt(attempt, "success", summary)
        else:
            campaign = args.campaign.resolve()
            attempt = _begin_attempt(campaign, command, stage17_root)
            policy = load_execution_policy(args.policy)
            summary = _run_smoke(args, policy, runner, attempt)
            _finish_attempt(attempt, "success", summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        if attempt is not None and _terminal_record(attempt) is None:
            _finish_attempt(
                attempt,
                "failed",
                {"reason_codes": [type(exc).__name__], "message": str(exc)},
            )
            (attempt / "stderr.txt").write_text(str(exc) + "\n", encoding="utf-8")
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
