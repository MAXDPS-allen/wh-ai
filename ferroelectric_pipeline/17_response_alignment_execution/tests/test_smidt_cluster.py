from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


STAGE17_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE17_ROOT / "src"))

from stage17.smidt_cluster import (  # noqa: E402
    TaskSpec,
    collect_live_probe,
    dispatch_submission,
    load_execution_policy,
    plan_submission,
    render_environment_exports,
    retry_allowed,
)


NOW = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)


def _smoke(profile: str) -> dict:
    return {
        "schema_version": 1,
        "node": "g4",
        "profile": profile,
        "created_utc": "2026-09-23T07:30:00+00:00",
        "input_manifest_sha256": "a" * 64,
        "executable": "/gpu/vasp" if profile == "gpu" else "/cpu/vasp",
        "launcher": "hpcx-mpirun-np1",
        "environment": "fe_gpu" if profile == "gpu" else "fe_dft",
        "exit_code": 0,
        "normally_terminated": True,
        "duration_seconds": 12.0,
        "output_sha256": "b" * 64,
    }


def _digest(value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _policy() -> dict:
    gpu_smoke = _smoke("gpu")
    cpu_smoke = _smoke("cpu")
    return {
        "schema_version": 1,
        "scope": "R3_smidt_fast_path",
        "probeable_nodes": ["g1", "g3", "g4", "g6", "g7"],
        "max_tasks_per_submission": 40,
        "probe_max_age_seconds": 300,
        "timeouts_seconds": {"static": 43200, "berry": 86400},
        "operational_retries": 1,
        "lock_path": "/tmp/stage17_smidt_fast.lock",
        "executables": {"gpu": "/gpu/vasp", "cpu": "/cpu/vasp"},
        "environment_setup": {
            "gpu": {
                "prepend_path": ["/gpu/mpi/bin"],
                "prepend_ld_library_path": ["/gpu/qd/lib", "/gpu/cuda/lib64"],
                "variables": {"OMP_NUM_THREADS": "4", "OMPI_MCA_pml": "ob1"},
            },
            "cpu": {
                "prepend_path": ["/cpu/mpi/bin"],
                "prepend_ld_library_path": ["/cpu/qd/lib"],
                "variables": {"OMP_NUM_THREADS": "1"},
            },
        },
        "nodes": {
            "g4": {
                "gpu": {
                    "smoke_record_sha256": _digest(gpu_smoke),
                    "max_concurrency": 4,
                    "cpu_cores": 16,
                    "ram_gib": 64,
                    "gpu_ids": [0, 1, 2, 3],
                },
                "cpu": {
                    "smoke_record_sha256": _digest(cpu_smoke),
                    "max_concurrency": 2,
                    "cpu_cores": 16,
                    "ram_gib": 64,
                    "gpu_ids": [],
                },
            }
        },
    }


def _probe(captured: datetime = NOW) -> dict:
    return {
        "schema_version": 1,
        "node": "g4",
        "captured_utc": captured.isoformat(),
        "logical_cpus": 128,
        "available_ram_gib": 300.0,
        "gpus": [
            {
                "index": index,
                "free_memory_gib": 39.0,
                "utilization_percent": 0.0,
                "compute_processes": [],
            }
            for index in range(4)
        ],
        "executables": {
            "gpu": {"path": "/gpu/vasp", "available": True},
            "cpu": {"path": "/cpu/vasp", "available": True},
        },
        "stage17_locks": [],
        "live_dfpt_processes": [],
    }


def _tasks(stage: str, count: int) -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id=f"image-{index:03d}",
            stage=stage,
            work_dir=f"runs/R3_candidate_screening/smidt-fast-test/work/coarse/{stage}/image-{index:03d}",
        )
        for index in range(count)
    ]


def test_repository_policy_is_dedicated_and_deny_by_default() -> None:
    policy = load_execution_policy(STAGE17_ROOT / "configs/r3_smidt_fast_path_execution_policy.json")
    assert policy["scope"] == "R3_smidt_fast_path"
    assert policy["max_tasks_per_submission"] == 40
    assert policy["nodes"] == {}


@pytest.mark.parametrize(("stage", "profile"), [("static", "gpu"), ("berry", "cpu")])
def test_planner_routes_static_to_gpu_and_berry_to_cpu(stage: str, profile: str) -> None:
    plan = plan_submission(
        stage,
        _tasks(stage, 10),
        _policy(),
        {"g4": _probe()},
        {("g4", profile): _smoke(profile)},
        now=NOW,
    )
    assert plan.profile == profile
    assert len(plan.assignments) == 10
    assert {assignment.node for assignment in plan.assignments} == {"g4"}
    if stage == "static":
        assert {assignment.gpu_id for assignment in plan.assignments} == {0, 1, 2, 3}
    else:
        assert {assignment.gpu_id for assignment in plan.assignments} == {None}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(captured_utc=(NOW - timedelta(seconds=301)).isoformat()), "stale"),
        (lambda p: p.update(stage17_locks=["/tmp/live.lock"]), "lock"),
        (lambda p: p.update(live_dfpt_processes=[{"pid": 1, "cwd": "/dfpt"}]), "DFPT"),
    ],
)
def test_planner_rejects_stale_or_busy_probe(mutate, message: str) -> None:
    probe = _probe()
    mutate(probe)
    with pytest.raises(ValueError, match=message):
        plan_submission(
            "static",
            _tasks("static", 1),
            _policy(),
            {"g4": probe},
            {("g4", "gpu"): _smoke("gpu")},
            now=NOW,
        )


def test_planner_rejects_smoke_hash_or_executable_mismatch() -> None:
    smoke = _smoke("gpu")
    smoke["output_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="smoke"):
        plan_submission(
            "static",
            _tasks("static", 1),
            _policy(),
            {"g4": _probe()},
            {("g4", "gpu"): smoke},
            now=NOW,
        )


def test_planner_rejects_submission_cap_and_path_escape() -> None:
    with pytest.raises(ValueError, match="40"):
        plan_submission(
            "static",
            _tasks("static", 41),
            _policy(),
            {"g4": _probe()},
            {("g4", "gpu"): _smoke("gpu")},
            now=NOW,
        )
    escaped = [TaskSpec("escape", "static", "../../16_method_validation_and_completion")]
    with pytest.raises(ValueError, match="work_dir"):
        plan_submission(
            "static",
            escaped,
            _policy(),
            {"g4": _probe()},
            {("g4", "gpu"): _smoke("gpu")},
            now=NOW,
        )


def test_retry_is_limited_to_one_operational_failure() -> None:
    policy = _policy()
    assert retry_allowed("operational", 0, policy) is True
    assert retry_allowed("operational", 1, policy) is False
    assert retry_allowed("scientific", 0, policy) is False


def test_environment_exports_are_shared_by_smoke_and_queue() -> None:
    lines = render_environment_exports("gpu", _policy())
    assert 'export PATH=/gpu/mpi/bin:"${PATH:-}"' in lines
    assert 'export LD_LIBRARY_PATH=/gpu/qd/lib:/gpu/cuda/lib64:"${LD_LIBRARY_PATH:-}"' in lines
    assert "export OMP_NUM_THREADS=4" in lines
    assert "export OMPI_MCA_pml=ob1" in lines


def test_dispatch_writes_scripts_and_uses_injected_runner(tmp_path: Path) -> None:
    plan = plan_submission(
        "static",
        _tasks("static", 3),
        _policy(),
        {"g4": _probe()},
        {("g4", "gpu"): _smoke("gpu")},
        now=NOW,
    )
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    dispatch = dispatch_submission(plan, tmp_path, _policy(), runner=fake_runner)
    assert len(calls) == 1
    assert calls[0][:3] == ["ssh", "-o", "BatchMode=yes"]
    queue_path = tmp_path / "dispatch/g4-static-queue.sh"
    assert queue_path.is_file()
    assert "export LD_LIBRARY_PATH=/gpu/qd/lib:/gpu/cuda/lib64" in queue_path.read_text()
    assert dispatch["status"] == "dispatched"
    assert dispatch["nodes"]["g4"]["pid"] == 12345


def test_live_probe_uses_read_only_ssh_and_builds_fresh_schema() -> None:
    commands: list[list[str]] = []

    def fake_runner(command, **kwargs):
        commands.append(list(command))
        remote = command[-1]
        if "_NPROCESSORS_ONLN" in remote:
            output = "128\n"
        elif "MemAvailable" in remote:
            output = str(300 * 1024 * 1024) + "\n"
        elif "test -x /gpu/vasp" in remote or "test -x /cpu/vasp" in remote:
            output = "1\n"
        elif "test -e /tmp/stage17_smidt_fast.lock" in remote:
            output = "0\n"
        elif "--query-gpu=index,memory.free,utilization.gpu" in remote:
            output = "0, 39936, 0\n1, 39936, 0\n"
        elif "--query-compute-apps=pid" in remote:
            output = ""
        elif "pgrep -f" in remote:
            output = ""
        else:
            raise AssertionError(remote)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    probe = collect_live_probe("g4", _policy(), runner=fake_runner, now=NOW)
    assert probe["schema_version"] == 1
    assert probe["captured_utc"] == NOW.isoformat()
    assert probe["logical_cpus"] == 128
    assert probe["available_ram_gib"] == pytest.approx(300.0)
    assert [gpu["index"] for gpu in probe["gpus"]] == [0, 1]
    assert probe["stage17_locks"] == []
    assert probe["live_dfpt_processes"] == []
    assert all(command[:3] == ["ssh", "-o", "BatchMode=yes"] for command in commands)
    lock_commands = [command[-1] for command in commands if "stage17_smidt_fast.lock" in command[-1]]
    assert len(lock_commands) == 1
    assert "flock -n" in lock_commands[0]
    process_commands = [command[-1] for command in commands if "pgrep -f" in command[-1]]
    assert len(process_commands) == 1
    assert "pgrep -f '[v]asp_std|[P]Wmat'" in process_commands[0]
