"""Deny-by-default planning and dispatch for Smidt static/Berry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import Callable, Mapping, Sequence


STAGE17_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    stage: str
    work_dir: str


@dataclass(frozen=True)
class Assignment:
    task_id: str
    stage: str
    work_dir: str
    node: str
    profile: str
    worker_index: int
    gpu_id: int | None


@dataclass(frozen=True)
class SubmissionPlan:
    stage: str
    profile: str
    executable: str
    assignments: tuple[Assignment, ...]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def load_execution_policy(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Smidt execution policy") from exc
    required = {
        "schema_version",
        "scope",
        "probeable_nodes",
        "max_tasks_per_submission",
        "probe_max_age_seconds",
        "timeouts_seconds",
        "operational_retries",
        "lock_path",
        "executables",
        "nodes",
    }
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("Smidt execution policy is missing required fields")
    if value["schema_version"] != 1 or value["scope"] != "R3_smidt_fast_path":
        raise ValueError("Smidt execution policy identity is invalid")
    if value["max_tasks_per_submission"] != 40:
        raise ValueError("Smidt execution policy task cap must be 40")
    if not isinstance(value["nodes"], dict):
        raise ValueError("Smidt execution policy nodes must be an object")
    return value


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("probe captured_utc is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("probe captured_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("probe captured_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _safe_work_dir(task: TaskSpec) -> PurePosixPath:
    if task.stage not in {"static", "berry"}:
        raise ValueError("task stage must be static or berry")
    path = PurePosixPath(task.work_dir)
    if (
        path.is_absolute()
        or "\\" in task.work_dir
        or ".." in path.parts
        or len(path.parts) < 7
        or path.parts[:2] != ("runs", "R3_candidate_screening")
        or "work" not in path.parts
        or task.stage not in path.parts
    ):
        raise ValueError("task work_dir is outside the Stage 17 R3 campaign")
    return path


def _admission_reasons(
    node: str,
    profile: str,
    node_policy: Mapping[str, object],
    policy: Mapping[str, object],
    probe: Mapping[str, object] | None,
    smoke: Mapping[str, object] | None,
    now: datetime,
) -> list[str]:
    reasons: list[str] = []
    profile_policy = node_policy.get(profile)
    if not isinstance(profile_policy, Mapping):
        return [f"{node} profile not admitted"]
    if probe is None or probe.get("schema_version") != 1 or probe.get("node") != node:
        return [f"{node} probe is missing or invalid"]
    try:
        age = (now.astimezone(timezone.utc) - _parse_utc(probe.get("captured_utc"))).total_seconds()
    except ValueError:
        age = float("inf")
    if age < 0 or age > int(policy["probe_max_age_seconds"]):
        reasons.append(f"{node} probe is stale")
    if probe.get("stage17_locks"):
        reasons.append(f"{node} has an active Stage 17 lock")
    if probe.get("live_dfpt_processes"):
        reasons.append(f"{node} has a live DFPT process")
    executable = str(policy["executables"][profile])  # type: ignore[index]
    executable_probe = probe.get("executables", {})
    if not isinstance(executable_probe, Mapping):
        reasons.append(f"{node} executable probe is invalid")
    else:
        row = executable_probe.get(profile)
        if not isinstance(row, Mapping) or row.get("path") != executable or row.get("available") is not True:
            reasons.append(f"{node} {profile} executable is unavailable")
    if float(probe.get("available_ram_gib", 0.0)) < float(profile_policy.get("ram_gib", 0.0)):
        reasons.append(f"{node} has insufficient RAM")
    gpu_ids = profile_policy.get("gpu_ids", [])
    if profile == "gpu":
        available = {
            int(row.get("index"))
            for row in probe.get("gpus", [])
            if isinstance(row, Mapping)
            and not row.get("compute_processes")
            and float(row.get("free_memory_gib", 0.0)) >= 20.0
            and float(row.get("utilization_percent", 101.0)) <= 5.0
        }
        if not isinstance(gpu_ids, list) or not gpu_ids or not set(gpu_ids) <= available:
            reasons.append(f"{node} admitted GPUs are unavailable")
    elif gpu_ids:
        reasons.append(f"{node} CPU profile must not reserve GPUs")
    if smoke is None:
        reasons.append(f"{node} {profile} smoke record is missing")
    else:
        expected_smoke = profile_policy.get("smoke_record_sha256")
        if _digest(smoke) != expected_smoke:
            reasons.append(f"{node} {profile} smoke hash mismatch")
        if (
            smoke.get("schema_version") != 1
            or smoke.get("node") != node
            or smoke.get("profile") != profile
            or smoke.get("executable") != executable
            or smoke.get("normally_terminated") is not True
            or smoke.get("exit_code") != 0
        ):
            reasons.append(f"{node} {profile} smoke record is invalid")
    return reasons


def plan_submission(
    stage: str,
    tasks: Sequence[TaskSpec],
    policy: Mapping[str, object],
    probe_records: Mapping[str, Mapping[str, object]],
    smoke_records: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    now: datetime,
) -> SubmissionPlan:
    if stage not in {"static", "berry"}:
        raise ValueError("stage must be static or berry")
    profile = "gpu" if stage == "static" else "cpu"
    task_rows = tuple(tasks)
    limit = int(policy.get("max_tasks_per_submission", 0))
    if len(task_rows) > limit or limit != 40:
        raise ValueError("one submission cannot contain more than 40 tasks")
    if not task_rows:
        raise ValueError("submission must contain at least one task")
    for task in task_rows:
        if not isinstance(task, TaskSpec) or task.stage != stage:
            raise ValueError("task stage does not match submission stage")
        _safe_work_dir(task)

    admitted: list[tuple[str, Mapping[str, object]]] = []
    rejected: list[str] = []
    node_policies = policy.get("nodes", {})
    if not isinstance(node_policies, Mapping):
        raise ValueError("policy nodes are invalid")
    for node in sorted(node_policies):
        node_policy = node_policies[node]
        if not isinstance(node_policy, Mapping):
            continue
        reasons = _admission_reasons(
            node,
            profile,
            node_policy,
            policy,
            probe_records.get(node),
            smoke_records.get((node, profile)),
            now,
        )
        if reasons:
            rejected.extend(reasons)
        else:
            admitted.append((node, node_policy[profile]))  # type: ignore[index]
    if not admitted:
        raise ValueError("; ".join(rejected) if rejected else "no node/profile is admitted")

    slots: list[tuple[str, int, int | None]] = []
    for node, profile_policy in admitted:
        concurrency = int(profile_policy.get("max_concurrency", 0))
        gpu_ids = list(profile_policy.get("gpu_ids", []))
        if concurrency < 1:
            continue
        for worker in range(concurrency):
            gpu_id = int(gpu_ids[worker % len(gpu_ids)]) if profile == "gpu" else None
            slots.append((node, worker, gpu_id))
    if not slots:
        raise ValueError("admitted policy contains no execution slots")
    assignments = []
    for index, task in enumerate(task_rows):
        node, worker, gpu_id = slots[index % len(slots)]
        assignments.append(
            Assignment(task.task_id, task.stage, task.work_dir, node, profile, worker, gpu_id)
        )
    return SubmissionPlan(
        stage,
        profile,
        str(policy["executables"][profile]),  # type: ignore[index]
        tuple(assignments),
    )


def retry_allowed(failure_kind: str, previous_retries: int, policy: Mapping[str, object]) -> bool:
    return (
        failure_kind == "operational"
        and type(previous_retries) is int
        and 0 <= previous_retries < int(policy.get("operational_retries", 0))
    )


def _ssh_read(
    node: str, remote_command: str, runner: Callable[..., subprocess.CompletedProcess[str]]
) -> str:
    completed = runner(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, remote_command],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def collect_live_probe(
    node: str,
    policy: Mapping[str, object],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: datetime | None = None,
) -> dict:
    """Collect current read-only node capacity and live-DFPT evidence."""
    if node not in policy.get("probeable_nodes", []):
        raise ValueError("node is not probeable under the Smidt policy")
    captured = now or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise ValueError("probe time must include a timezone")
    try:
        logical_cpus = int(_ssh_read(node, "getconf _NPROCESSORS_ONLN", runner))
        available_kib = int(
            _ssh_read(node, "awk '/MemAvailable:/ {print $2}' /proc/meminfo", runner)
        )
        executables = {}
        for profile, executable in policy.get("executables", {}).items():
            available = _ssh_read(
                node, f"test -x {shlex.quote(str(executable))} && echo 1 || echo 0", runner
            )
            executables[str(profile)] = {
                "path": str(executable),
                "available": available == "1",
            }
        lock_path = str(policy.get("lock_path", ""))
        lock_present = _ssh_read(
            node, f"test -e {shlex.quote(lock_path)} && echo 1 || echo 0", runner
        ) == "1"
        try:
            gpu_text = _ssh_read(
                node,
                "nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
                "--format=csv,noheader,nounits",
                runner,
            )
            process_text = _ssh_read(
                node,
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits",
                runner,
            )
        except subprocess.SubprocessError:
            gpu_text = ""
            process_text = ""
        compute_processes = [
            int(line.strip()) for line in process_text.splitlines() if line.strip().isdigit()
        ]
        gpus = []
        for line in gpu_text.splitlines():
            if not line.strip():
                continue
            index, free_mib, utilization = [value.strip() for value in line.split(",")]
            gpus.append(
                {
                    "index": int(index),
                    "free_memory_gib": float(free_mib) / 1024.0,
                    "utilization_percent": float(utilization),
                    "compute_processes": list(compute_processes),
                }
            )
        process_command = (
            "for pid in $(pgrep -f 'vasp_std|PWmat' || true); do "
            "cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null || true); "
            "args=$(tr '\\0' ' ' </proc/$pid/cmdline 2>/dev/null || true); "
            "case \"$cwd $args\" in *dfpt*|*gamma*) "
            "printf '%s\\t%s\\t%s\\n' \"$pid\" \"$cwd\" \"$args\";; esac; done"
        )
        process_rows = _ssh_read(node, process_command, runner)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(f"could not collect live probe from {node}") from exc
    live_dfpt = []
    for line in process_rows.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].isdigit():
            live_dfpt.append({"pid": int(parts[0]), "cwd": parts[1], "command": parts[2]})
    return {
        "schema_version": 1,
        "node": node,
        "captured_utc": captured.astimezone(timezone.utc).isoformat(),
        "logical_cpus": logical_cpus,
        "available_ram_gib": available_kib / (1024**2),
        "gpus": gpus,
        "executables": executables,
        "stage17_locks": [lock_path] if lock_present else [],
        "live_dfpt_processes": live_dfpt,
    }


def _queue_script(
    node: str, assignments: Sequence[Assignment], plan: SubmissionPlan, policy: Mapping[str, object]
) -> str:
    launcher = policy.get("launchers", {}).get(plan.profile, ["mpirun", "-np", "1"])  # type: ignore[union-attr]
    if not isinstance(launcher, list) or not launcher:
        raise ValueError("launcher policy is invalid")
    command = " ".join(shlex.quote(str(value)) for value in [*launcher, plan.executable])
    timeout = int(policy["timeouts_seconds"][plan.stage])  # type: ignore[index]
    groups: dict[int, list[Assignment]] = {}
    for assignment in assignments:
        groups.setdefault(assignment.worker_index, []).append(assignment)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"ROOT={shlex.quote(str(STAGE17_ROOT))}",
        f"exec 9>{shlex.quote(str(policy['lock_path']))}",
        "flock -n 9 || exit 75",
    ]
    for worker, rows in sorted(groups.items()):
        function = f"worker_{worker}"
        lines.append(f"{function}() {{")
        for row in rows:
            work = shlex.quote(row.work_dir)
            gpu = "" if row.gpu_id is None else f"CUDA_VISIBLE_DEVICES={row.gpu_id} "
            lines.extend(
                [
                    f"  cd \"$ROOT\"/{work}",
                    "  date -Is > started_at.txt",
                    f"  set +e; {gpu}timeout {timeout} {command} > vasp.out 2>&1; code=$?; set -e",
                    "  echo \"$code\" > exit_code.txt",
                    "  date -Is > finished_at.txt",
                ]
            )
        lines.append("}")
        lines.append(f"{function} &")
    lines.append("wait")
    return "\n".join(lines) + "\n"


def dispatch_submission(
    plan: SubmissionPlan,
    campaign_dir: Path,
    policy: Mapping[str, object],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict:
    """Write no-clobber queue scripts and dispatch them through injected SSH."""
    campaign_dir = Path(campaign_dir)
    dispatch_dir = campaign_dir / "dispatch"
    dispatch_dir.mkdir(parents=True, exist_ok=False)
    nodes: dict[str, dict[str, object]] = {}
    by_node: dict[str, list[Assignment]] = {}
    for assignment in plan.assignments:
        by_node.setdefault(assignment.node, []).append(assignment)
    for node, assignments in sorted(by_node.items()):
        script = dispatch_dir / f"{node}-{plan.stage}-queue.sh"
        script.write_text(_queue_script(node, assignments, plan, policy), encoding="utf-8")
        script.chmod(0o700)
        log = dispatch_dir / f"{node}-{plan.stage}-queue.log"
        remote = (
            f"nohup bash {shlex.quote(str(script))} > {shlex.quote(str(log))} 2>&1 "
            "< /dev/null & echo $!"
        )
        completed = runner(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, remote],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        try:
            pid = int(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{node} dispatch did not return a PID") from exc
        nodes[node] = {
            "pid": pid,
            "script": str(script),
            "log": str(log),
            "task_count": len(assignments),
        }
    return {
        "schema_version": 1,
        "status": "dispatched",
        "stage": plan.stage,
        "profile": plan.profile,
        "task_count": len(plan.assignments),
        "nodes": nodes,
    }
