from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_RE = re.compile(r"^attempt-(\d{2,})$")
OUTPUT_PACKAGE_RE = re.compile(r"^R[0-8](?:_|$)")
STAGE17_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_MARKERS = ("SUCCESS", "COMPLETED", "FAILED", "DISPATCHED")
TERMINAL_RECORD_NAMES = frozenset(("result.json", "failure.json", "dispatch.json"))


@dataclass(frozen=True)
class AttemptInspection:
    state: str
    reasons: tuple[str, ...]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_request(request: dict) -> tuple[str, ...]:
    reasons: list[str] = []
    required = {
        "schema_version",
        "request_id",
        "created_utc",
        "inputs",
        "command",
        "environment",
        "node",
        "resources",
        "output_roots",
    }
    if not required <= set(request):
        reasons.append("missing_request_field")
    if request.get("schema_version") != 1:
        reasons.append("invalid_request_schema")
    inputs = request.get("inputs", {})
    if not isinstance(inputs, dict) or not inputs:
        reasons.append("missing_inputs")
    else:
        for item in inputs.values():
            if not isinstance(item, dict) or not SHA256_RE.fullmatch(str(item.get("sha256", ""))):
                reasons.append("invalid_input_sha256")
                break
    if not isinstance(request.get("command"), list) or not request.get("command"):
        reasons.append("invalid_command")
    output_roots = request.get("output_roots")
    if not isinstance(output_roots, list) or not output_roots:
        reasons.append("invalid_output_roots")
    else:
        for raw in output_roots:
            path = PurePosixPath(str(raw))
            if (
                path.is_absolute()
                or "\\" in str(raw)
                or ".." in path.parts
                or len(path.parts) < 2
                or path.parts[0] not in {"runs", "results", "reports", "manifests"}
                or not OUTPUT_PACKAGE_RE.match(path.parts[1])
            ):
                reasons.append("invalid_output_root")
                break
    return tuple(dict.fromkeys(reasons))


def validate_result(result: dict, request_path: Path) -> tuple[str, ...]:
    reasons: list[str] = []
    required = {
        "schema_version",
        "request_sha256",
        "status",
        "outputs",
        "started_utc",
        "finished_utc",
        "exit_code",
        "reason_codes",
    }
    if not required <= set(result):
        reasons.append("missing_result_field")
    if result.get("schema_version") != 1:
        reasons.append("invalid_result_schema")
    if result.get("status") not in {"success", "failed"}:
        reasons.append("invalid_result_status")
    if not SHA256_RE.fullmatch(str(result.get("request_sha256", ""))):
        reasons.append("invalid_request_sha256")
    elif result["request_sha256"] != _sha256(request_path):
        reasons.append("request_hash_mismatch")
    try:
        request = _read_json(request_path)
    except (OSError, json.JSONDecodeError):
        request = {}
        reasons.append("invalid_request_json")
    output_roots = [PurePosixPath(str(value)) for value in request.get("output_roots", [])]
    outputs = result.get("outputs", {})
    if not isinstance(outputs, dict):
        reasons.append("invalid_outputs")
    else:
        for item in outputs.values():
            if not isinstance(item, dict) or not SHA256_RE.fullmatch(str(item.get("sha256", ""))):
                reasons.append("invalid_output_sha256")
                continue
            raw_path = str(item.get("path", ""))
            output_path = PurePosixPath(raw_path)
            inside_declared_root = any(
                output_path != root and root in output_path.parents for root in output_roots
            )
            if (
                not raw_path
                or output_path.is_absolute()
                or "\\" in raw_path
                or ".." in output_path.parts
                or not inside_declared_root
            ):
                reasons.append("invalid_output_path")
                continue
            target = STAGE17_ROOT.joinpath(*output_path.parts)
            if not target.is_file():
                reasons.append("missing_output")
            elif _sha256(target) != item["sha256"]:
                reasons.append("output_hash_mismatch")
    return tuple(dict.fromkeys(reasons))


def validate_dispatch(dispatch: dict, request_path: Path) -> tuple[str, ...]:
    reasons: list[str] = []
    required = {
        "schema_version",
        "request_sha256",
        "status",
        "work_roots",
        "dispatched_utc",
        "reason_codes",
    }
    if not required <= set(dispatch):
        reasons.append("missing_dispatch_field")
    if dispatch.get("schema_version") != 1:
        reasons.append("invalid_dispatch_schema")
    if dispatch.get("status") != "dispatched":
        reasons.append("invalid_dispatch_status")
    if not SHA256_RE.fullmatch(str(dispatch.get("request_sha256", ""))):
        reasons.append("invalid_request_sha256")
    elif dispatch["request_sha256"] != _sha256(request_path):
        reasons.append("request_hash_mismatch")
    try:
        request = _read_json(request_path)
    except (OSError, json.JSONDecodeError):
        request = {}
        reasons.append("invalid_request_json")
    output_roots = [PurePosixPath(str(value)) for value in request.get("output_roots", [])]
    work_roots = dispatch.get("work_roots")
    if not isinstance(work_roots, list) or not work_roots:
        reasons.append("invalid_work_roots")
    else:
        for raw_path in work_roots:
            path = PurePosixPath(str(raw_path))
            inside_declared_root = any(path != root and root in path.parents for root in output_roots)
            if (
                path.is_absolute()
                or "\\" in str(raw_path)
                or ".." in path.parts
                or not inside_declared_root
            ):
                reasons.append("invalid_work_root")
                break
    if not isinstance(dispatch.get("reason_codes"), list):
        reasons.append("invalid_reason_codes")
    return tuple(dict.fromkeys(reasons))


def _record_for_marker(attempt_dir: Path, marker: str) -> Path:
    if marker in {"SUCCESS", "COMPLETED"}:
        return attempt_dir / "result.json"
    if marker == "DISPATCHED":
        return attempt_dir / "dispatch.json"
    failure = attempt_dir / "failure.json"
    return failure if failure.exists() else attempt_dir / "result.json"


def _validate_terminal_record(record_path: Path, request_path: Path) -> tuple[str, ...]:
    try:
        record = _read_json(record_path)
    except (OSError, json.JSONDecodeError):
        return ("invalid_terminal_record_json",)
    if record_path.name == "dispatch.json":
        return validate_dispatch(record, request_path)
    reasons = list(validate_result(record, request_path))
    if record_path.name == "failure.json" and record.get("status") != "failed":
        reasons.append("failure_record_status_mismatch")
    return tuple(dict.fromkeys(reasons))


def inspect_attempt(attempt_dir: Path) -> AttemptInspection:
    request_path = attempt_dir / "request.json"
    reasons: list[str] = []

    if not request_path.is_file():
        return AttemptInspection("invalid", ("missing_request",))
    try:
        reasons.extend(validate_request(_read_json(request_path)))
    except (OSError, json.JSONDecodeError):
        return AttemptInspection("invalid", ("invalid_request_json",))

    markers = [attempt_dir / name for name in TERMINAL_MARKERS if (attempt_dir / name).exists()]
    if len(markers) > 1:
        reasons.append("multiple_terminal_markers")
    if not markers:
        orphaned = [name for name in TERMINAL_RECORD_NAMES if (attempt_dir / name).is_file()]
        if orphaned:
            state = "orphaned_result" if orphaned == ["result.json"] else "orphaned_terminal_record"
            return AttemptInspection(state, ())
        return AttemptInspection("invalid" if reasons else "prepared", tuple(dict.fromkeys(reasons)))
    if len(markers) != 1:
        return AttemptInspection("invalid", tuple(dict.fromkeys(reasons)))

    marker = markers[0]
    record_path = _record_for_marker(attempt_dir, marker.name)
    if not record_path.is_file():
        reason = "marker_without_result" if record_path.name == "result.json" else "marker_without_terminal_record"
        reasons.append(reason)
        return AttemptInspection("invalid", tuple(dict.fromkeys(reasons)))
    reasons.extend(_validate_terminal_record(record_path, request_path))
    if marker.stat().st_mtime_ns < record_path.stat().st_mtime_ns:
        reasons.append("marker_before_result")
    record = _read_json(record_path)
    expected_markers = {
        "success": {"SUCCESS", "COMPLETED"},
        "failed": {"FAILED"},
        "dispatched": {"DISPATCHED"},
    }.get(record.get("status"), set())
    if marker.name not in expected_markers:
        reasons.append("marker_status_mismatch")
    if reasons:
        return AttemptInspection("invalid", tuple(reasons))
    return AttemptInspection(str(record["status"]), ())


def next_attempt_path(run_dir: Path) -> Path:
    numbers = []
    if run_dir.exists():
        for child in run_dir.iterdir():
            match = ATTEMPT_RE.fullmatch(child.name)
            if child.is_dir() and match:
                numbers.append(int(match.group(1)))
    return run_dir / f"attempt-{max(numbers, default=0) + 1:02d}"


def validate_resume(previous_attempt: Path, new_request: dict) -> tuple[str, ...]:
    reasons = list(validate_request(new_request))
    request_path = previous_attempt / "request.json"
    if not request_path.is_file():
        reasons.append("missing_resume_source")
        return tuple(dict.fromkeys(reasons))
    resume = new_request.get("resume_from")
    if not isinstance(resume, dict):
        reasons.append("missing_resume_reference")
        return tuple(dict.fromkeys(reasons))
    if resume.get("attempt") != previous_attempt.name:
        reasons.append("resume_attempt_mismatch")
    terminal_name = resume.get("terminal_record")
    if terminal_name is not None:
        if terminal_name not in TERMINAL_RECORD_NAMES:
            reasons.append("invalid_resume_terminal_record")
            record_path = previous_attempt / "__invalid__"
        else:
            record_path = previous_attempt / str(terminal_name)
        if not record_path.is_file():
            reasons.append("missing_resume_source")
        else:
            reasons.extend(_validate_terminal_record(record_path, request_path))
            if resume.get("terminal_record_sha256") != _sha256(record_path):
                reasons.append("resume_terminal_hash_mismatch")
    else:
        result_path = previous_attempt / "result.json"
        if not result_path.is_file():
            reasons.append("missing_resume_source")
            return tuple(dict.fromkeys(reasons))
        try:
            reasons.extend(validate_result(_read_json(result_path), request_path))
        except (OSError, json.JSONDecodeError):
            reasons.append("invalid_result_json")
        if resume.get("request_sha256") != _sha256(request_path):
            reasons.append("resume_request_hash_mismatch")
        if resume.get("result_sha256") != _sha256(result_path):
            reasons.append("resume_result_hash_mismatch")
    previous = _read_json(request_path)
    old_inputs = {key: value.get("sha256") for key, value in previous.get("inputs", {}).items()}
    new_inputs = {key: value.get("sha256") for key, value in new_request.get("inputs", {}).items()}
    if old_inputs != new_inputs:
        reasons.append("input_hash_drift")
    return tuple(dict.fromkeys(reasons))
