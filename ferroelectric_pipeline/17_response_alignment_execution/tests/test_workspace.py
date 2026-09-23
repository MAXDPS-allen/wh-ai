from __future__ import annotations

import json
import importlib.util
import hashlib
import copy
import sys
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


STAGE17 = Path(__file__).resolve().parents[1]
REPO_ROOT = STAGE17.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WorkspaceStructureTests(unittest.TestCase):
    def test_required_root_files_exist(self) -> None:
        required = [
            "README.md",
            "STATUS.md",
            "configs/README.md",
            "configs/cluster_policy.json",
            "configs/path_registry.json",
            "configs/work_packages.json",
            "docs/README.md",
            "docs/execution_rules.md",
            "docs/roadmap_addendum.md",
            "docs/source_reuse_map.md",
            "src/README.md",
            "scripts/README.md",
            "scripts/render_status.py",
            "scripts/probe_cluster.py",
            "scripts/validate_workspace.py",
            "manifests/README.md",
            "runs/README.md",
            "results/README.md",
            "reports/README.md",
        ]
        missing = [path for path in required if not (STAGE17 / path).is_file()]
        self.assertEqual(missing, [])

    def test_required_work_packages_exist(self) -> None:
        expected = {
            "00_r0_collect_and_parents",
            "01_r1_response_alignment",
            "02_r2_mechanism_analysis",
            "03_r3_candidate_screening",
            "04_r4_cdte_control",
            "05_r5_new_candidate_closure",
            "06_r6_headline_validation",
            "07_r7_publication_audit",
            "08_r8_manuscript_package",
        }
        root = STAGE17 / "work_packages"
        actual = {path.name for path in root.iterdir() if path.is_dir()} if root.exists() else set()
        self.assertEqual(actual, expected)
        self.assertTrue(all((root / name / "README.md").is_file() for name in expected))

    def test_canonical_state_covers_r0_through_r8(self) -> None:
        path = STAGE17 / "configs/work_packages.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({item["id"] for item in data["packages"]}, {f"R{i}" for i in range(9)})


class StatusRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        path = STAGE17 / "scripts/render_status.py"
        self.assertTrue(path.is_file())
        self.renderer = load_module("stage17_render_status", path)

    def test_render_sorts_packages_by_numeric_id(self) -> None:
        packages = [
            {"id": "R8", "title": "Last", "status": "not_started", "next_action": "Wait"},
            {"id": "R0", "title": "First", "status": "ready", "next_action": "Start"},
        ]
        rendered = self.renderer.render(packages)
        self.assertLess(rendered.index("R0 — First"), rendered.index("R8 — Last"))

    def test_render_rejects_unknown_status(self) -> None:
        packages = [{"id": "R0", "title": "Bad", "status": "unknown", "next_action": "None"}]
        with self.assertRaisesRegex(ValueError, "invalid status"):
            self.renderer.render(packages)

    def test_status_file_equals_renderer_output(self) -> None:
        config = json.loads((STAGE17 / "configs/work_packages.json").read_text(encoding="utf-8"))
        expected = self.renderer.render(config["packages"])
        actual = (STAGE17 / "STATUS.md").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)


class PathRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        path = STAGE17 / "configs/path_registry.json"
        self.assertTrue(path.is_file())
        self.registry = json.loads(path.read_text(encoding="utf-8"))

    def test_registry_has_unique_exact_resolving_entries(self) -> None:
        self.assertEqual(self.registry.get("schema_version"), 1)
        entries = self.registry.get("assets", [])
        ids = [entry.get("id") for entry in entries]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 20)
        for entry in entries:
            self.assertEqual(
                set(entry), {"id", "stage", "path", "kind", "role", "access", "consumers"}
            )
            self.assertIn(entry["stage"], range(8, 17))
            self.assertIn(entry["kind"], {"code", "data", "model", "result", "documentation"})
            self.assertEqual(entry["access"], "read_only")
            self.assertTrue(entry["consumers"])
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertTrue((REPO_ROOT / entry["path"]).exists(), entry["path"])

    def test_registry_contains_all_protected_stage16_files(self) -> None:
        expected = {
            "s16_cluster_runtime_policy",
            "s16_dft_contracts",
            "s16_cluster_nodes_policy",
            "s16_cluster_probe",
        }
        ids = {entry["id"] for entry in self.registry["assets"]}
        self.assertTrue(expected <= ids)

    def test_protected_source_manifest_matches_files(self) -> None:
        manifest = STAGE17 / "manifests/protected_sources.sha256"
        self.assertTrue(manifest.is_file())
        lines = [line.split(maxsplit=1) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 4)
        for digest, relpath in lines:
            target = REPO_ROOT / relpath.strip()
            self.assertTrue(target.is_file())
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


class ClusterPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        policy_path = STAGE17 / "configs/cluster_policy.json"
        module_path = STAGE17 / "src/stage17/preflight.py"
        self.assertTrue(policy_path.is_file())
        self.assertTrue(module_path.is_file())
        self.policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.preflight = load_module("stage17_preflight", module_path)

    def base_probe(self) -> dict:
        return {
            "node": "g4",
            "ssh_ok": True,
            "logical_cpus": 128,
            "one_minute_load": 10.0,
            "available_ram_gib": 480.0,
            "scheduler_exclusive": False,
            "active_stage17_tasks": 0,
            "executable_ok": {"cpu": True, "gpu": True},
            "gpu_samples": [
                {"free_memory_gib": 24.0, "utilization_percent": 0.0, "foreign_compute_processes": 0},
                {"free_memory_gib": 24.0, "utilization_percent": 0.0, "foreign_compute_processes": 0},
            ],
        }

    def base_request(self, profile: str = "gpu") -> dict:
        return {
            "profile": profile,
            "cpu_cores": 16,
            "ram_gib": 64,
            "gpu_count": 1 if profile == "gpu" else 0,
            "scientific_execution": False,
        }

    def enabled_policy(self) -> dict:
        policy = copy.deepcopy(self.policy)
        policy["probe_only"] = False
        policy["allow_scientific_execution"] = True
        policy["profiles"]["gpu"].update(
            usable=True, launcher="direct", environment_profile="fe_gpu"
        )
        policy["profiles"]["cpu"].update(
            usable=True, launcher="mpirun", environment_profile="fe_dft"
        )
        policy["node_gates"]["g4_smoke_passed"] = True
        return policy

    def test_probe_only_blocks_scientific_execution_even_when_other_gates_open(self) -> None:
        policy = self.enabled_policy()
        policy["probe_only"] = True
        request = self.base_request()
        request["scientific_execution"] = True
        decision = self.preflight.evaluate_probe(policy, self.base_probe(), request)
        self.assertFalse(decision.admitted)
        self.assertIn("probe_only_enabled", decision.reasons)

    def test_policy_is_stage17_only_and_defaults_fail_closed(self) -> None:
        self.assertEqual(set(self.policy["probeable_nodes"]), {"g1", "g3", "g4", "g6", "g7"})
        self.assertTrue(self.policy["probe_only"])
        self.assertFalse(self.policy["allow_scientific_execution"])
        self.assertEqual(self.policy["max_formal_tasks_per_node"], 1)
        for path in self.policy["executables"].values():
            self.assertTrue(Path(path).is_absolute())
        decision = self.preflight.evaluate_probe(
            self.policy, self.base_probe(), self.base_request()
        )
        self.assertFalse(decision.admitted)

    def test_recorded_g4_load_rejects_64_core_request(self) -> None:
        probe = self.base_probe()
        probe["one_minute_load"] = 65.0
        request = self.base_request("cpu")
        request.update(cpu_cores=64, scientific_execution=True)
        decision = self.preflight.evaluate_probe(self.enabled_policy(), probe, request)
        self.assertFalse(decision.admitted)
        self.assertIn("cpu_headroom", decision.reasons)

    def test_resource_and_ownership_failures_are_reason_coded(self) -> None:
        cases = [
            ("ssh_failed", lambda p: p.update(ssh_ok=False)),
            ("ram_headroom", lambda p: p.update(available_ram_gib=70.0)),
            ("active_stage17_task", lambda p: p.update(active_stage17_tasks=1)),
            ("gpu_two_samples_required", lambda p: p.update(gpu_samples=p["gpu_samples"][:1])),
            ("gpu_memory", lambda p: p["gpu_samples"][0].update(free_memory_gib=10.0)),
            ("gpu_utilization", lambda p: p["gpu_samples"][1].update(utilization_percent=8.0)),
            ("gpu_foreign_process", lambda p: p["gpu_samples"][0].update(foreign_compute_processes=1)),
            ("executable_unavailable", lambda p: p["executable_ok"].update(gpu=False)),
        ]
        request = self.base_request()
        request["scientific_execution"] = True
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                probe = self.base_probe()
                mutate(probe)
                decision = self.preflight.evaluate_probe(self.enabled_policy(), probe, request)
                self.assertFalse(decision.admitted)
                self.assertIn(expected, decision.reasons)

    def test_profile_and_g4_smoke_gates_are_independent(self) -> None:
        request = self.base_request()
        request["scientific_execution"] = True
        policy = self.enabled_policy()
        policy["profiles"]["gpu"]["launcher"] = ""
        self.assertIn(
            "missing_launcher",
            self.preflight.evaluate_probe(policy, self.base_probe(), request).reasons,
        )
        policy = self.enabled_policy()
        policy["profiles"]["gpu"]["environment_profile"] = ""
        self.assertIn(
            "missing_environment",
            self.preflight.evaluate_probe(policy, self.base_probe(), request).reasons,
        )
        policy = self.enabled_policy()
        policy["node_gates"]["g4_smoke_passed"] = False
        self.assertIn(
            "g4_smoke_required",
            self.preflight.evaluate_probe(policy, self.base_probe(), request).reasons,
        )

    def test_fully_valid_enabled_probe_is_admitted(self) -> None:
        request = self.base_request()
        request["scientific_execution"] = True
        decision = self.preflight.evaluate_probe(
            self.enabled_policy(), self.base_probe(), request
        )
        self.assertTrue(decision.admitted, decision.reasons)


class WorkPackageContractTests(unittest.TestCase):
    REQUIRED_HEADINGS = [
        "## Objective",
        "## Status",
        "## Entry criteria",
        "## Inputs",
        "## Outputs",
        "## Reusable assets",
        "## Minimal command",
        "## Checks",
        "## Stop/go criteria",
        "## Duration and resource",
        "## Protected sources",
        "## Output location",
    ]

    def setUp(self) -> None:
        state = json.loads((STAGE17 / "configs/work_packages.json").read_text(encoding="utf-8"))
        self.packages = state["packages"]
        registry = json.loads((STAGE17 / "configs/path_registry.json").read_text(encoding="utf-8"))
        self.registry = registry["assets"]
        self.registry_ids = {entry["id"] for entry in registry["assets"]}

    def test_each_work_package_has_complete_contract(self) -> None:
        for package in self.packages:
            with self.subTest(package=package["id"]):
                path = STAGE17 / package["directory"] / "README.md"
                self.assertTrue(path.is_file())
                text = path.read_text(encoding="utf-8")
                positions = [text.find(heading) for heading in self.REQUIRED_HEADINGS]
                self.assertTrue(all(position >= 0 for position in positions))
                self.assertEqual(positions, sorted(positions))
                self.assertIn(f"`{package['status']}`", text)
                self.assertTrue("implementation pending" in text or "python3 -B" in text)
                self.assertIn("Stage 17", text)

    def test_reusable_asset_ids_are_registered(self) -> None:
        pattern = re.compile(r"`(s\d{2}_[a-z0-9_]+)`")
        for package in self.packages:
            path = STAGE17 / package["directory"] / "README.md"
            self.assertTrue(path.is_file(), package["id"])
            text = path.read_text(encoding="utf-8")
            ids = set(pattern.findall(text))
            self.assertTrue(ids, package["id"])
            self.assertEqual(ids - self.registry_ids, set(), package["id"])

    def test_each_output_location_is_inside_stage17(self) -> None:
        for package in self.packages:
            path = STAGE17 / package["directory"] / "README.md"
            self.assertTrue(path.is_file(), package["id"])
            text = path.read_text(encoding="utf-8")
            self.assertRegex(text, r"`(?:runs|results|reports|manifests)/R[0-8][^`]*`")

    def test_canonical_dependencies_match_entry_gates(self) -> None:
        by_id = {item["id"]: item for item in self.packages}
        self.assertEqual(set(by_id["R5"]["depends_on"]), {"R3", "R4"})
        for package in self.packages:
            self.assertNotIn(package["id"], package["depends_on"])

    def test_registry_consumers_cover_readme_usage(self) -> None:
        pattern = re.compile(r"`(s\d{2}_[a-z0-9_]+)`")
        actual = {entry["id"]: set() for entry in self.registry}
        package_ids = {item["id"] for item in self.packages}
        for package in self.packages:
            text = (STAGE17 / package["directory"] / "README.md").read_text(encoding="utf-8")
            for asset_id in pattern.findall(text):
                actual[asset_id].add(package["id"])
        for entry in self.registry:
            with self.subTest(asset=entry["id"]):
                declared = set(entry["consumers"])
                self.assertTrue(declared <= package_ids)
                self.assertEqual(actual[entry["id"]] - declared, set())


class RunContractTests(unittest.TestCase):
    DIGEST = "a" * 64

    def setUp(self) -> None:
        module_path = STAGE17 / "src/stage17/run_contract.py"
        self.assertTrue(module_path.is_file())
        self.contract = load_module("stage17_run_contract", module_path)

    def request(self) -> dict:
        return {
            "schema_version": 1,
            "request_id": "test-request",
            "created_utc": "2026-09-16T00:00:00Z",
            "inputs": {"structure": {"path": "input.json", "sha256": self.DIGEST}},
            "command": ["safe-command", "--dry-run"],
            "environment": "test",
            "node": "g4",
            "resources": {"cpu_cores": 1, "ram_gib": 1, "gpu_count": 0},
            "output_roots": ["results/R0_test", "runs/R0_test"],
        }

    def write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")

    def result(
        self, request_path: Path, workspace_root: Path | None = None, status: str = "success"
    ) -> dict:
        output_path = "results/R0_test/table.json"
        output_digest = "b" * 64
        if workspace_root is not None:
            target = workspace_root / output_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("verified output\n", encoding="utf-8")
            output_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {
            "schema_version": 1,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "status": status,
            "outputs": {"table": {"path": output_path, "sha256": output_digest}},
            "started_utc": "2026-09-16T00:00:01Z",
            "finished_utc": "2026-09-16T00:00:02Z",
            "exit_code": 0 if status == "success" else 1,
            "reason_codes": [],
        }

    def dispatch(self, request_path: Path) -> dict:
        return {
            "schema_version": 1,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "status": "dispatched",
            "work_roots": ["runs/R0_test/work/static/image-000"],
            "dispatched_utc": "2026-09-16T00:00:02Z",
            "reason_codes": [],
        }

    def test_result_then_marker_forms_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            attempt = Path(raw) / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            self.write_json(request_path, self.request())
            self.contract.STAGE17_ROOT = Path(raw)
            self.write_json(attempt / "result.json", self.result(request_path, Path(raw)))
            time.sleep(0.002)
            (attempt / "SUCCESS").write_text("\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(attempt)
            self.assertEqual(inspection.state, "success")
            self.assertEqual(inspection.reasons, ())

    def test_new_completed_and_dispatched_terminal_records_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.contract.STAGE17_ROOT = root
            completed = root / "attempt-01"
            completed.mkdir()
            completed_request = completed / "request.json"
            self.write_json(completed_request, self.request())
            self.write_json(completed / "result.json", self.result(completed_request, root))
            time.sleep(0.002)
            (completed / "COMPLETED").write_text("\n", encoding="utf-8")
            self.assertEqual(self.contract.inspect_attempt(completed).state, "success")

            dispatched = root / "attempt-02"
            dispatched.mkdir()
            dispatched_request = dispatched / "request.json"
            self.write_json(dispatched_request, self.request())
            self.write_json(dispatched / "dispatch.json", self.dispatch(dispatched_request))
            time.sleep(0.002)
            (dispatched / "DISPATCHED").write_text("\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(dispatched)
            self.assertEqual(inspection.state, "dispatched")
            self.assertEqual(inspection.reasons, ())

    def test_failed_terminal_uses_failure_record_and_rejects_multiple_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.contract.STAGE17_ROOT = root
            attempt = root / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            self.write_json(request_path, self.request())
            self.write_json(attempt / "failure.json", self.result(request_path, root, status="failed"))
            time.sleep(0.002)
            (attempt / "FAILED").write_text("\n", encoding="utf-8")
            self.assertEqual(self.contract.inspect_attempt(attempt).state, "failed")
            (attempt / "COMPLETED").write_text("\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(attempt)
            self.assertEqual(inspection.state, "invalid")
            self.assertIn("multiple_terminal_markers", inspection.reasons)

    def test_orphan_and_marker_without_result_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            attempt = Path(raw) / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            self.write_json(request_path, self.request())
            self.contract.STAGE17_ROOT = Path(raw)
            self.write_json(attempt / "result.json", self.result(request_path, Path(raw)))
            self.assertEqual(self.contract.inspect_attempt(attempt).state, "orphaned_result")
            (attempt / "result.json").unlink()
            (attempt / "SUCCESS").write_text("\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(attempt)
            self.assertEqual(inspection.state, "invalid")
            self.assertIn("marker_without_result", inspection.reasons)

    def test_next_attempt_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            (run_dir / "attempt-01").mkdir()
            (run_dir / "attempt-03").mkdir()
            self.assertEqual(self.contract.next_attempt_path(run_dir).name, "attempt-04")

    def test_resume_references_hashes_and_rejects_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            attempt = Path(raw) / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            result_path = attempt / "result.json"
            previous_request = self.request()
            self.write_json(request_path, previous_request)
            self.contract.STAGE17_ROOT = Path(raw)
            self.write_json(result_path, self.result(request_path, Path(raw), status="failed"))
            resumed = self.request()
            resumed["request_id"] = "resume-request"
            resumed["resume_from"] = {
                "attempt": "attempt-01",
                "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
            self.assertEqual(self.contract.validate_resume(attempt, resumed), ())
            resumed["inputs"]["structure"]["sha256"] = "c" * 64
            self.assertIn("input_hash_drift", self.contract.validate_resume(attempt, resumed))

    def test_resume_can_bind_dispatched_terminal_record_and_detect_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.contract.STAGE17_ROOT = root
            attempt = root / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            dispatch_path = attempt / "dispatch.json"
            self.write_json(request_path, self.request())
            self.write_json(dispatch_path, self.dispatch(request_path))
            (attempt / "DISPATCHED").write_text("\n", encoding="utf-8")
            resumed = self.request()
            resumed["request_id"] = "resume-dispatch"
            resumed["resume_from"] = {
                "attempt": "attempt-01",
                "terminal_record": "dispatch.json",
                "terminal_record_sha256": hashlib.sha256(dispatch_path.read_bytes()).hexdigest(),
            }
            self.assertEqual(self.contract.validate_resume(attempt, resumed), ())
            resumed["resume_from"]["terminal_record_sha256"] = "c" * 64
            self.assertIn(
                "resume_terminal_hash_mismatch",
                self.contract.validate_resume(attempt, resumed),
            )

    def test_invalid_sha256_is_rejected(self) -> None:
        request = self.request()
        request["inputs"]["structure"]["sha256"] = "not-a-digest"
        self.assertIn("invalid_input_sha256", self.contract.validate_request(request))

    def test_output_roots_cannot_escape_stage17(self) -> None:
        invalid = [
            "../../16_method_validation_and_completion/runs",
            "/tmp/outside",
            "runs/not_a_package",
            "other/R0_test",
        ]
        for output_root in invalid:
            with self.subTest(output_root=output_root):
                request = self.request()
                request["output_roots"] = [output_root]
                self.assertIn("invalid_output_root", self.contract.validate_request(request))
        request = self.request()
        request["output_roots"] = ["runs/R0_test", "results/R0_test", "reports/R0_test"]
        self.assertNotIn("invalid_output_root", self.contract.validate_request(request))

    def test_terminal_success_requires_existing_hash_matched_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.contract.STAGE17_ROOT = root
            attempt = root / "attempt-01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            self.write_json(request_path, self.request())
            self.write_json(attempt / "result.json", self.result(request_path))
            (attempt / "SUCCESS").write_text("\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(attempt)
            self.assertEqual(inspection.state, "invalid")
            self.assertIn("missing_output", inspection.reasons)

            target = root / "results/R0_test/table.json"
            target.parent.mkdir(parents=True)
            target.write_text("tampered\n", encoding="utf-8")
            inspection = self.contract.inspect_attempt(attempt)
            self.assertEqual(inspection.state, "invalid")
            self.assertIn("output_hash_mismatch", inspection.reasons)


class WorkspaceValidatorTests(unittest.TestCase):
    def test_incomplete_attempt_is_rejected(self) -> None:
        validator = load_module(
            "stage17_workspace_validator_for_attempts",
            STAGE17 / "scripts/validate_workspace.py",
        )
        with tempfile.TemporaryDirectory() as raw:
            attempt = Path(raw) / "R0_example/run/attempt-01"
            attempt.mkdir(parents=True)
            (attempt / "request.json").write_text("{}\n", encoding="utf-8")
            errors = validator.validate_attempts(Path(raw))
            self.assertTrue(any("attempt-01" in error for error in errors))

    def test_one_command_validator_passes_complete_scaffold(self) -> None:
        script = STAGE17 / "scripts/validate_workspace.py"
        self.assertTrue(script.is_file())
        completed = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=REPO_ROOT,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PASS: Stage 17 workspace is structurally valid", completed.stdout)


if __name__ == "__main__":
    unittest.main()
