"""evomcp evaluator for the ecdsa.fail Rust benchmark."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from evomcp.pipeline import Budget, Candidate, CostMetrics, EvalResult, FailureClass
from evomcp.pipeline.evaluator import materialize_prog_genome
from evomcp.pipeline.registry import DEFAULT_REGISTRY


ROOT = Path(__file__).resolve().parents[2]
INT_ENV_KEYS = {
    "DIALOG_GCD_COMPARE_BITS",
    "DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS",
    "DIALOG_GCD_ACTIVE_ITERATIONS",
    "DIALOG_GCD_WIDTH_MARGIN",
    "DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN",
    "DIALOG_GCD_APPLY_CHUNKED_F_CUT",
    "DIALOG_GCD_APPLY_CHUNKED_F_CUT2",
    "DIALOG_REROLL",
    "DIALOG_POST_SUB_REROLL",
}


class CommandError(RuntimeError):
    """Subprocess failure with logs kept for correctness-aware scoring."""

    def __init__(self, prefix: str, returncode: int, stdout: str, stderr: str):
        super().__init__(f"{prefix} failed with code {returncode}: {stderr[-1000:]}")
        self.prefix = prefix
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class EcdsaFailEvaluator:
    """Run build_circuit/eval_circuit and score lower score.json values higher."""

    version = "ecdsa.fail-v1"

    def __init__(self, project_root: Path = ROOT):
        self.project_root = project_root

    def evaluate(self, candidate: Candidate, budget: Budget, seed: int, *, run_dir: Path) -> EvalResult:
        started = time.monotonic()
        bundle = run_dir / f"{candidate.candidate_id[:12]}-seed{seed}"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2))
        (bundle / "budget.json").write_text(json.dumps(budget.to_dict(), indent=2))

        env_map = self._candidate_env(candidate, budget)
        (bundle / "inputs.json").write_text(json.dumps(env_map, indent=2, sort_keys=True))
        env = os.environ.copy()
        env.update({key: str(value) for key, value in env_map.items()})

        mode = str(budget.env_overrides.get("ECDSAFAIL_EVAL_MODE", "full"))
        try:
            self._ensure_binaries(env, bundle, budget.timeout_s)
            build = self._run(
                ["./target/release/build_circuit"],
                env=env,
                timeout=budget.timeout_s,
                bundle=bundle,
                prefix="build",
            )
            build_metrics = _parse_build_stdout(build.stdout)
            if mode == "build_only":
                emitted = build_metrics.get("emitted_ops", 0.0)
                qubits = build_metrics.get("qubits", 1.0)
                proxy_score = emitted * qubits
                return self._result(
                    candidate,
                    budget,
                    seed,
                    bundle,
                    success=True,
                    primary_score=-proxy_score,
                    secondary={
                        "proxy_score": proxy_score,
                        "emitted_ops": emitted,
                        "qubits": qubits,
                        "wall_s": time.monotonic() - started,
                    },
                )

            note = f"evomcp {candidate.candidate_id[:12]}"
            try:
                eval_run = self._run(
                    ["./target/release/eval_circuit", "--note", note],
                    env=env,
                    timeout=budget.timeout_s,
                    bundle=bundle,
                    prefix="eval",
                )
            except CommandError as exc:
                if exc.prefix != "eval":
                    raise
                metrics = _parse_eval_stdout(exc.stdout)
                metrics.update(_parse_eval_header(exc.stdout))
                emitted = build_metrics.get("emitted_ops", metrics.get("loaded_ops", 0.0))
                invalid_units = (
                    metrics.get("classical_mismatches", 0.0)
                    + metrics.get("phase_garbage_batches", 0.0)
                    + metrics.get("ancilla_garbage_batches", 0.0)
                    + 1.0
                )
                invalid_score = 1_000_000_000_000.0 + emitted + 100_000_000.0 * invalid_units
                secondary = {
                    "valid": 0.0,
                    "invalid_penalty_score": invalid_score,
                    "emitted_ops": emitted,
                    "wall_s": time.monotonic() - started,
                }
                secondary.update(metrics)
                return self._result(
                    candidate,
                    budget,
                    seed,
                    bundle,
                    success=True,
                    primary_score=-invalid_score,
                    secondary=secondary,
                )
            metrics = json.loads((self.project_root / "score.json").read_text())["metrics"]
            score = int(json.loads((self.project_root / "score.json").read_text())["score"])
            secondary = {
                "valid": 1.0,
                "score": float(score),
                "toffoli": float(metrics["toffoli"]),
                "qubits": float(metrics["qubits"]),
                "emitted_ops": build_metrics.get("emitted_ops", 0.0),
                "wall_s": time.monotonic() - started,
            }
            secondary.update(_parse_eval_stdout(eval_run.stdout))
            return self._result(
                candidate,
                budget,
                seed,
                bundle,
                success=True,
                primary_score=-float(score),
                secondary=secondary,
            )
        except subprocess.TimeoutExpired as exc:
            return EvalResult.penalized(
                candidate.candidate_id,
                FailureClass.TIMEOUT,
                f"timeout after {exc.timeout}s",
                stage=budget.stage,
                evaluator_version=self.version,
                seed=seed,
                dataset_version="ecdsa.fail",
            )
        except Exception as exc:  # noqa: BLE001
            (bundle / "failure.txt").write_text(str(exc))
            return EvalResult.penalized(
                candidate.candidate_id,
                FailureClass.RUNTIME,
                str(exc),
                stage=budget.stage,
                evaluator_version=self.version,
                seed=seed,
                dataset_version="ecdsa.fail",
            )

    def _candidate_env(self, candidate: Candidate, budget: Budget) -> dict[str, Any]:
        patch_id = str(candidate.prog_genome.get("patch_id", "current_1434"))
        patch_env = DEFAULT_REGISTRY.resolve_patch_env(patch_id)
        env_map = materialize_prog_genome(candidate, budget, base_prog=patch_env)
        env_map.pop("patch_id", None)
        for key in INT_ENV_KEYS & env_map.keys():
            env_map[key] = int(round(float(env_map[key])))
        return env_map

    def _ensure_binaries(self, env: dict[str, str], bundle: Path, timeout_s: int) -> None:
        build_bin = self.project_root / "target/release/build_circuit"
        eval_bin = self.project_root / "target/release/eval_circuit"
        if build_bin.exists() and eval_bin.exists():
            return
        self._run(
            ["cargo", "build", "--release", "--locked", "--bin", "build_circuit", "--bin", "eval_circuit"],
            env=env,
            timeout=max(timeout_s, 300),
            bundle=bundle,
            prefix="cargo",
        )

    def _run(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        timeout: int,
        bundle: Path,
        prefix: str,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        (bundle / f"{prefix}.stdout.log").write_text(result.stdout)
        (bundle / f"{prefix}.stderr.log").write_text(result.stderr)
        if result.returncode != 0:
            raise CommandError(prefix, result.returncode, result.stdout, result.stderr)
        return result

    def _result(
        self,
        candidate: Candidate,
        budget: Budget,
        seed: int,
        bundle: Path,
        *,
        success: bool,
        primary_score: float,
        secondary: dict[str, float],
    ) -> EvalResult:
        result = EvalResult(
            candidate_id=candidate.candidate_id,
            success=success,
            primary_score=primary_score,
            secondary_scores=secondary,
            cost=CostMetrics(wall_s=secondary.get("wall_s", 0.0), calls=1),
            trace_bundle_dir=bundle,
            evaluator_version=self.version,
            seed=seed,
            dataset_version="ecdsa.fail",
            stage=budget.stage,
        )
        (bundle / "result.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return result


def _parse_build_stdout(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    peak = re.search(r"DEBUG peak_qubits=(\d+)", stdout)
    emitted = re.search(r"emitted ops\s*:\s*(\d+)", stdout)
    if peak:
        out["qubits"] = float(peak.group(1))
    if emitted:
        out["emitted_ops"] = float(emitted.group(1))
    return out


def _parse_eval_stdout(stdout: str) -> dict[str, float]:
    fields = {
        "classical_mismatches": r"classical mismatches\s*:\s*(\d+)",
        "phase_garbage_batches": r"phase-garbage batches\s*:\s*(\d+)",
        "ancilla_garbage_batches": r"ancilla-garbage batches\s*:\s*(\d+)",
    }
    out = {}
    for key, pattern in fields.items():
        match = re.search(pattern, stdout)
        if match:
            out[key] = float(match.group(1))
    return out


def _parse_eval_header(stdout: str) -> dict[str, float]:
    fields = {
        "loaded_ops": r"loaded ops\s*:\s*(\d+)",
        "qubits": r"qubits\s*:\s*(\d+)",
        "bits": r"bits\s*:\s*(\d+)",
    }
    out = {}
    for key, pattern in fields.items():
        match = re.search(pattern, stdout)
        if match:
            out[key] = float(match.group(1))
    return out
