"""Agentic plan evaluator for ecdsa.fail circuit-score attacks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evomcp.pipeline import Budget, Candidate, CostMetrics, EvalResult, FailureClass
from evomcp.pipeline.evaluator import materialize_prog_genome

from optim.agentic import extract_json_object, run_agent_plan


ROOT = Path(__file__).resolve().parents[2]

TASKS: dict[str, dict[str, str]] = {
    "reduce_peak_qubits": {
        "goal": "Find a narrow Rust mutation that can reduce the 1434-qubit peak.",
        "surface": "src/point_add/mod.rs, src/point_add/*dialog*, src/point_add/memory/",
    },
    "reduce_toffoli": {
        "goal": "Find a correctness-preserving Toffoli reduction below 1,729,565.",
        "surface": "src/point_add/mod.rs, src/point_add/primitive_costs.rs",
    },
    "reroll_clean_island": {
        "goal": "Search for a reroll/width island that keeps eval_circuit clean.",
        "surface": "environment knobs, src/point_add/memory/",
    },
    "phase_garbage_guard": {
        "goal": "Propose a guardrail that rejects phase/ancilla garbage earlier in the loop.",
        "surface": "src/bin/eval_circuit.rs, src/point_add/*",
    },
}

REQUIRED_KEYS = {
    "hypothesis",
    "edit_plan",
    "allowed_files",
    "eval_commands",
    "expected_score_effect",
    "failure_modes",
    "risk_controls",
}


class AgenticEcdsaFailEvaluator:
    """Score local LLM mutation plans before expensive Rust evaluation."""

    version = "ecdsa.fail-agentic-plan-v7"

    def __init__(self, project_root: Path = ROOT):
        self.project_root = project_root

    def evaluate(self, candidate: Candidate, budget: Budget, seed: int, *, run_dir: Path) -> EvalResult:
        started = time.monotonic()
        bundle = run_dir / f"{candidate.candidate_id[:12]}-seed{seed}"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2))
        (bundle / "budget.json").write_text(json.dumps(budget.to_dict(), indent=2))

        effective = materialize_prog_genome(candidate, budget)
        (bundle / "inputs.json").write_text(json.dumps(effective, indent=2, sort_keys=True))
        task_id = str(effective.get("agent_task_id", "reduce_peak_qubits"))
        task = TASKS.get(task_id, TASKS["reduce_peak_qubits"])
        backend = str(effective.get("AGENT_BACKEND", effective.get("agent_backend", "mock")))
        model = str(effective.get("AGENT_MODEL", effective.get("agent_model", "haiku")))
        max_usd = _as_float(effective.get("AGENT_MAX_USD", 0.03))
        mode = str(effective.get("AGENT_MODE", effective.get("agent_mode", "plan_only")))
        num_proposals = _clamp_int(effective.get("AGENT_NUM_PROPOSALS", 1), default=1, lo=1, hi=8)
        if mode != "plan_only":
            return _penalized(candidate, budget, seed, "only plan_only is enabled", bundle=bundle)

        prompt = _build_prompt(task_id, task, self.project_root, num_proposals)
        try:
            agent = run_agent_plan(
                backend=backend,
                model=model,
                prompt=prompt,
                cwd=self.project_root,
                artifact_dir=bundle,
                timeout_s=budget.timeout_s,
                max_usd=max_usd,
            )
            (bundle / "agent.raw.json").write_text(json.dumps(agent.raw, indent=2, default=str))
            if not agent.ok:
                return _penalized(
                    candidate,
                    budget,
                    seed,
                    f"agent failed rc={agent.returncode}: {agent.stderr[-500:] or agent.text[-500:]}",
                    bundle=bundle,
                )
            parsed = extract_json_object(agent.text)
            (bundle / "agent.parsed.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
            proposals = _normalise_proposals(parsed)
            if not proposals:
                return _penalized(candidate, budget, seed, "agent returned no usable proposals", bundle=bundle)

            scored: list[dict[str, Any]] = []
            for index, proposal in enumerate(proposals):
                proposal_metrics = _score_plan(proposal, task_id, self.project_root)
                scored.append(
                    {
                        "index": index,
                        "score": proposal_metrics["plan_score"],
                        "metrics": proposal_metrics,
                        "plan": proposal,
                    }
                )
            best = max(scored, key=lambda item: float(item["score"]))
            best_plan = best["plan"]
            (bundle / "proposals.json").write_text(json.dumps(scored, indent=2, ensure_ascii=False))
            (bundle / "plan.json").write_text(json.dumps(best_plan, indent=2, ensure_ascii=False))

            metrics = dict(best["metrics"])
            proposal_count = len(scored)
            metrics.update(
                {
                    "proposal_count": float(proposal_count),
                    "requested_proposal_count": float(num_proposals),
                    "best_proposal_index": float(best["index"]),
                    "mean_proposal_score": sum(float(item["score"]) for item in scored) / proposal_count,
                    "model_calls_per_proposal": 1.0 / proposal_count,
                    "agent_usd_per_proposal": agent.usd / proposal_count,
                    "wall_s": time.monotonic() - started,
                    "agent_wall_s": agent.wall_s,
                    "agent_usd": agent.usd,
                    "input_tokens": float(agent.input_tokens),
                    "output_tokens": float(agent.output_tokens),
                }
            )
            result = EvalResult(
                candidate_id=candidate.candidate_id,
                success=True,
                primary_score=metrics["plan_score"],
                secondary_scores=metrics,
                cost=CostMetrics(
                    usd=agent.usd,
                    wall_s=metrics["wall_s"],
                    calls=1,
                    input_tokens=agent.input_tokens,
                    output_tokens=agent.output_tokens,
                ),
                trace_bundle_dir=bundle,
                evaluator_version=self.version,
                seed=seed,
                dataset_version="ecdsa.fail-local",
                stage=budget.stage,
            )
            (bundle / "result.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
            return result
        except Exception as exc:  # noqa: BLE001
            (bundle / "failure.txt").write_text(str(exc))
            return _penalized(candidate, budget, seed, str(exc), bundle=bundle)


def _build_prompt(task_id: str, task: dict[str, str], project_root: Path, num_proposals: int) -> str:
    proposal_schema = {
        "hypothesis": "one sentence",
        "edit_plan": ["ordered concrete edit or experiment"],
        "allowed_files": ["path"],
        "eval_commands": ["command"],
        "expected_score_effect": "specific effect on score/qubits/toffoli",
        "failure_modes": ["what would falsify this"],
        "risk_controls": ["deterministic guard"],
    }
    schema = {"proposals": [proposal_schema]}
    return f"""
Return only JSON with this schema:
{json.dumps(schema, indent=2)}

Project: ecdsa.fail quantum point-addition benchmark.
Baseline verified on this branch:
- score = 2,480,196,210 = 1,729,565 Toffoli * 1434 qubits.
- build_circuit emits about 11,126,484 ops.
- eval_circuit passed 9024 shots with 0 classical mismatches, 0 phase garbage,
  and 0 ancilla garbage.
- Active pressure points from TRACE include materialized dialog-GCD special
  chunks, terminal reacquire sites, quotient/product pair blocks, and reroll
  islands.

Hard constraints:
- Do not weaken eval checks, change score.json semantics, or fake metrics.
- Correctness must remain clean under eval_circuit.
- Code edits must target existing Rust files from the relevant-file list.
- New files are allowed only as memory/*.md notes, not as Rust implementation files.
- Rust is required only for final ground-truth scoring; this stage is a cheap
  hypothesis filter.

Mutation task id: {task_id}
Goal: {task["goal"]}
Allowed surface hint: {task["surface"]}
Existing relevant files:
{_compact_file_context(project_root, ("src/point_add", "src/bin", "configs"))}

Hard output budget:
- Return minified JSON, with no markdown and no commentary.
- Generate exactly {num_proposals} diverse proposals.
- Every string must be at most 160 characters.
- Each proposal must use 2-3 edit_plan items, at most 3 allowed_files,
  exactly 3 eval_commands, at most 2 failure_modes, and at most 2 risk_controls.

Each proposal must be one narrow mutation or experiment. It must be falsifiable
by the listed commands and should name the expected qubit/Toffoli/score
direction. Keep proposals non-overlapping so a local evaluator can pick the best
one. Do not name files that are absent from the existing relevant-file list
unless the plan explicitly creates them.
Prefer these ground-truth commands:
- cargo build --release --locked --bin build_circuit --bin eval_circuit
- TRACE_PEAK=1 ./target/release/build_circuit
- ./target/release/eval_circuit --note agentic
""".strip()


def _normalise_proposals(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = parsed.get("proposals")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(parsed.get("proposal"), dict):
        return [parsed["proposal"]]
    if REQUIRED_KEYS & set(parsed):
        return [parsed]
    return []


def _score_plan(plan: dict[str, Any], task_id: str, project_root: Path) -> dict[str, float]:
    score = 0.0
    missing = REQUIRED_KEYS - set(plan)
    if not missing:
        score += 25.0
    score += min(15.0, 3.0 * len(plan.get("edit_plan", []) or []))
    score += min(15.0, 5.0 * len(plan.get("eval_commands", []) or []))
    text = json.dumps(plan, sort_keys=True).lower()
    for token in ("1434", "toffoli", "eval_circuit", "build_circuit", "garbage", "score"):
        if token in text:
            score += 5.0
    task_tokens = {
        "reduce_peak_qubits": ("qubit", "peak", "live"),
        "reduce_toffoli": ("toffoli", "gate", "primitive"),
        "reroll_clean_island": ("reroll", "island", "clean"),
        "phase_garbage_guard": ("phase", "ancilla", "guard"),
    }[task_id]
    score += 5.0 * sum(token in text for token in task_tokens)
    forbidden = ("score.json", "skip eval", "disable", "always true", "git push", "rm ")
    if any(token in text for token in forbidden):
        score -= 30.0
    commands = " ".join(str(cmd) for cmd in plan.get("eval_commands", []) or []).lower()
    if "build_circuit" not in commands:
        score -= 10.0
    if "eval_circuit" not in commands:
        score -= 10.0
    if "cargo test" in commands and "eval_circuit" not in commands:
        score -= 5.0
    for path in plan.get("allowed_files", []) or []:
        if Path(str(path)).name.startswith("._"):
            score -= 20.0
    absent = _absent_allowed_files(plan, project_root)
    score -= 7.0 * len(absent)
    return {
        "plan_score": max(0.0, min(100.0, score)),
        "missing_required_keys": float(len(missing)),
        "edit_steps": float(len(plan.get("edit_plan", []) or [])),
        "eval_commands": float(len(plan.get("eval_commands", []) or [])),
        "absent_allowed_files": float(len(absent)),
    }


def _compact_file_context(project_root: Path, prefixes: tuple[str, ...], limit: int = 100) -> str:
    files: list[str] = []
    for prefix in prefixes:
        base = project_root / prefix
        if base.is_file():
            files.append(prefix)
        elif base.is_dir():
            for path in sorted(base.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and not path.name.startswith("._")
                ):
                    files.append(path.relative_to(project_root).as_posix())
    files = files[:limit]
    return "\n".join(f"- {path}" for path in files)


def _absent_allowed_files(plan: dict[str, Any], project_root: Path) -> list[str]:
    absent: list[str] = []
    plan_text = json.dumps(plan, sort_keys=True).lower()
    for raw in plan.get("allowed_files", []) or []:
        path = str(raw)
        if "*" in path or path.endswith("/") or path.startswith("environment"):
            continue
        if path.endswith(".md") and path.lower() in plan_text and ("create" in plan_text or "add" in plan_text):
            continue
        if not (project_root / path).exists():
            absent.append(path)
    return absent


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _penalized(
    candidate: Candidate,
    budget: Budget,
    seed: int,
    message: str,
    *,
    bundle: Path | None = None,
) -> EvalResult:
    result = EvalResult.penalized(
        candidate.candidate_id,
        FailureClass.RUNTIME,
        message,
        stage=budget.stage,
        evaluator_version=AgenticEcdsaFailEvaluator.version,
        seed=seed,
        dataset_version="ecdsa.fail-local",
    )
    result.trace_bundle_dir = bundle
    return result
