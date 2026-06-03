"""Minimal Claude/Codex CLI bridge for local ecdsa.fail evolution."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COMPACT_SYSTEM_PROMPT = (
    "You are a local AlphaEvolve-style mutation planner for a Rust quantum "
    "circuit benchmark. Return only valid JSON. Prefer testable, narrow edits "
    "that preserve correctness. Do not call tools."
)


@dataclass
class AgentRunResult:
    backend: str
    model: str
    returncode: int
    stdout: str
    stderr: str
    text: str
    wall_s: float
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not bool(self.raw.get("is_error"))


def run_agent_plan(
    *,
    backend: str,
    model: str,
    prompt: str,
    cwd: Path,
    artifact_dir: Path,
    timeout_s: int,
    max_usd: float | None = None,
    tools: str = "",
    allowed_tools: str = "",
) -> AgentRunResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt.txt").write_text(prompt)

    if backend == "mock":
        started = time.monotonic()
        proposals = [
            {
                "hypothesis": "Reduce peak qubits by shortening a live range, not by weakening checks.",
                "edit_plan": [
                    "Inspect materialized dialog-GCD apply chunks and terminal reacquire sites.",
                    "Move one scratch allocation across a clean uncompute boundary or try a reroll island.",
                ],
                "allowed_files": ["src/point_add/mod.rs", "src/point_add/memory/"],
                "eval_commands": [
                    "cargo build --release --locked --bin build_circuit --bin eval_circuit",
                    "TRACE_PEAK=1 ./target/release/build_circuit",
                    "./target/release/eval_circuit --note agentic-smoke",
                ],
                "expected_score_effect": "Lower qubits below 1434 with zero classical/phase/ancilla garbage.",
                "failure_modes": [
                    "support island miss",
                    "phase garbage",
                    "increased Toffoli outweighs qubit drop",
                ],
                "risk_controls": ["compare score.json", "reject score tampering", "preserve 9024-shot eval"],
            },
            {
                "hypothesis": "A primitive-cost edit should only be accepted if eval_circuit stays clean.",
                "edit_plan": [
                    "Inspect a single high-frequency Toffoli primitive in point_add.",
                    "Replace it with an equivalent lower-Toffoli sequence behind a local guard.",
                ],
                "allowed_files": ["src/point_add/primitive_costs.rs", "src/point_add/mod.rs"],
                "eval_commands": [
                    "cargo build --release --locked --bin build_circuit --bin eval_circuit",
                    "TRACE_PEAK=1 ./target/release/build_circuit",
                    "./target/release/eval_circuit --note agentic-toffoli",
                ],
                "expected_score_effect": "Reduce Toffoli count while keeping peak qubits and garbage checks clean.",
                "failure_modes": ["Toffoli count unchanged", "phase garbage", "ancilla garbage"],
                "risk_controls": ["diff score.json", "reject if eval_circuit reports any mismatch"],
            },
            {
                "hypothesis": "An explicit guardrail can reject phase garbage before expensive reroll scoring.",
                "edit_plan": [
                    "Add a memory note describing the expected clean-island invariant.",
                    "Run build_circuit and eval_circuit with the note tag before any patch promotion.",
                ],
                "allowed_files": ["src/point_add/memory/agentic_guard.md", "src/bin/eval_circuit.rs"],
                "eval_commands": [
                    "cargo build --release --locked --bin build_circuit --bin eval_circuit",
                    "TRACE_PEAK=1 ./target/release/build_circuit",
                    "./target/release/eval_circuit --note agentic-guard",
                ],
                "expected_score_effect": "No direct score change; fewer bad candidates reach expensive scoring.",
                "failure_modes": ["guard note is not actionable", "eval_circuit misses phase garbage"],
                "risk_controls": ["do not change score.json semantics", "preserve strict eval checks"],
            },
        ]
        text = json.dumps(
            {"proposals": proposals},
            indent=2,
        )
        return AgentRunResult(
            backend=backend,
            model=model,
            returncode=0,
            stdout=text,
            stderr="",
            text=text,
            wall_s=time.monotonic() - started,
        )

    if backend in {"claude_subscription", "claude_bare"}:
        return _run_claude(
            backend=backend,
            model=model,
            prompt=prompt,
            cwd=cwd,
            artifact_dir=artifact_dir,
            timeout_s=timeout_s,
            max_usd=max_usd,
            tools=tools,
            allowed_tools=allowed_tools,
        )
    if backend == "codex":
        return _run_codex(
            model=model,
            prompt=prompt,
            cwd=cwd,
            artifact_dir=artifact_dir,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unknown backend: {backend}")


def _run_claude(
    *,
    backend: str,
    model: str,
    prompt: str,
    cwd: Path,
    artifact_dir: Path,
    timeout_s: int,
    max_usd: float | None,
    tools: str,
    allowed_tools: str,
) -> AgentRunResult:
    cmd = ["claude"]
    if backend == "claude_bare":
        cmd.append("--bare")
    cmd.extend(
        [
            "-p",
            prompt,
            "--model",
            model,
            "--effort",
            "low",
            "--system-prompt",
            COMPACT_SYSTEM_PROMPT,
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--setting-sources",
            "user",
            "--output-format",
            "json",
            "--max-turns",
            "3",
            "--prompt-suggestions",
            "false",
            "--no-session-persistence",
        ]
    )
    if max_usd is not None and max_usd > 0:
        cmd.extend(["--max-budget-usd", f"{max_usd:.4f}"])
    if tools:
        cmd.extend(["--tools", tools, "--permission-mode", "dontAsk"])
        if allowed_tools:
            cmd.extend(["--allowedTools", allowed_tools])
    else:
        cmd.extend(["--tools", ""])
    return _run_json_command(cmd, cwd=cwd, artifact_dir=artifact_dir, timeout_s=timeout_s)


def _run_codex(*, model: str, prompt: str, cwd: Path, artifact_dir: Path, timeout_s: int) -> AgentRunResult:
    last_message = artifact_dir / "codex-last-message.txt"
    cmd = [
        "codex",
        "exec",
        "--config",
        'model_reasoning_effort="low"',
        "--sandbox",
        "read-only",
        "--cd",
        str(cwd),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
        "--output-last-message",
        str(last_message),
        prompt,
    ]
    if model and model not in {"default", "codex-default"}:
        cmd[2:2] = ["--model", model]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    wall_s = time.monotonic() - started
    (artifact_dir / "agent.stdout.log").write_text(proc.stdout)
    (artifact_dir / "agent.stderr.log").write_text(proc.stderr)
    text = last_message.read_text() if last_message.exists() else proc.stdout
    return AgentRunResult("codex", model, proc.returncode, proc.stdout, proc.stderr, text, wall_s)


def _run_json_command(cmd: list[str], *, cwd: Path, artifact_dir: Path, timeout_s: int) -> AgentRunResult:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    wall_s = time.monotonic() - started
    (artifact_dir / "agent.stdout.log").write_text(proc.stdout)
    (artifact_dir / "agent.stderr.log").write_text(proc.stderr)
    raw: dict[str, Any] = {}
    text = proc.stdout.strip()
    try:
        raw = json.loads(proc.stdout)
        text = str(raw.get("result") or "")
    except json.JSONDecodeError:
        raw = {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    model = str(raw.get("model") or "")
    if not model and "--model" in cmd:
        model = str(cmd[cmd.index("--model") + 1])
    return AgentRunResult(
        backend="claude",
        model=model,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        text=text.strip(),
        wall_s=wall_s,
        usd=float(raw.get("total_cost_usd") or 0.0),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        raw=raw,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            line for line in stripped.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("agent output JSON is not an object")
    return parsed
