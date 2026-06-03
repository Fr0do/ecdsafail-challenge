"""Minimal Claude/Codex CLI bridge for local ecdsa.fail evolution."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


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
    openrouter_api_key_file: str | None = None,
    openrouter_max_tokens: int = 2048,
    openrouter_prompt_usd_per_token: float = 0.0,
    openrouter_completion_usd_per_token: float = 0.0,
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
    if backend == "openrouter":
        return _run_openrouter(
            model=model,
            prompt=prompt,
            cwd=cwd,
            artifact_dir=artifact_dir,
            timeout_s=timeout_s,
            system_prompt=COMPACT_SYSTEM_PROMPT,
            api_key_file=openrouter_api_key_file,
            max_tokens=openrouter_max_tokens,
            prompt_usd_per_token=openrouter_prompt_usd_per_token,
            completion_usd_per_token=openrouter_completion_usd_per_token,
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


def _run_openrouter(
    *,
    model: str,
    prompt: str,
    cwd: Path,
    artifact_dir: Path,
    timeout_s: int,
    system_prompt: str,
    api_key_file: str | None,
    max_tokens: int,
    prompt_usd_per_token: float,
    completion_usd_per_token: float,
) -> AgentRunResult:
    api_key = _read_openrouter_key(api_key_file)
    if not api_key:
        return AgentRunResult(
            backend="openrouter",
            model=model,
            returncode=1,
            stdout="",
            stderr="missing OPENROUTER_API_KEY or readable API key file",
            text="",
            wall_s=0.0,
        )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": max(256, int(max_tokens)),
    }
    (artifact_dir / "agent.request.json").write_text(
        json.dumps({**body, "messages": "[redacted prompt written separately]"}, indent=2)
    )
    req = request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Fr0do/ecdsafail-challenge",
            "X-Title": "ecdsafail-challenge-evomcp",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            stdout = response.read().decode("utf-8", errors="replace")
            returncode = 0
            stderr = ""
    except error.HTTPError as exc:
        stdout = exc.read().decode("utf-8", errors="replace")
        returncode = 1
        stderr = f"HTTP {exc.code}: {exc.reason}"
    except error.URLError as exc:
        stdout = ""
        returncode = 1
        stderr = str(exc.reason)
    wall_s = time.monotonic() - started
    (artifact_dir / "agent.stdout.log").write_text(stdout)
    (artifact_dir / "agent.stderr.log").write_text(stderr)

    raw: dict[str, Any] = {}
    text = stdout.strip()
    try:
        raw = json.loads(stdout)
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, list):
            text = "".join(str(part.get("text", part)) for part in content)
        else:
            text = str(content or "")
    except (json.JSONDecodeError, IndexError, TypeError):
        raw = {}

    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    usd = _as_float(usage.get("cost") or raw.get("cost"))
    if usd == 0.0 and (prompt_usd_per_token or completion_usd_per_token):
        usd = input_tokens * prompt_usd_per_token + output_tokens * completion_usd_per_token
    return AgentRunResult(
        backend="openrouter",
        model=model,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        text=text.strip(),
        wall_s=wall_s,
        usd=usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=raw,
    )


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


def _read_openrouter_key(api_key_file: str | None) -> str:
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    candidates = []
    if api_key_file:
        candidates.append(Path(api_key_file).expanduser())
    candidates.append(Path("/Users/mkurkin/experiments/projects/openrouter.txt"))
    for path in candidates:
        if path.exists():
            return path.read_text().strip()
    return ""


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
