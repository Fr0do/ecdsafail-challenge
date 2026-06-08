#!/usr/bin/env python3
"""Promote agentic plans into isolated Rust patches and trusted evaluations.

The GigaEvo-lite marathon is intentionally plan-only. This worker is the next
stage: poll plan archives, ask a coding agent to apply one narrow patch in a
linked git worktree, then run the real ecdsa.fail build/eval gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required; run with: uv run --with pyyaml ...") from exc


@dataclass(frozen=True)
class PlanRecord:
    source: str
    source_id: str
    run_id: str
    candidate_id: str
    trace_dir: str
    score: float
    mean_score: float | None
    task_id: str
    context_profile: str
    strategy_profile: str
    prior_id: str
    plan: dict[str, Any]
    result: dict[str, Any]
    plan_hash: str


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--once", action="store_true", help="run at most one trial, then exit")
    parser.add_argument("--dry-run", action="store_true", help="show pending plans without patching")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    output_dir = project_root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trials").mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"

    lock_fh = None
    if not args.dry_run and bool(cfg.get("single_worker_lock", True)):
        lock_fh = _acquire_worker_lock(output_dir, cfg, events_path)
        if lock_fh is None:
            return

    db = sqlite3.connect(output_dir / "patch_eval.sqlite")
    try:
        db.row_factory = sqlite3.Row
        _init_db(db)
        _append_event(events_path, {"type": "worker_start", "config": str(args.config), "time": _now()})

        started = time.monotonic()
        max_wall_s = float(cfg.get("max_wall_hours", 12.0)) * 3600.0
        max_trials = int(cfg.get("max_trials", 1))
        poll_s = float(cfg.get("poll_s", 60.0))
        stop_file = output_dir / str(cfg.get("stop_file", "STOP"))
        completed_trials = 0

        while completed_trials < max_trials and not _should_stop(started, max_wall_s, stop_file):
            plans = _collect_plans(project_root, cfg)
            pending = _pending_plans(db, plans, cfg)
            if args.dry_run:
                for plan in pending[: int(cfg.get("top_k", 20))]:
                    print(
                        f"{plan.score:.1f}\t{plan.task_id}\t{plan.strategy_profile}\t"
                        f"{plan.prior_id}\t{plan.candidate_id}\t{plan.plan.get('hypothesis', '')}"
                    )
                break
            if not pending:
                _append_event(events_path, {"type": "poll_empty", "plans": len(plans), "time": _now()})
                if args.once:
                    break
                time.sleep(poll_s)
                continue

            plan = pending[0]
            trial_id = _next_trial_id(db, plan)
            _append_event(
                events_path,
                {
                    "type": "trial_start",
                    "trial_id": trial_id,
                    "plan_hash": plan.plan_hash,
                    "candidate_id": plan.candidate_id,
                    "score": plan.score,
                    "time": _now(),
                },
            )
            _record_trial_start(db, trial_id, plan, cfg)
            summary = _run_trial(project_root, output_dir, cfg, trial_id, plan)
            _record_trial_finish(db, trial_id, summary)
            _append_event(events_path, {"type": "trial_finish", "trial_id": trial_id, "summary": summary, "time": _now()})
            completed_trials += 1
            if args.once:
                break
            time.sleep(float(cfg.get("sleep_between_trials_s", 5.0)))

        _append_event(events_path, {"type": "worker_exit", "completed_trials": completed_trials, "time": _now()})
    finally:
        db.close()
        if lock_fh is not None:
            _release_worker_lock(lock_fh)


def _acquire_worker_lock(output_dir: Path, cfg: dict[str, Any], events_path: Path):
    lock_name = str(cfg.get("worker_lock_file", "worker.lock"))
    lock_path = output_dir / lock_name
    lock_fh = lock_path.open("a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.seek(0)
        holder = lock_fh.read().strip()
        payload = {"type": "worker_lock_busy", "lock_path": str(lock_path), "holder": holder, "time": _now()}
        _append_event(events_path, payload)
        print(f"worker lock busy: {lock_path}; holder={holder}", file=sys.stderr)
        lock_fh.close()
        return None
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(json.dumps({"pid": os.getpid(), "time": _now()}, sort_keys=True) + "\n")
    lock_fh.flush()
    return lock_fh


def _release_worker_lock(lock_fh) -> None:
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fh.close()


def _run_trial(
    project_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    trial_id: str,
    plan: PlanRecord,
) -> dict[str, Any]:
    trial_dir = output_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "plan.json").write_text(json.dumps(plan.plan, indent=2, ensure_ascii=False))
    (trial_dir / "source_result.json").write_text(json.dumps(plan.result, indent=2, ensure_ascii=False))
    (trial_dir / "source.json").write_text(json.dumps(plan.__dict__, indent=2, ensure_ascii=False))

    worktree_root = Path(cfg["worktree_root"]).expanduser()
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_root / trial_id
    branch_prefix = str(cfg.get("branch_prefix", "agentic-patch-eval/"))
    branch = f"{branch_prefix}{trial_id}"
    created_worktree = False
    try:
        _create_worktree(project_root, worktree_path, branch, cfg)
        created_worktree = True
        env_specs = _env_only_specs(plan, cfg)
        if env_specs:
            eval_summary = _run_env_only_eval(worktree_path, trial_dir, cfg, trial_id, env_specs)
            status = _status_from_eval(eval_summary, cfg)
            return _summary(
                status,
                trial_id,
                plan,
                branch,
                worktree_path,
                [],
                patcher=CommandResult(0, "env_only", "", 0.0),
                eval_summary=eval_summary,
            )

        patcher = _run_patcher(worktree_path, trial_dir, cfg, plan, trial_id)
        changed_files = _changed_files(worktree_path)
        (trial_dir / "changed_files.json").write_text(json.dumps(changed_files, indent=2))
        _write_patch(worktree_path, trial_dir, changed_files)

        if patcher.returncode != 0 or patcher.timed_out:
            return _summary(
                "patcher_failed",
                trial_id,
                plan,
                branch,
                worktree_path,
                changed_files,
                error=_tail(patcher.stderr or patcher.stdout),
                patcher=patcher,
            )
        policy_error = _file_policy_error(changed_files, cfg)
        if policy_error:
            return _summary(
                "rejected",
                trial_id,
                plan,
                branch,
                worktree_path,
                changed_files,
                error=policy_error,
                patcher=patcher,
            )
        if not changed_files:
            return _summary(
                "no_patch",
                trial_id,
                plan,
                branch,
                worktree_path,
                changed_files,
                error="agent left no worktree changes",
                patcher=patcher,
            )
        if not any(path.endswith(".rs") for path in changed_files):
            return _summary(
                "docs_only",
                trial_id,
                plan,
                branch,
                worktree_path,
                changed_files,
                error="no Rust changes to evaluate",
                patcher=patcher,
            )

        if not bool(cfg.get("evaluator", {}).get("enabled", True)):
            return _summary(
                "patch_only",
                trial_id,
                plan,
                branch,
                worktree_path,
                changed_files,
                patcher=patcher,
            )
        eval_summary = _run_trusted_eval(worktree_path, trial_dir, cfg, trial_id)
        status = _status_from_eval(eval_summary, cfg)
        return _summary(
            status,
            trial_id,
            plan,
            branch,
            worktree_path,
            changed_files,
            patcher=patcher,
            eval_summary=eval_summary,
        )
    except Exception as exc:  # noqa: BLE001
        return _summary(
            "worker_error",
            trial_id,
            plan,
            branch,
            worktree_path,
            _changed_files(worktree_path) if worktree_path.exists() else [],
            error=str(exc),
        )
    finally:
        if created_worktree and bool(cfg.get("cleanup_worktrees", False)):
            _remove_worktree(project_root, worktree_path)
            _delete_branch(project_root, branch)


def _run_patcher(
    worktree_path: Path,
    trial_dir: Path,
    cfg: dict[str, Any],
    plan: PlanRecord,
    trial_id: str,
) -> CommandResult:
    patcher = dict(cfg.get("patcher", {}))
    backend = str(patcher.get("backend", "codex"))
    prompt = _patch_prompt(plan, cfg, trial_id)
    (trial_dir / "patch_prompt.txt").write_text(prompt)
    if backend == "mock":
        note = worktree_path / "src/point_add/memory/agentic_patch_eval_smoke.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(f"# Patch Eval Smoke\n\ntrial: `{trial_id}`\nplan: {plan.plan.get('hypothesis', '')}\n")
        return CommandResult(0, '{"status":"patched","summary":"mock note"}', "", 0.0)

    if backend != "codex":
        raise ValueError(f"unsupported patcher backend: {backend}")
    last_message = trial_dir / "codex-last-message.txt"
    cmd = [
        "codex",
        "--ask-for-approval",
        "never",
    ]
    if patcher.get("fast_mode", False):
        cmd.extend(["--enable", "fast_mode"])
    model = str(patcher.get("model", ""))
    if model and model not in {"default", "codex-default"}:
        cmd.extend(["--model", model])
    cmd.extend(
        [
            "exec",
            "--config",
            f"model_reasoning_effort=\"{patcher.get('reasoning_effort', 'high')}\"",
            "--sandbox",
            str(patcher.get("sandbox", "workspace-write")),
            "--cd",
            str(worktree_path),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            str(last_message),
            prompt,
        ]
    )
    (trial_dir / "patcher_command.json").write_text(json.dumps(cmd, indent=2))
    return _run_command(cmd, cwd=worktree_path, timeout=int(patcher.get("timeout_s", 2400)), prefix="patcher", out_dir=trial_dir)


def _run_trusted_eval(worktree_path: Path, trial_dir: Path, cfg: dict[str, Any], trial_id: str) -> dict[str, Any]:
    evaluator = dict(cfg.get("evaluator", {}))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(evaluator.get("env", {})).items()})
    build_timeout = int(evaluator.get("build_timeout_s", 900))
    circuit_timeout = int(evaluator.get("circuit_timeout_s", 900))
    eval_timeout = int(evaluator.get("eval_timeout_s", 1200))

    cargo = _run_command(
        ["cargo", "build", "--release", "--locked", "--bin", "build_circuit", "--bin", "eval_circuit"],
        cwd=worktree_path,
        timeout=build_timeout,
        prefix="cargo",
        out_dir=trial_dir,
        env=env,
    )
    if cargo.returncode != 0 or cargo.timed_out:
        return {"valid": False, "stage": "cargo", "error": _tail(cargo.stderr or cargo.stdout), "commands": _command_dict(cargo)}

    build_bin = worktree_path / "target/release/build_circuit"
    eval_bin = worktree_path / "target/release/eval_circuit"
    build_env = dict(env)
    build_env["TRACE_PEAK"] = str(evaluator.get("trace_peak", "1"))
    score_path = worktree_path / "score.json"
    score_path.unlink(missing_ok=True)
    build = _run_command(
        [str(build_bin)],
        cwd=worktree_path,
        timeout=circuit_timeout,
        prefix="build_circuit",
        out_dir=trial_dir,
        env=build_env,
    )
    if build.returncode != 0 or build.timed_out:
        return {
            "valid": False,
            "stage": "build_circuit",
            "error": _tail(build.stderr or build.stdout),
            "build_metrics": _parse_build_stdout(build.stdout),
            "commands": {"cargo": _command_dict(cargo), "build_circuit": _command_dict(build)},
        }

    note = f"patch-eval {trial_id}"
    eval_run = _run_command(
        [str(eval_bin), "--note", note],
        cwd=worktree_path,
        timeout=eval_timeout,
        prefix="eval_circuit",
        out_dir=trial_dir,
        env=env,
    )
    score_json: dict[str, Any] = {}
    if score_path.exists():
        shutil.copy2(score_path, trial_dir / "score.json")
        try:
            score_json = json.loads(score_path.read_text())
        except json.JSONDecodeError:
            score_json = {}
    results_path = worktree_path / "results.tsv"
    if results_path.exists():
        shutil.copy2(results_path, trial_dir / "results.tsv")
    valid = eval_run.returncode == 0 and not eval_run.timed_out and bool(score_json)
    return {
        "valid": valid,
        "baseline_score": evaluator.get("baseline_score", 2_479_548_042),
        "stage": "eval_circuit",
        "error": "" if valid else _tail(eval_run.stderr or eval_run.stdout),
        "score": score_json.get("score"),
        "metrics": score_json.get("metrics", {}),
        "build_metrics": _parse_build_stdout(build.stdout),
        "eval_metrics": _parse_eval_stdout(eval_run.stdout),
        "commands": {
            "cargo": _command_dict(cargo),
            "build_circuit": _command_dict(build),
            "eval_circuit": _command_dict(eval_run),
        },
    }


def _run_env_only_eval(
    worktree_path: Path,
    trial_dir: Path,
    cfg: dict[str, Any],
    trial_id: str,
    env_specs: list[dict[str, str]],
) -> dict[str, Any]:
    evaluator = dict(cfg.get("evaluator", {}))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(evaluator.get("env", {})).items()})
    build_timeout = int(evaluator.get("build_timeout_s", 900))
    circuit_timeout = int(evaluator.get("circuit_timeout_s", 900))
    eval_timeout = int(evaluator.get("eval_timeout_s", 1200))
    cargo = _run_command(
        ["cargo", "build", "--release", "--locked", "--bin", "build_circuit", "--bin", "eval_circuit"],
        cwd=worktree_path,
        timeout=build_timeout,
        prefix="cargo",
        out_dir=trial_dir,
        env=env,
    )
    if cargo.returncode != 0 or cargo.timed_out:
        return {
            "valid": False,
            "env_only": True,
            "stage": "cargo",
            "error": _tail(cargo.stderr or cargo.stdout),
            "commands": {"cargo": _command_dict(cargo)},
        }

    variants: list[dict[str, Any]] = []
    for index, spec in enumerate(env_specs):
        variant_dir = trial_dir / f"env_variant_{index:02d}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant = _run_eval_variant(
            worktree_path=worktree_path,
            out_dir=variant_dir,
            cfg=cfg,
            trial_id=f"{trial_id}-v{index:02d}",
            env_map=spec,
            circuit_timeout=circuit_timeout,
            eval_timeout=eval_timeout,
        )
        variants.append(variant)
        if bool(cfg.get("evaluator", {}).get("stop_env_variants_on_improvement", True)):
            score = _maybe_int(variant.get("score"))
            baseline = _maybe_int(evaluator.get("baseline_score"))
            if variant.get("valid") and score is not None and baseline is not None and score < baseline:
                break

    valid_variants = [item for item in variants if item.get("valid")]
    best = min(valid_variants, key=lambda item: int(item["score"])) if valid_variants else None
    if best:
        return {
            "valid": True,
            "env_only": True,
            "baseline_score": evaluator.get("baseline_score", 2_479_548_042),
            "stage": "env_eval",
            "error": "",
            "score": best.get("score"),
            "metrics": best.get("metrics", {}),
            "build_metrics": best.get("build_metrics", {}),
            "eval_metrics": best.get("eval_metrics", {}),
            "env": best.get("env", {}),
            "variants": variants,
            "commands": {"cargo": _command_dict(cargo)},
        }
    return {
        "valid": False,
        "env_only": True,
        "baseline_score": evaluator.get("baseline_score", 2_479_548_042),
        "stage": "env_eval",
        "error": _tail("\n\n".join(str(item.get("error", "")) for item in variants if item.get("error"))),
        "variants": variants,
        "commands": {"cargo": _command_dict(cargo)},
    }


def _run_eval_variant(
    *,
    worktree_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    trial_id: str,
    env_map: dict[str, str],
    circuit_timeout: int,
    eval_timeout: int,
) -> dict[str, Any]:
    evaluator = dict(cfg.get("evaluator", {}))
    base_env = os.environ.copy()
    base_env.update({str(k): str(v) for k, v in dict(evaluator.get("env", {})).items()})
    base_env.update({str(k): str(v) for k, v in env_map.items()})
    build_env = dict(base_env)
    build_env["TRACE_PEAK"] = str(evaluator.get("trace_peak", "1"))
    build_bin = worktree_path / "target/release/build_circuit"
    eval_bin = worktree_path / "target/release/eval_circuit"
    score_path = worktree_path / "score.json"
    score_path.unlink(missing_ok=True)

    build = _run_command(
        [str(build_bin)],
        cwd=worktree_path,
        timeout=circuit_timeout,
        prefix="build_circuit",
        out_dir=out_dir,
        env=build_env,
    )
    if build.returncode != 0 or build.timed_out:
        return {
            "valid": False,
            "env": env_map,
            "stage": "build_circuit",
            "error": _tail(build.stderr or build.stdout),
            "build_metrics": _parse_build_stdout(build.stdout),
            "commands": {"build_circuit": _command_dict(build)},
        }

    note = _env_note(trial_id, env_map)
    eval_run = _run_command(
        [str(eval_bin), "--note", note],
        cwd=worktree_path,
        timeout=eval_timeout,
        prefix="eval_circuit",
        out_dir=out_dir,
        env=base_env,
    )
    score_json: dict[str, Any] = {}
    if score_path.exists():
        shutil.copy2(score_path, out_dir / "score.json")
        try:
            score_json = json.loads(score_path.read_text())
        except json.JSONDecodeError:
            score_json = {}
    valid = eval_run.returncode == 0 and not eval_run.timed_out and bool(score_json)
    return {
        "valid": valid,
        "env": env_map,
        "stage": "eval_circuit",
        "error": "" if valid else _tail(eval_run.stderr or eval_run.stdout),
        "score": score_json.get("score"),
        "metrics": score_json.get("metrics", {}),
        "build_metrics": _parse_build_stdout(build.stdout),
        "eval_metrics": _parse_eval_stdout(eval_run.stdout),
        "commands": {
            "build_circuit": _command_dict(build),
            "eval_circuit": _command_dict(eval_run),
        },
    }


def _patch_prompt(plan: PlanRecord, cfg: dict[str, Any], trial_id: str) -> str:
    evaluator = dict(cfg.get("evaluator", {}))
    allow_prefixes = "\n".join(f"- {item}" for item in evaluator.get("allow_prefixes", ["src/point_add/"]))
    deny_paths = "\n".join(f"- {item}" for item in evaluator.get("deny_paths", []))
    baseline = evaluator.get("baseline_score", 2_479_548_042)
    return f"""
You are applying one narrow patch for the ecdsa.fail quantum point-addition benchmark.

Worktree trial: {trial_id}
Plan source: {plan.source_id}
Plan score: {plan.score}
Profile: task={plan.task_id} context={plan.context_profile} strategy={plan.strategy_profile} prior={plan.prior_id}

Plan JSON:
{json.dumps(plan.plan, indent=2, ensure_ascii=False)}

Hard constraints:
- Preserve the trusted harness. Do not edit eval/build binaries, scoring semantics, score.json, results.tsv, Cargo.lock, or CI.
- Only change Rust implementation files under allowed prefixes unless a tiny memory note is useful.
- Allowed implementation prefixes:
{allow_prefixes}
- Denied paths/prefixes:
{deny_paths}
- Keep the patch small and inspectable. Prefer one mechanism over a broad rewrite.
- Do not commit. Leave changes in the worktree.
- The supervisor will run:
  cargo build --release --locked --bin build_circuit --bin eval_circuit
  TRACE_PEAK=1 ./target/release/build_circuit
  ./target/release/eval_circuit --note patch-eval
- Current promoted baseline score is {baseline}; lower score is better.

Before finishing, run only cheap local checks if useful. Avoid long full evals; the supervisor handles them.
Return a compact final JSON object with keys: status, summary, changed_files, expected_effect, risk.
""".strip()


def _env_only_specs(plan: PlanRecord, cfg: dict[str, Any]) -> list[dict[str, str]]:
    text = json.dumps(plan.plan, ensure_ascii=False).lower()
    allowed_files = [str(item) for item in plan.plan.get("allowed_files", []) or []]
    env_hint = "env-only" in text or "no rust edit" in text or "no rust" in text
    only_notes = bool(allowed_files) and all(path.endswith(".md") for path in allowed_files)
    if not (env_hint or only_notes):
        return []

    specs: list[dict[str, str]] = []
    for command in plan.plan.get("eval_commands", []) or []:
        spec = _env_prefix_from_command(str(command))
        if _is_reroll_spec(spec):
            specs.append(spec)

    compare_bits = _first_value(specs, "DIALOG_GCD_COMPARE_BITS")
    if not compare_bits:
        compare_match = re.search(r"compare\s*([0-9]{2,3})|cb\s*([0-9]{2,3})", text)
        compare_bits = next((group for group in compare_match.groups() if group), "") if compare_match else ""
    for reroll, post_sub in re.findall(r"\b([0-9]{2,5})/([0-9]{2,5})\b", text):
        if compare_bits:
            specs.append(
                {
                    "DIALOG_GCD_COMPARE_BITS": compare_bits,
                    "DIALOG_REROLL": reroll,
                    "DIALOG_POST_SUB_REROLL": post_sub,
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for spec in specs:
        cleaned = {key: str(value) for key, value in spec.items() if key != "TRACE_PEAK"}
        key = tuple(sorted(cleaned.items()))
        if key and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped[: int(cfg.get("evaluator", {}).get("max_env_variants", 6))]


def _env_prefix_from_command(command: str) -> dict[str, str]:
    spec: dict[str, str] = {}
    for token in command.split():
        if "=" not in token or token.startswith("./") or token.startswith("cargo"):
            break
        key, value = token.split("=", 1)
        if re.fullmatch(r"[A-Z0-9_]+", key):
            spec[key] = value.strip("'\"")
            continue
        break
    return spec


def _is_reroll_spec(spec: dict[str, str]) -> bool:
    return bool(
        spec.get("DIALOG_GCD_COMPARE_BITS")
        or spec.get("DIALOG_REROLL")
        or spec.get("DIALOG_POST_SUB_REROLL")
    )


def _first_value(specs: list[dict[str, str]], key: str) -> str:
    for spec in specs:
        if spec.get(key):
            return str(spec[key])
    return ""


def _env_note(trial_id: str, env_map: dict[str, str]) -> str:
    compare = env_map.get("DIALOG_GCD_COMPARE_BITS", "na")
    reroll = env_map.get("DIALOG_REROLL", "na")
    post = env_map.get("DIALOG_POST_SUB_REROLL", "na")
    return f"patch-eval {trial_id} cb{compare}-r{reroll}-p{post}"


def _collect_plans(project_root: Path, cfg: dict[str, Any]) -> list[PlanRecord]:
    plans: dict[str, PlanRecord] = {}
    for plan in _collect_marathon_db(project_root, cfg):
        plans.setdefault(plan.plan_hash, plan)
    for plan in _collect_trace_plans(project_root, cfg):
        prev = plans.get(plan.plan_hash)
        if prev is None or plan.score > prev.score:
            plans[plan.plan_hash] = plan
    return sorted(plans.values(), key=lambda item: (item.score, item.mean_score or 0.0), reverse=True)


def _collect_marathon_db(project_root: Path, cfg: dict[str, Any]) -> list[PlanRecord]:
    db_path = project_root / str(cfg.get("source", {}).get("marathon_db", ""))
    if not db_path.exists():
        return []
    out: list[PlanRecord] = []
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            """
            select id, run_id, candidate_id, trace_dir, score, mean_score,
                   task_id, context_profile, strategy_profile, prior_id,
                   plan_json, result_json
            from plans
            order by score desc, coalesce(mean_score, 0) desc
            """
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        db.close()
    for row in rows:
        try:
            plan = json.loads(row["plan_json"])
            result = json.loads(row["result_json"])
        except json.JSONDecodeError:
            continue
        out.append(
            _make_plan_record(
                source="marathon_db",
                source_id=f"{db_path}:{row['id']}",
                run_id=str(row["run_id"]),
                candidate_id=str(row["candidate_id"]),
                trace_dir=str(row["trace_dir"]),
                score=float(row["score"]),
                mean_score=_maybe_float(row["mean_score"]),
                task_id=str(row["task_id"] or ""),
                context_profile=str(row["context_profile"] or ""),
                strategy_profile=str(row["strategy_profile"] or ""),
                prior_id=str(row["prior_id"] or ""),
                plan=plan,
                result=result,
            )
        )
    return out


def _collect_trace_plans(project_root: Path, cfg: dict[str, Any]) -> list[PlanRecord]:
    out: list[PlanRecord] = []
    globs = list(cfg.get("source", {}).get("trace_globs", []))
    for pattern in globs:
        for plan_path in project_root.glob(str(pattern)):
            trace_dir = plan_path.parent
            result_path = trace_dir / "result.json"
            if not result_path.exists():
                continue
            profile_path = trace_dir / "prompt_profile.json"
            try:
                plan = json.loads(plan_path.read_text())
                result = json.loads(result_path.read_text())
                profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
            except (json.JSONDecodeError, OSError):
                continue
            run_id = _infer_run_id(trace_dir)
            scores = result.get("secondary_scores", {}) if isinstance(result.get("secondary_scores"), dict) else {}
            out.append(
                _make_plan_record(
                    source="trace",
                    source_id=str(trace_dir),
                    run_id=run_id,
                    candidate_id=str(result.get("candidate_id", trace_dir.name.replace("-seed0", ""))),
                    trace_dir=str(trace_dir),
                    score=float(result.get("primary_score", 0.0)),
                    mean_score=_maybe_float(scores.get("mean_proposal_score")),
                    task_id=str(profile.get("task_id", "")),
                    context_profile=str(profile.get("context_profile", "")),
                    strategy_profile=str(profile.get("strategy_profile", "")),
                    prior_id=str(profile.get("prior_id", "")),
                    plan=plan,
                    result=result,
                )
            )
    return out


def _make_plan_record(
    *,
    source: str,
    source_id: str,
    run_id: str,
    candidate_id: str,
    trace_dir: str,
    score: float,
    mean_score: float | None,
    task_id: str,
    context_profile: str,
    strategy_profile: str,
    prior_id: str,
    plan: dict[str, Any],
    result: dict[str, Any],
) -> PlanRecord:
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return PlanRecord(
        source,
        source_id,
        run_id,
        candidate_id,
        trace_dir,
        score,
        mean_score,
        task_id,
        context_profile,
        strategy_profile,
        prior_id,
        plan,
        result,
        plan_hash,
    )


def _pending_plans(db: sqlite3.Connection, plans: list[PlanRecord], cfg: dict[str, Any]) -> list[PlanRecord]:
    min_score = float(cfg.get("min_plan_score", 0.0))
    top_k = int(cfg.get("top_k", len(plans)))
    retry_failed = bool(cfg.get("retry_failed", False))
    feedback = _feedback_profile(db, cfg)
    pending: list[PlanRecord] = []
    ordered = sorted(
        plans,
        key=lambda item: (
            _selection_score(item, feedback, cfg),
            item.mean_score or 0.0,
            item.candidate_id,
        ),
        reverse=True,
    )
    for plan in ordered:
        if plan.score < min_score:
            continue
        if not _plan_policy_compatible(plan, cfg):
            continue
        row = db.execute(
            "select status, summary_json from trials where plan_hash=? order by id desc limit 1",
            (plan.plan_hash,),
        ).fetchone()
        if row and not retry_failed:
            if _env_only_specs(plan, cfg) and not _trial_was_env_only(row["summary_json"]):
                pending.append(plan)
                if len(pending) >= top_k:
                    break
                continue
            continue
        if row and retry_failed and row["status"] not in {"patcher_failed", "worker_error"}:
            continue
        pending.append(plan)
        if len(pending) >= top_k:
            break
    return pending


def _feedback_profile(db: sqlite3.Connection, cfg: dict[str, Any]) -> dict[str, Any]:
    feedback_cfg = dict(cfg.get("feedback", {}))
    if not bool(feedback_cfg.get("enabled", True)):
        return {"families": {}}
    try:
        rows = db.execute(
            """
            select status, hypothesis, plan_json
            from trials
            where status in ('invalid', 'build_failed', 'rejected', 'no_patch', 'docs_only')
               or (status = 'valid' and coalesce(score_delta, 0) <= 0)
            order by id desc
            limit ?
            """,
            (int(feedback_cfg.get("lookback", 64)),),
        ).fetchall()
    except sqlite3.Error:
        return {"families": {}}

    families: dict[str, int] = {}
    for row in rows:
        text = f"{row['hypothesis'] or ''}\n{row['plan_json'] or ''}"
        for family in _plan_families_from_text(text):
            families[family] = families.get(family, 0) + 1
    return {"families": families}


def _selection_score(plan: PlanRecord, feedback: dict[str, Any], cfg: dict[str, Any]) -> float:
    feedback_cfg = dict(cfg.get("feedback", {}))
    families = feedback.get("families", {}) if isinstance(feedback.get("families"), dict) else {}
    penalty_per = float(feedback_cfg.get("family_penalty", 8.0))
    max_penalty = float(feedback_cfg.get("max_family_penalty", 32.0))
    text = _plan_text(plan)
    plan_families = _plan_families_from_text(text)
    penalty = min(max_penalty, penalty_per * sum(int(families.get(family, 0)) for family in plan_families))
    novelty_bonus = 0.0
    if plan_families and not any(families.get(family, 0) for family in plan_families):
        novelty_bonus += float(feedback_cfg.get("new_family_bonus", 3.0))
    if "negative sample" in text.lower() or "negative_sample" in text.lower():
        penalty += float(feedback_cfg.get("negative_sample_penalty", 4.0))
    return float(plan.score) + novelty_bonus - penalty


def _plan_text(plan: PlanRecord) -> str:
    return "\n".join(
        [
            plan.task_id,
            plan.context_profile,
            plan.strategy_profile,
            plan.prior_id,
            json.dumps(plan.plan, ensure_ascii=False, sort_keys=True),
        ]
    )


def _plan_families_from_text(text: str) -> set[str]:
    lowered = text.lower()
    families: set[str] = set()
    if (
        "compare56" in lowered
        or "compare 56" in lowered
        or "compare_bits=56" in lowered
        or "compare_bits\": \"56" in lowered
        or "pa_compare_bits=56" in lowered
        or "dialog_gcd_compare_bits=56" in lowered
    ):
        families.add("compare56")
    if "reroll" in lowered:
        families.add("reroll")
    if "raw_block" in lowered and ("compressed" in lowered or "terminal" in lowered):
        families.add("compressed_raw_block_lifetime")
    if "delay" in lowered and "product" in lowered and "cleanup" in lowered:
        families.add("delayed_product_cleanup")
    if "underflow" in lowered or "borrow" in lowered:
        families.add("underflow_cleanup")
    if (
        "phase_garbage_guard" in lowered
        or "phase guard" in lowered
        or "phase-garbage guard" in lowered
        or "phase failure boundary" in lowered
    ):
        families.add("phase_garbage")
    if "route" in lowered or "lifetime" in lowered or "stream" in lowered:
        families.add("route_lifetime")
    return families or {"uncategorized"}


def _trial_was_env_only(summary_json: str) -> bool:
    try:
        summary = json.loads(summary_json or "{}")
    except json.JSONDecodeError:
        return False
    eval_summary = summary.get("eval", {}) if isinstance(summary.get("eval"), dict) else {}
    return bool(eval_summary.get("env_only"))


def _init_db(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        create table if not exists trials (
          id integer primary key autoincrement,
          trial_id text unique not null,
          plan_hash text not null,
          source text not null,
          source_id text not null,
          run_id text not null,
          candidate_id text not null,
          plan_score real not null,
          hypothesis text,
          task_id text,
          context_profile text,
          strategy_profile text,
          prior_id text,
          status text not null,
          branch text,
          worktree_path text,
          changed_files_json text not null default '[]',
          score integer,
          toffoli integer,
          qubits integer,
          score_delta integer,
          error text,
          plan_json text not null,
          summary_json text not null default '{}',
          started_at text not null,
          finished_at text
        );
        create index if not exists idx_trials_plan_hash on trials(plan_hash);
        create index if not exists idx_trials_status on trials(status, score_delta desc);
        """
    )
    _ensure_column(db, "trials", "hypothesis", "text")
    db.commit()


def _ensure_column(db: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    existing = {row[1] for row in db.execute(f"pragma table_info({table})").fetchall()}
    if column not in existing:
        db.execute(f"alter table {table} add column {column} {decl}")


def _record_trial_start(db: sqlite3.Connection, trial_id: str, plan: PlanRecord, cfg: dict[str, Any]) -> None:
    branch = f"{cfg.get('branch_prefix', 'agentic-patch-eval/')}{trial_id}"
    worktree_path = str(Path(cfg["worktree_root"]).expanduser() / trial_id)
    db.execute(
        """
        insert into trials
        (trial_id, plan_hash, source, source_id, run_id, candidate_id, plan_score,
         hypothesis, task_id, context_profile, strategy_profile, prior_id, status, branch,
         worktree_path, plan_json, started_at)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
        """,
        (
            trial_id,
            plan.plan_hash,
            plan.source,
            plan.source_id,
            plan.run_id,
            plan.candidate_id,
            plan.score,
            str(plan.plan.get("hypothesis", "")),
            plan.task_id,
            plan.context_profile,
            plan.strategy_profile,
            plan.prior_id,
            branch,
            worktree_path,
            json.dumps(plan.plan, ensure_ascii=False),
            _now(),
        ),
    )
    db.commit()


def _record_trial_finish(db: sqlite3.Connection, trial_id: str, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    db.execute(
        """
        update trials set status=?, changed_files_json=?, score=?, toffoli=?,
          qubits=?, score_delta=?, error=?, summary_json=?, finished_at=?
        where trial_id=?
        """,
        (
            summary.get("status", "unknown"),
            json.dumps(summary.get("changed_files", []), ensure_ascii=False),
            _maybe_int(summary.get("score")),
            _maybe_int(metrics.get("toffoli")),
            _maybe_int(metrics.get("qubits")),
            _maybe_int(summary.get("score_delta")),
            str(summary.get("error", ""))[:4000],
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
            _now(),
            trial_id,
        ),
    )
    db.commit()


def _next_trial_id(db: sqlite3.Connection, plan: PlanRecord) -> str:
    index = int(db.execute("select coalesce(max(id), 0) + 1 from trials").fetchone()[0])
    safe_candidate = re.sub(r"[^A-Za-z0-9_-]", "-", plan.candidate_id[:10] or "candidate")
    return f"pe{index:05d}-{safe_candidate}-{plan.plan_hash[:8]}"


def _create_worktree(project_root: Path, worktree_path: Path, branch: str, cfg: dict[str, Any]) -> None:
    if worktree_path.exists():
        if bool(cfg.get("reuse_existing_worktree", False)):
            return
        _remove_worktree(project_root, worktree_path)
    branch_check = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if branch_check.returncode == 0:
        subprocess.run(["git", "branch", "-D", branch], cwd=project_root, text=True, capture_output=True, check=False)
    cmd = ["git", "worktree", "add", "-b", branch, str(worktree_path), "HEAD"]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {result.stderr[-1000:]}")


def _remove_worktree(project_root: Path, worktree_path: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree_path)], cwd=project_root, check=False)
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


def _delete_branch(project_root: Path, branch: str) -> None:
    subprocess.run(["git", "branch", "-D", branch], cwd=project_root, text=True, capture_output=True, check=False)


def _changed_files(worktree_path: Path) -> list[str]:
    if not worktree_path.exists():
        return []
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=worktree_path,
        text=True,
        capture_output=True,
        check=False,
    )
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return sorted(set(files))


def _write_patch(worktree_path: Path, trial_dir: Path, changed_files: list[str]) -> None:
    untracked = [path for path in changed_files if (worktree_path / path).exists()]
    if untracked:
        subprocess.run(["git", "add", "-N", "--", *untracked], cwd=worktree_path, check=False)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=worktree_path,
        text=True,
        capture_output=True,
        check=False,
    )
    (trial_dir / "patch.diff").write_text(diff.stdout)


def _file_policy_error(changed_files: list[str], cfg: dict[str, Any]) -> str:
    for path in changed_files:
        error = _path_policy_error(path, cfg)
        if error:
            return error
    return ""


def _plan_policy_compatible(plan: PlanRecord, cfg: dict[str, Any]) -> bool:
    if _env_only_specs(plan, cfg):
        return True
    allowed_files = [str(path) for path in plan.plan.get("allowed_files", []) or []]
    if not allowed_files:
        return True
    return any(not _path_policy_error(path, cfg) for path in allowed_files)


def _path_policy_error(path: str, cfg: dict[str, Any]) -> str:
    evaluator = dict(cfg.get("evaluator", {}))
    allow_prefixes = tuple(str(item) for item in evaluator.get("allow_prefixes", ["src/point_add/"]))
    allow_suffixes = tuple(str(item) for item in evaluator.get("allow_suffixes", [".rs", ".md"]))
    deny_paths = tuple(str(item) for item in evaluator.get("deny_paths", []))
    if any(path == denied or path.startswith(denied.rstrip("/") + "/") for denied in deny_paths):
        return f"denied file changed: {path}"
    if not path.endswith(allow_suffixes):
        return f"unsupported changed file suffix: {path}"
    if not any(path.startswith(prefix) for prefix in allow_prefixes):
        return f"changed file outside allowed prefixes: {path}"
    return ""


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    prefix: str,
    out_dir: Path,
    env: dict[str, str] | None = None,
) -> CommandResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started)
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            124,
            str(exc.stdout or ""),
            str(exc.stderr or f"timeout after {timeout}s"),
            time.monotonic() - started,
            timed_out=True,
        )
    (out_dir / f"{prefix}.stdout.log").write_text(result.stdout)
    (out_dir / f"{prefix}.stderr.log").write_text(result.stderr)
    (out_dir / f"{prefix}.result.json").write_text(json.dumps(_command_dict(result), indent=2))
    return result


def _command_dict(result: CommandResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "wall_s": result.wall_s,
        "timed_out": result.timed_out,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _status_from_eval(eval_summary: dict[str, Any], cfg: dict[str, Any]) -> str:
    if not eval_summary.get("valid"):
        stage = str(eval_summary.get("stage", "eval"))
        return "build_failed" if stage in {"cargo", "build_circuit"} else "invalid"
    score = _maybe_int(eval_summary.get("score"))
    baseline = _maybe_int(dict(cfg.get("evaluator", {})).get("baseline_score"))
    if score is not None and baseline is not None and score < baseline:
        return "improved"
    return "valid"


def _summary(
    status: str,
    trial_id: str,
    plan: PlanRecord,
    branch: str,
    worktree_path: Path,
    changed_files: list[str],
    *,
    error: str = "",
    patcher: CommandResult | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = _maybe_int((eval_summary or {}).get("score"))
    metrics = (eval_summary or {}).get("metrics", {})
    baseline = _maybe_int((eval_summary or {}).get("baseline_score")) or 2_479_548_042
    score_delta = None if score is None else baseline - score
    return {
        "trial_id": trial_id,
        "status": status,
        "plan_hash": plan.plan_hash,
        "candidate_id": plan.candidate_id,
        "plan_score": plan.score,
        "hypothesis": plan.plan.get("hypothesis", ""),
        "branch": branch,
        "worktree_path": str(worktree_path),
        "changed_files": changed_files,
        "score": score,
        "score_delta": score_delta,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "error": error or str((eval_summary or {}).get("error", "")),
        "patcher": _command_dict(patcher) if patcher else {},
        "eval": eval_summary or {},
    }


def _parse_build_stdout(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    emitted = re.search(r"emitted ops\s*:\s*(\d+)", stdout)
    if emitted:
        out["emitted_ops"] = float(emitted.group(1))
    peak = re.search(r"DEBUG peak_qubits=(\d+)", stdout)
    if peak:
        out["qubits"] = float(peak.group(1))
    return out


def _parse_eval_stdout(stdout: str) -> dict[str, float]:
    patterns = {
        "loaded_ops": r"loaded ops\s*:\s*(\d+)",
        "tested_shots": r"tested shots\s*:\s*(\d+)",
        "classical_mismatches": r"classical mismatches\s*:\s*(\d+)",
        "phase_garbage_batches": r"phase-garbage batches\s*:\s*(\d+)",
        "ancilla_garbage_batches": r"ancilla-garbage batches\s*:\s*(\d+)",
        "avg_toffoli": r"avg executed Toffoli\s*:\s*([0-9.]+)",
        "avg_clifford": r"avg executed Clifford\s*:\s*([0-9.]+)",
        "emitted_ops": r"emitted ops\s*:\s*(\d+)",
        "qubits": r"qubits\s*:\s*(\d+)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            out[key] = float(match.group(1))
    return out


def _infer_run_id(trace_dir: Path) -> str:
    parts = trace_dir.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _should_stop(started: float, max_wall_s: float, stop_file: Path) -> bool:
    return stop_file.exists() or (time.monotonic() - started) >= max_wall_s


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


if __name__ == "__main__":
    main()
