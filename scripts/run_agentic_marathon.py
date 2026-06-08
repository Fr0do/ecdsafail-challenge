#!/usr/bin/env python3
"""Long-running GigaEvo-inspired supervisor for local Codex evolution.

This script intentionally stays lighter than full GigaEvo: it keeps our proven
evomcp evaluator, but adds marathon scheduling, island lanes, lineage context,
restartable artifacts, and a central SQLite archive.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required; run with: uv run --with pyyaml ...") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--evomcp-root", type=Path, default=Path("/private/tmp/evomcp"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    output_dir = project_root / cfg["output_dir"]
    generated_config_dir = output_dir / "generated-configs"
    run_root = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(output_dir / "marathon.sqlite")
    db.row_factory = sqlite3.Row
    _init_db(db)
    started = time.monotonic()
    max_wall_s = float(cfg.get("max_wall_hours", 24.0)) * 3600.0
    max_cycles = int(cfg.get("max_cycles", 1))
    stop_file = output_dir / str(cfg.get("stop_file", "STOP"))
    context_file = output_dir / "lineage_context.md"
    events_path = output_dir / "events.jsonl"
    base_evox = dict(cfg["base_evox"])
    islands = list(cfg.get("islands", []))
    if not islands:
        raise SystemExit("marathon config requires at least one island")

    _append_event(events_path, {"type": "marathon_start", "config": str(args.config), "time": _now()})
    for cycle in range(max_cycles):
        if _should_stop(started, max_wall_s, stop_file):
            break
        for island_index, island in enumerate(islands):
            if _should_stop(started, max_wall_s, stop_file):
                break
            run_id = _run_id(str(cfg.get("name", "agentic-marathon")), cycle, island_index, island)
            run_config = _build_run_config(
                cfg,
                base_evox,
                island,
                project_root=project_root,
                run_root=run_root,
                run_id=run_id,
                context_file=context_file,
            )
            run_config_path = generated_config_dir / f"{run_id}.yaml"
            run_config_path.write_text(yaml.safe_dump(run_config, sort_keys=False))
            _record_run_start(db, run_id, cycle, island_index, island, run_config_path)
            _append_event(
                events_path,
                {
                    "type": "run_start",
                    "run_id": run_id,
                    "cycle": cycle,
                    "island": island.get("name", f"island{island_index}"),
                    "config": str(run_config_path),
                    "time": _now(),
                },
            )
            result = _run_evox(project_root, args.evomcp_root, run_config_path, island, run_id)
            summary = _ingest_run(db, project_root, run_config, run_id, cycle, island_index, island)
            _record_run_finish(db, run_id, result.returncode, summary)
            _write_lineage_context(db, context_file, cfg)
            _append_event(
                events_path,
                {
                    "type": "run_finish",
                    "run_id": run_id,
                    "returncode": result.returncode,
                    "summary": summary,
                    "time": _now(),
                },
            )
            if result.returncode != 0 and not bool(cfg.get("continue_on_failure", True)):
                _append_event(events_path, {"type": "marathon_stop", "reason": "run_failed", "time": _now()})
                db.close()
                raise SystemExit(result.returncode)
            time.sleep(float(cfg.get("sleep_between_runs_s", 5.0)))

    _write_lineage_context(db, context_file, cfg)
    _append_event(events_path, {"type": "marathon_complete", "time": _now()})
    db.close()


def _build_run_config(
    cfg: dict[str, Any],
    base_evox: dict[str, Any],
    island: dict[str, Any],
    *,
    project_root: Path,
    run_root: Path,
    run_id: str,
    context_file: Path,
) -> dict[str, Any]:
    run_cfg = copy.deepcopy(base_evox)
    run_cfg["output_dir"] = _project_relative(run_root / run_id, project_root)
    run_cfg.setdefault("cache", {})["dir"] = str(
        _project_relative(run_root / f"{run_id}-cache", project_root)
    )
    if "target_prog_slots" in island:
        run_cfg["target_prog_slots"] = list(island["target_prog_slots"])
    if "evox" in island:
        run_cfg.setdefault("evox", {}).update(dict(island["evox"]))
    for budget in run_cfg.get("budgets", []):
        if budget.get("name") != island.get("budget_name", "proxy"):
            continue
        if "timeout_s" in island:
            budget["timeout_s"] = int(island["timeout_s"])
        env_overrides = dict(budget.get("env_overrides", {}))
        env_overrides.update(dict(island.get("env_overrides", {})))
        env_overrides["AGENT_EXTRA_CONTEXT_FILE"] = str(context_file)
        budget["env_overrides"] = env_overrides
    return run_cfg


def _project_relative(path: Path, project_root: Path) -> str:
    return os.path.relpath(path.resolve(), project_root.resolve())


def _run_evox(
    project_root: Path,
    evomcp_root: Path,
    config_path: Path,
    island: dict[str, Any],
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    log_path = project_root / island.get("log_path", "") if island.get("log_path") else None
    run_output_dir = project_root / yaml.safe_load(config_path.read_text())["output_dir"]
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_output_dir / "supervisor.log"
    cmd = [
        sys.executable,
        str(project_root / "scripts/run_agentic_evomcp.py"),
        str(config_path),
        "--evomcp-root",
        str(evomcp_root),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{evomcp_root}:{project_root}:{env.get('PYTHONPATH', '')}"
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"--- run {run_id} start {_now()} ---\n")
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=int(island.get("run_timeout_s", island.get("timeout_s", 900))) + 120,
            check=False,
        )
        log.write(f"--- run {run_id} exit {_now()} code={proc.returncode} wall={time.monotonic() - started:.1f}s ---\n")
    return proc


def _init_db(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        create table if not exists runs (
          run_id text primary key,
          cycle integer not null,
          island_index integer not null,
          island_name text not null,
          status text not null,
          returncode integer,
          config_path text not null,
          started_at text not null,
          finished_at text,
          summary_json text not null default '{}'
        );
        create table if not exists plans (
          id integer primary key autoincrement,
          run_id text not null,
          candidate_id text not null,
          trace_dir text not null,
          score real not null,
          mean_score real,
          best_proposal_index real,
          agent_wall_s real,
          task_id text,
          context_profile text,
          strategy_profile text,
          prior_id text,
          hypothesis text,
          expected_effect text,
          allowed_files_json text not null,
          plan_json text not null,
          result_json text not null,
          created_at text not null,
          unique(run_id, candidate_id)
        );
        create index if not exists idx_plans_score on plans(score desc, mean_score desc);
        create index if not exists idx_plans_profile on plans(task_id, context_profile, strategy_profile, prior_id);
        """
    )
    db.commit()


def _record_run_start(
    db: sqlite3.Connection,
    run_id: str,
    cycle: int,
    island_index: int,
    island: dict[str, Any],
    config_path: Path,
) -> None:
    db.execute(
        """
        insert or replace into runs
        (run_id, cycle, island_index, island_name, status, returncode, config_path, started_at, finished_at, summary_json)
        values (?, ?, ?, ?, 'running', null, ?, ?, null, '{}')
        """,
        (run_id, cycle, island_index, str(island.get("name", f"island{island_index}")), str(config_path), _now()),
    )
    db.commit()


def _record_run_finish(
    db: sqlite3.Connection,
    run_id: str,
    returncode: int,
    summary: dict[str, Any],
) -> None:
    db.execute(
        "update runs set status=?, returncode=?, finished_at=?, summary_json=? where run_id=?",
        ("complete" if returncode == 0 else "failed", returncode, _now(), json.dumps(summary, sort_keys=True), run_id),
    )
    db.commit()


def _ingest_run(
    db: sqlite3.Connection,
    project_root: Path,
    run_cfg: dict[str, Any],
    run_id: str,
    cycle: int,
    island_index: int,
    island: dict[str, Any],
) -> dict[str, Any]:
    del cycle, island_index, island
    run_dir = project_root / run_cfg["output_dir"]
    traces_dir = run_dir / "traces"
    n_results = 0
    best_score: float | None = None
    best_candidate = None
    if traces_dir.exists():
        for result_path in sorted(traces_dir.glob("*/result.json")):
            trace_dir = result_path.parent
            plan_path = trace_dir / "plan.json"
            profile_path = trace_dir / "prompt_profile.json"
            if not plan_path.exists():
                continue
            result = json.loads(result_path.read_text())
            plan = json.loads(plan_path.read_text())
            profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
            scores = result.get("secondary_scores", {})
            score = float(result.get("primary_score", 0.0))
            n_results += 1
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = result.get("candidate_id")
            db.execute(
                """
                insert or replace into plans
                (run_id, candidate_id, trace_dir, score, mean_score, best_proposal_index, agent_wall_s,
                 task_id, context_profile, strategy_profile, prior_id, hypothesis, expected_effect,
                 allowed_files_json, plan_json, result_json, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(result.get("candidate_id", trace_dir.name.replace("-seed0", ""))),
                    str(trace_dir),
                    score,
                    _maybe_float(scores.get("mean_proposal_score")),
                    _maybe_float(scores.get("best_proposal_index")),
                    _maybe_float(scores.get("agent_wall_s")),
                    str(profile.get("task_id", "")),
                    str(profile.get("context_profile", "")),
                    str(profile.get("strategy_profile", "")),
                    str(profile.get("prior_id", "")),
                    str(plan.get("hypothesis", "")),
                    str(plan.get("expected_score_effect", "")),
                    json.dumps(plan.get("allowed_files", []), ensure_ascii=False),
                    json.dumps(plan, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    _now(),
                ),
            )
    db.commit()
    return {"n_results": n_results, "best_score": best_score, "best_candidate": best_candidate}


def _write_lineage_context(db: sqlite3.Connection, path: Path, cfg: dict[str, Any]) -> None:
    top_n = int(cfg.get("lineage_top_n", 12))
    rows = db.execute(
        """
        select score, mean_score, task_id, context_profile, strategy_profile, prior_id,
               hypothesis, expected_effect, allowed_files_json, run_id, candidate_id
        from plans
        order by score desc, coalesce(mean_score, 0) desc, id desc
        limit ?
        """,
        (top_n,),
    ).fetchall()
    lines = [
        "# Marathon Lineage Context",
        "",
        "Use this as prior evidence, not as truth. Every promoted idea still needs build_circuit/eval_circuit.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        files = ", ".join(json.loads(row["allowed_files_json"] or "[]"))
        lines.extend(
            [
                f"## {index}. score={row['score']} mean={row['mean_score']} candidate={row['candidate_id']}",
                f"- run: `{row['run_id']}`",
                f"- profile: task={row['task_id']} context={row['context_profile']} strategy={row['strategy_profile']} prior={row['prior_id']}",
                f"- hypothesis: {row['hypothesis']}",
                f"- expected effect: {row['expected_effect']}",
                f"- files: {files}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _run_id(prefix: str, cycle: int, island_index: int, island: dict[str, Any]) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(island.get("name", f"island{island_index}")))
    return f"{prefix}-c{cycle:03d}-i{island_index:02d}-{safe_name}"


def _should_stop(started: float, max_wall_s: float, stop_file: Path) -> bool:
    return stop_file.exists() or (time.monotonic() - started) >= max_wall_s


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
