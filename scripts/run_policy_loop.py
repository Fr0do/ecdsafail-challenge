#!/usr/bin/env python3
"""Correctness-gated policy loop for ecdsa.fail/T-count-style rewrites.

This runner is intentionally between the old tabular island sampler and full
agentic patch evolution:

1. sample a large cheap pool from a categorical policy near known clean islands;
2. run build-only proxy scoring for the pool;
3. select a small diverse batch for full trusted eval_circuit;
4. update the policy only from correctness-gated full results;
5. persist JSONL plus ProgramDB/SQLite for parent selection in later runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by system-python users
    raise SystemExit(
        "PyYAML is required. Run via: "
        "PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml "
        "python scripts/run_policy_loop.py <config>"
    ) from exc


@dataclass
class ProxyRecord:
    candidate: Any
    genome: dict[str, Any]
    result: Any
    proxy_score: float
    novelty: float
    island_distance: int
    acquisition: float


@dataclass
class FullRecord:
    candidate: Any
    genome: dict[str, Any]
    result: Any
    reward: float
    novelty: float
    island_distance: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--evomcp-root", type=Path, default=Path("/private/tmp/evomcp"))
    parser.add_argument("--dry-run", action="store_true", help="write sampled pool without evaluation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.evomcp_root))
    sys.path.insert(0, str(project_root))

    import optim.search_spaces  # noqa: F401
    from evomcp.pipeline import Candidate
    from evomcp.pipeline.evaluator import cache_key, load_eval_cache, store_eval_cache
    from evomcp.pipeline.program_db import ProgramDB
    from optim.evaluators.ecdsafail import EcdsaFailEvaluator

    cfg = yaml.safe_load(args.config.read_text())
    output_dir = project_root / cfg["output_dir"]
    trace_dir = output_dir / "traces"
    cache_dir = project_root / cfg.get("cache", {}).get("dir", str(output_dir / "cache"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(cfg.get("seed", 0)))
    slots: dict[str, list[Any]] = {name: list(values) for name, values in cfg["slots"].items()}
    slot_names = list(slots)
    seed_genomes = [
        _complete_genome(dict(seed), slots, slot_names, rng)
        for seed in cfg.get("seed_genomes", [])
    ]
    seed_keys = {_genome_key(seed) for seed in seed_genomes}
    logits = _initial_logits(slots, seed_genomes, float(cfg.get("seed_prior_strength", 1.0)))

    if args.dry_run:
        pool = _candidate_pool(cfg, slots, slot_names, seed_genomes, logits, rng, seen=set(), round_index=0)
        (output_dir / "dry_pool.json").write_text(json.dumps(pool, indent=2, sort_keys=True))
        print(f"DRY_POOL {len(pool)} -> {output_dir / 'dry_pool.json'}")
        return

    program_db = ProgramDB.from_config(
        cfg,
        config_path=args.config,
        output_dir=output_dir,
        mode="ecdsa_policy_loop",
    )
    evaluator = EcdsaFailEvaluator(project_root)
    proxy_budget = _load_budget(cfg, "proxy", default_stage=1, default_timeout=240, extra_env={"ECDSAFAIL_EVAL_MODE": "build_only"})
    full_budget = _load_budget(cfg, "full", default_stage=2, default_timeout=360, extra_env={"ECDSAFAIL_EVAL_MODE": "full"})

    events_path = output_dir / "events.jsonl"
    seen_proxy: set[str] = set()
    seen_full: set[str] = set()
    proxy_archive: list[ProxyRecord] = []
    full_archive: list[FullRecord] = []
    started = time.monotonic()

    try:
        for round_index in range(int(cfg.get("rounds", 1))):
            pool = _candidate_pool(
                cfg,
                slots,
                slot_names,
                seed_genomes,
                logits,
                rng,
                seen=seen_proxy,
                round_index=round_index,
            )
            round_proxy: list[ProxyRecord] = []
            for genome in pool:
                key = _genome_key(genome)
                if key in seen_proxy:
                    continue
                seen_proxy.add(key)
                candidate = Candidate(
                    prog_genome=genome,
                    metadata={"mode": "policy_loop", "round": round_index, "stage": "proxy"},
                )
                if program_db:
                    program_db.record_program(candidate, generation=round_index, island=str(genome.get("patch_id", "main")))
                result = _evaluate_cached(
                    evaluator=evaluator,
                    candidate=candidate,
                    budget=proxy_budget,
                    seed=0,
                    run_dir=trace_dir,
                    cache_dir=cache_dir,
                    dataset_version=str(cfg.get("evaluator", {}).get("dataset_version", "ecdsa.fail-policy-loop")),
                    evaluator_version=str(cfg.get("evaluator", {}).get("version", evaluator.version)),
                    cache_key=cache_key,
                    load_eval_cache=load_eval_cache,
                    store_eval_cache=store_eval_cache,
                )
                if program_db:
                    program_db.record_evaluation(candidate, result, generation=round_index, seed=0, budget=proxy_budget)
                proxy_score = _proxy_score(result)
                novelty = _novelty(genome, [item.genome for item in proxy_archive], slot_names)
                island_distance = _nearest_distance(genome, seed_genomes, slot_names)
                acquisition = _acquisition(
                    genome=genome,
                    result=result,
                    proxy_score=proxy_score,
                    novelty=novelty,
                    island_distance=island_distance,
                    logits=logits,
                    slots=slots,
                    cfg=cfg,
                    is_seed=key in seed_keys,
                )
                record = ProxyRecord(candidate, genome, result, proxy_score, novelty, island_distance, acquisition)
                proxy_archive.append(record)
                round_proxy.append(record)
                event = {
                    "type": "proxy_eval",
                    "generation": round_index,
                    "candidate_id": candidate.candidate_id,
                    "genome": genome,
                    "proxy_score": proxy_score,
                    "primary_score": result.primary_score,
                    "secondary_scores": result.secondary_scores,
                    "novelty": novelty,
                    "island_distance": island_distance,
                    "acquisition": acquisition,
                    "trace_bundle_dir": str(result.trace_bundle_dir),
                }
                _append_jsonl(events_path, event)
                if program_db:
                    program_db.record_event(event)

            selected = _select_full(round_proxy, full_archive, seed_keys, seen_full, cfg, slot_names)
            round_full: list[FullRecord] = []
            for proxy_record in selected:
                key = _genome_key(proxy_record.genome)
                if key in seen_full:
                    continue
                seen_full.add(key)
                candidate = Candidate(
                    prog_genome=proxy_record.genome,
                    metadata={"mode": "policy_loop", "round": round_index, "stage": "full"},
                )
                if program_db:
                    program_db.record_program(candidate, generation=round_index, island=str(proxy_record.genome.get("patch_id", "main")))
                result = _evaluate_cached(
                    evaluator=evaluator,
                    candidate=candidate,
                    budget=full_budget,
                    seed=0,
                    run_dir=trace_dir,
                    cache_dir=cache_dir,
                    dataset_version=str(cfg.get("evaluator", {}).get("dataset_version", "ecdsa.fail-policy-loop")),
                    evaluator_version=str(cfg.get("evaluator", {}).get("version", evaluator.version)),
                    cache_key=cache_key,
                    load_eval_cache=load_eval_cache,
                    store_eval_cache=store_eval_cache,
                )
                if program_db:
                    program_db.record_evaluation(candidate, result, generation=round_index, seed=0, budget=full_budget)
                novelty = _novelty(proxy_record.genome, [item.genome for item in full_archive], slot_names)
                island_distance = _nearest_distance(proxy_record.genome, seed_genomes, slot_names)
                reward = _terminal_reward(result.secondary_scores, cfg, novelty)
                record = FullRecord(candidate, proxy_record.genome, result, reward, novelty, island_distance)
                full_archive.append(record)
                round_full.append(record)
                event = {
                    "type": "full_eval",
                    "generation": round_index,
                    "candidate_id": candidate.candidate_id,
                    "genome": proxy_record.genome,
                    "reward": reward,
                    "primary_score": result.primary_score,
                    "secondary_scores": result.secondary_scores,
                    "novelty": novelty,
                    "island_distance": island_distance,
                    "trace_bundle_dir": str(result.trace_bundle_dir),
                }
                _append_jsonl(events_path, event)
                if program_db:
                    program_db.record_event(event)

            _update_logits(logits, slots, round_full, lr=float(cfg.get("learning_rate", 0.6)))
            _write_policy(output_dir, slots, logits, round_index, proxy_archive, full_archive)
            if program_db:
                full_entries = [(item.candidate, item.result) for item in full_archive]
                valid_entries = [
                    (item.candidate, item.result)
                    for item in full_archive
                    if item.result.secondary_scores.get("valid", 0.0) >= 1.0
                ]
                program_db.record_archive(full_entries, generation=round_index, archive_name="policy_full")
                program_db.record_archive(valid_entries, generation=round_index, archive_name="policy_valid")
                program_db.record_event(
                    {
                        "type": "round_summary",
                        "generation": round_index,
                        "proxy_evaluations": len(round_proxy),
                        "full_evaluations": len(round_full),
                        "valid_full": sum(item.result.secondary_scores.get("valid", 0.0) >= 1.0 for item in round_full),
                    }
                )

        summary = _summary(full_archive, proxy_archive, started)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("SUMMARY", json.dumps(summary, sort_keys=True, default=str))
    finally:
        if program_db:
            program_db.record_run(status="complete")
            program_db.close()


def _load_budget(cfg: dict[str, Any], name: str, *, default_stage: int, default_timeout: int, extra_env: dict[str, Any]) -> Any:
    from evomcp.pipeline import Budget

    raw = {}
    budgets = cfg.get("budgets", {})
    if isinstance(budgets, dict):
        raw = dict(budgets.get(name, {}))
    elif isinstance(budgets, list):
        for item in budgets:
            if item.get("name") == name:
                raw = dict(item)
                break
    env = dict(cfg.get("fixed_env", {}))
    env.update(dict(raw.get("env_overrides", {})))
    env.update(extra_env)
    return Budget(
        stage=int(raw.get("stage", default_stage)),
        max_plans=int(raw.get("max_plans", 0)),
        max_judges=int(raw.get("max_judges", 0)),
        timeout_s=int(raw.get("timeout_s", default_timeout)),
        env_overrides=env,
    )


def _candidate_pool(
    cfg: dict[str, Any],
    slots: dict[str, list[Any]],
    slot_names: list[str],
    seed_genomes: list[dict[str, Any]],
    logits: dict[str, list[float]],
    rng: random.Random,
    *,
    seen: set[str],
    round_index: int,
) -> list[dict[str, Any]]:
    pool_size = int(cfg.get("pool_size", 16))
    pool: list[dict[str, Any]] = []
    if round_index == 0 or bool(cfg.get("replay_seeds_each_round", False)):
        for seed in seed_genomes:
            _append_unique_genome(pool, seed, seen)
            if len(pool) >= pool_size:
                return pool

    mutation_fraction = float(cfg.get("mutation_fraction", 0.7))
    attempts = 0
    while len(pool) < pool_size and attempts < max(1000, pool_size * 100):
        attempts += 1
        if seed_genomes and rng.random() < mutation_fraction:
            genome = _mutate_seed(seed_genomes, slots, slot_names, cfg, rng)
        else:
            genome = _sample_policy(slots, slot_names, logits, float(cfg.get("temperature", 1.2)), rng)
        _append_unique_genome(pool, genome, seen)
    return pool


def _mutate_seed(
    seed_genomes: list[dict[str, Any]],
    slots: dict[str, list[Any]],
    slot_names: list[str],
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    parent = dict(rng.choice(seed_genomes))
    mutable = list(cfg.get("mutable_slots", [name for name in slot_names if len(slots[name]) > 1]))
    hamming_choices = list(cfg.get("mutation_hamming_choices", [1, 1, 2]))
    n_changes = int(rng.choice(hamming_choices)) if hamming_choices else 1
    for name in rng.sample(mutable, min(n_changes, len(mutable))):
        values = slots[name]
        current = parent.get(name)
        if current in values and rng.random() < float(cfg.get("local_neighbor_probability", 0.75)):
            idx = values.index(current)
            radius = int(cfg.get("local_neighbor_radius", 2))
            lo = max(0, idx - radius)
            hi = min(len(values), idx + radius + 1)
            choices = [value for value in values[lo:hi] if value != current]
        else:
            choices = [value for value in values if value != current]
        if choices:
            parent[name] = rng.choice(choices)
    return parent


def _sample_policy(
    slots: dict[str, list[Any]],
    slot_names: list[str],
    logits: dict[str, list[float]],
    temperature: float,
    rng: random.Random,
) -> dict[str, Any]:
    return {
        name: _sample_choice(slots[name], logits[name], temperature, rng)
        for name in slot_names
    }


def _sample_choice(values: list[Any], logits: list[float], temperature: float, rng: random.Random) -> Any:
    scaled = [value / max(temperature, 1e-6) for value in logits]
    max_logit = max(scaled)
    weights = [math.exp(value - max_logit) for value in scaled]
    total = sum(weights)
    draw = rng.random() * total
    acc = 0.0
    for value, weight in zip(values, weights, strict=True):
        acc += weight
        if draw <= acc:
            return value
    return values[-1]


def _select_full(
    round_proxy: list[ProxyRecord],
    full_archive: list[FullRecord],
    seed_keys: set[str],
    seen_full: set[str],
    cfg: dict[str, Any],
    slot_names: list[str],
) -> list[ProxyRecord]:
    n_full = int(cfg.get("full_evals_per_round", 2))
    if n_full <= 0:
        return []
    max_distance = cfg.get("max_full_island_distance")
    eligible = [item for item in round_proxy if _genome_key(item.genome) not in seen_full]
    if max_distance is not None:
        eligible = [
            item for item in eligible
            if item.island_distance <= int(max_distance) or _genome_key(item.genome) in seed_keys
        ]
    forced = [
        item for item in eligible
        if bool(cfg.get("force_seed_full_eval", True)) and _genome_key(item.genome) in seed_keys
    ]
    selected: list[ProxyRecord] = []
    for item in sorted(forced, key=lambda rec: rec.proxy_score):
        _append_selected(selected, item, slot_names, min_distance=0)
        if len(selected) >= n_full:
            return selected

    remaining = [item for item in eligible if item not in selected]
    remaining.sort(key=lambda rec: rec.acquisition, reverse=True)
    min_selected_distance = int(cfg.get("min_selected_hamming_distance", 1))
    for item in remaining:
        if _append_selected(selected, item, slot_names, min_distance=min_selected_distance):
            if len(selected) >= n_full:
                return selected
    for item in remaining:
        if _append_selected(selected, item, slot_names, min_distance=0):
            if len(selected) >= n_full:
                return selected
    return selected


def _append_selected(selected: list[ProxyRecord], item: ProxyRecord, slot_names: list[str], *, min_distance: int) -> bool:
    key = _genome_key(item.genome)
    if any(_genome_key(existing.genome) == key for existing in selected):
        return False
    if min_distance > 0 and selected:
        nearest = min(_hamming(item.genome, existing.genome, slot_names) for existing in selected)
        if nearest < min_distance:
            return False
    selected.append(item)
    return True


def _evaluate_cached(
    *,
    evaluator: Any,
    candidate: Any,
    budget: Any,
    seed: int,
    run_dir: Path,
    cache_dir: Path,
    dataset_version: str,
    evaluator_version: str,
    cache_key: Any,
    load_eval_cache: Any,
    store_eval_cache: Any,
) -> Any:
    key = cache_key(candidate, budget, seed, dataset_version, evaluator_version)
    cached = load_eval_cache(cache_dir, key)
    if cached is not None:
        return cached
    result = evaluator.evaluate(candidate, budget, seed, run_dir=run_dir)
    store_eval_cache(cache_dir, key, result)
    return result


def _acquisition(
    *,
    genome: dict[str, Any],
    result: Any,
    proxy_score: float,
    novelty: float,
    island_distance: int,
    logits: dict[str, list[float]],
    slots: dict[str, list[Any]],
    cfg: dict[str, Any],
    is_seed: bool,
) -> float:
    baseline_proxy = float(cfg.get("baseline_proxy_score", proxy_score))
    proxy_scale = float(cfg.get("proxy_score_scale", max(abs(baseline_proxy), 1.0)))
    proxy_gain = (baseline_proxy - proxy_score) / max(proxy_scale, 1.0)
    prior = _policy_logprob(genome, slots, logits)
    seed_bonus = float(cfg.get("seed_full_bonus", 3.0)) if is_seed else 0.0
    success_bonus = 0.1 if result.success else -0.5
    return (
        seed_bonus
        + success_bonus
        + float(cfg.get("proxy_weight", 1.0)) * proxy_gain
        + float(cfg.get("acquisition_novelty_weight", 0.35)) * novelty
        + float(cfg.get("policy_prior_weight", 0.05)) * prior
        - float(cfg.get("island_distance_penalty", 0.4)) * island_distance
    )


def _terminal_reward(scores: dict[str, float], cfg: dict[str, Any], novelty: float) -> float:
    novelty_bonus = float(cfg.get("novelty_weight", 0.1)) * novelty
    if scores.get("valid", 0.0) >= 1.0:
        baseline = float(cfg.get("baseline_score", 0.0))
        scale = float(cfg.get("score_scale", 100000.0))
        score = float(scores.get("score", baseline))
        relative = (baseline - score) / max(scale, 1.0)
        score_term = max(float(cfg.get("worse_valid_score_floor", -0.9)), relative)
        return max(1e-6, float(cfg.get("valid_base_reward", 1.0)) + score_term + novelty_bonus)
    failures = (
        scores.get("classical_mismatches", 0.0)
        + scores.get("phase_garbage_batches", 0.0)
        + scores.get("ancilla_garbage_batches", 0.0)
    )
    return float(cfg.get("invalid_base_reward", 0.02)) / (1.0 + failures) + novelty_bonus * 0.05


def _update_logits(
    logits: dict[str, list[float]],
    slots: dict[str, list[Any]],
    scored: list[FullRecord],
    *,
    lr: float,
) -> None:
    if not scored:
        return
    log_rewards = [math.log(max(item.reward, 1e-9)) for item in scored]
    mean_log_reward = sum(log_rewards) / len(log_rewards)
    for item, log_reward in zip(scored, log_rewards, strict=True):
        advantage = log_reward - mean_log_reward
        for name, values in slots.items():
            idx = values.index(item.genome[name])
            logits[name][idx] += lr * advantage


def _write_policy(
    output_dir: Path,
    slots: dict[str, list[Any]],
    logits: dict[str, list[float]],
    round_index: int,
    proxy_archive: list[ProxyRecord],
    full_archive: list[FullRecord],
) -> None:
    policy = {}
    for name, values in slots.items():
        probs = _softmax(logits[name])
        policy[name] = [
            {"value": value, "logit": logit, "prob": prob}
            for value, logit, prob in zip(values, logits[name], probs, strict=True)
        ]
    payload = {
        "round": round_index,
        "policy": policy,
        "proxy_archive": [_proxy_item_dict(item) for item in proxy_archive],
        "full_archive": [_full_item_dict(item) for item in full_archive],
    }
    (output_dir / "policy.json").write_text(json.dumps(payload, indent=2, default=str))


def _summary(full_archive: list[FullRecord], proxy_archive: list[ProxyRecord], started: float) -> dict[str, Any]:
    valid = [item for item in full_archive if item.result.secondary_scores.get("valid", 0.0) >= 1.0]
    best_full = max(full_archive, key=lambda item: item.result.primary_score) if full_archive else None
    best_valid = max(valid, key=lambda item: item.result.primary_score) if valid else None
    return {
        "wall_s": time.monotonic() - started,
        "proxy_evaluations": len(proxy_archive),
        "full_evaluations": len(full_archive),
        "valid_full_evaluations": len(valid),
        "best_full": _full_item_dict(best_full) if best_full else None,
        "best_valid": _full_item_dict(best_valid) if best_valid else None,
    }


def _complete_genome(
    genome: dict[str, Any],
    slots: dict[str, list[Any]],
    slot_names: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    return {name: genome.get(name, rng.choice(slots[name])) for name in slot_names}


def _append_unique_genome(pool: list[dict[str, Any]], genome: dict[str, Any], seen: set[str]) -> bool:
    key = _genome_key(genome)
    if key in seen or any(_genome_key(item) == key for item in pool):
        return False
    pool.append(dict(genome))
    return True


def _initial_logits(
    slots: dict[str, list[Any]],
    seed_genomes: list[dict[str, Any]],
    seed_prior_strength: float,
) -> dict[str, list[float]]:
    logits = {name: [0.0 for _ in values] for name, values in slots.items()}
    for seed in seed_genomes:
        for name, values in slots.items():
            if seed.get(name) in values:
                logits[name][values.index(seed[name])] += seed_prior_strength
    return logits


def _policy_logprob(genome: dict[str, Any], slots: dict[str, list[Any]], logits: dict[str, list[float]]) -> float:
    logp = 0.0
    for name, values in slots.items():
        probs = _softmax(logits[name])
        logp += math.log(max(probs[values.index(genome[name])], 1e-12))
    return logp


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    weights = [math.exp(value - max_value) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def _proxy_score(result: Any) -> float:
    scores = result.secondary_scores
    if "proxy_score" in scores:
        return float(scores["proxy_score"])
    emitted = scores.get("emitted_ops")
    qubits = scores.get("qubits")
    if emitted is not None and qubits is not None:
        return float(emitted) * float(qubits)
    return -float(result.primary_score)


def _novelty(genome: dict[str, Any], archive: list[dict[str, Any]], slot_names: list[str]) -> float:
    if not archive:
        return 1.0
    return min(_hamming(genome, item, slot_names) / max(len(slot_names), 1) for item in archive)


def _nearest_distance(genome: dict[str, Any], seeds: list[dict[str, Any]], slot_names: list[str]) -> int:
    if not seeds:
        return 0
    return min(_hamming(genome, seed, slot_names) for seed in seeds)


def _hamming(left: dict[str, Any], right: dict[str, Any], slot_names: list[str]) -> int:
    return sum(left.get(name) != right.get(name) for name in slot_names)


def _proxy_item_dict(item: ProxyRecord) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate.candidate_id,
        "genome": item.genome,
        "proxy_score": item.proxy_score,
        "novelty": item.novelty,
        "island_distance": item.island_distance,
        "acquisition": item.acquisition,
        "primary_score": item.result.primary_score,
        "secondary": item.result.secondary_scores,
    }


def _full_item_dict(item: FullRecord | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "candidate_id": item.candidate.candidate_id,
        "genome": item.genome,
        "reward": item.reward,
        "novelty": item.novelty,
        "island_distance": item.island_distance,
        "primary_score": item.result.primary_score,
        "secondary": item.result.secondary_scores,
    }


def _genome_key(genome: dict[str, Any]) -> str:
    return json.dumps(genome, sort_keys=True, separators=(",", ":"), default=str)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
