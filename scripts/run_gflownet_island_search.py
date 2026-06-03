#!/usr/bin/env python3
"""Tabular GFlowNet-style island search for ecdsa.fail knobs."""

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

import yaml


@dataclass
class ScoredGenome:
    genome: dict[str, Any]
    candidate_id: str
    reward: float
    novelty: float
    primary_score: float
    secondary: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--evomcp-root", type=Path, default=Path("/private/tmp/evomcp"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.evomcp_root))
    sys.path.insert(0, str(project_root))

    import optim.search_spaces  # noqa: F401
    from evomcp.pipeline import Budget, Candidate
    from optim.evaluators.ecdsafail import EcdsaFailEvaluator

    cfg = yaml.safe_load(args.config.read_text())
    output_dir = project_root / cfg["output_dir"]
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(cfg.get("seed", 0)))
    slots: dict[str, list[Any]] = {name: list(values) for name, values in cfg["slots"].items()}
    slot_names = list(slots)
    logits = {name: [0.0 for _ in values] for name, values in slots.items()}
    seen: set[str] = set()
    archive: list[ScoredGenome] = []
    evaluator = EcdsaFailEvaluator(project_root)
    budget = Budget(
        stage=int(cfg.get("stage", 2)),
        max_plans=0,
        max_judges=0,
        timeout_s=int(cfg.get("timeout_s", 360)),
        env_overrides=dict(cfg.get("fixed_env", {})),
    )

    events_path = output_dir / "events.jsonl"
    started = time.monotonic()
    for round_index in range(int(cfg.get("rounds", 1))):
        batch = _next_batch(
            cfg=cfg,
            slots=slots,
            slot_names=slot_names,
            logits=logits,
            rng=rng,
            seen=seen,
            round_index=round_index,
        )
        scored = []
        for genome in batch:
            candidate = Candidate(
                prog_genome=genome,
                metadata={"sampler": "tabular_gflownet", "round": round_index},
            )
            result = evaluator.evaluate(candidate, budget, 0, run_dir=trace_dir)
            novelty = _novelty(genome, archive, slot_names)
            reward = _terminal_reward(result.secondary_scores, cfg, novelty)
            item = ScoredGenome(
                genome=genome,
                candidate_id=candidate.candidate_id,
                reward=reward,
                novelty=novelty,
                primary_score=result.primary_score,
                secondary=dict(result.secondary_scores),
            )
            archive.append(item)
            scored.append(item)
            seen.add(_genome_key(genome))
            _append_jsonl(
                events_path,
                {
                    "round": round_index,
                    "candidate_id": candidate.candidate_id,
                    "genome": genome,
                    "reward": reward,
                    "novelty": novelty,
                    "primary_score": result.primary_score,
                    "secondary_scores": result.secondary_scores,
                    "trace_bundle_dir": str(result.trace_bundle_dir),
                },
            )
        _update_logits(logits, slots, scored, lr=float(cfg.get("learning_rate", 0.7)))
        _write_policy(output_dir, slots, logits, round_index, archive)

    best = max(archive, key=lambda item: item.primary_score)
    valid = [item for item in archive if item.secondary.get("valid", 0.0) >= 1.0]
    summary = {
        "wall_s": time.monotonic() - started,
        "evaluations": len(archive),
        "valid_evaluations": len(valid),
        "best": _item_dict(best),
        "best_valid": _item_dict(max(valid, key=lambda item: item.primary_score)) if valid else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("BEST", best.candidate_id, best.primary_score, best.secondary, best.genome)


def _next_batch(
    *,
    cfg: dict[str, Any],
    slots: dict[str, list[Any]],
    slot_names: list[str],
    logits: dict[str, list[float]],
    rng: random.Random,
    seen: set[str],
    round_index: int,
) -> list[dict[str, Any]]:
    batch_size = int(cfg.get("batch_size", 4))
    batch: list[dict[str, Any]] = []
    if round_index == 0:
        for seed in cfg.get("seed_genomes", []):
            genome = _complete_genome(dict(seed), slots, slot_names, rng)
            if _genome_key(genome) not in seen:
                batch.append(genome)
            if len(batch) >= batch_size:
                return batch
    attempts = 0
    while len(batch) < batch_size and attempts < 1000:
        attempts += 1
        genome = {
            name: _sample_choice(slots[name], logits[name], float(cfg.get("temperature", 1.2)), rng)
            for name in slot_names
        }
        key = _genome_key(genome)
        if key in seen or any(_genome_key(item) == key for item in batch):
            continue
        batch.append(genome)
    return batch


def _complete_genome(
    genome: dict[str, Any],
    slots: dict[str, list[Any]],
    slot_names: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    return {name: genome.get(name, rng.choice(slots[name])) for name in slot_names}


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


def _terminal_reward(scores: dict[str, float], cfg: dict[str, Any], novelty: float) -> float:
    novelty_bonus = float(cfg.get("novelty_weight", 0.15)) * novelty
    if scores.get("valid", 0.0) >= 1.0:
        baseline = float(cfg.get("baseline_score", 0.0))
        scale = float(cfg.get("score_scale", 100000.0))
        score = float(scores.get("score", baseline))
        improvement = max(0.0, baseline - score) / max(scale, 1.0)
        return float(cfg.get("valid_base_reward", 1.0)) + improvement + novelty_bonus
    failures = (
        scores.get("classical_mismatches", 0.0)
        + scores.get("phase_garbage_batches", 0.0)
        + scores.get("ancilla_garbage_batches", 0.0)
    )
    return float(cfg.get("invalid_base_reward", 0.02)) / (1.0 + failures) + novelty_bonus * 0.1


def _novelty(genome: dict[str, Any], archive: list[ScoredGenome], slot_names: list[str]) -> float:
    if not archive:
        return 1.0
    distances = []
    for item in archive:
        diff = sum(genome[name] != item.genome.get(name) for name in slot_names)
        distances.append(diff / max(len(slot_names), 1))
    return min(distances)


def _update_logits(
    logits: dict[str, list[float]],
    slots: dict[str, list[Any]],
    scored: list[ScoredGenome],
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
    archive: list[ScoredGenome],
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
        "archive": [_item_dict(item) for item in archive],
    }
    (output_dir / "policy.json").write_text(json.dumps(payload, indent=2, default=str))


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    weights = [math.exp(value - max_value) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def _item_dict(item: ScoredGenome) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id,
        "genome": item.genome,
        "reward": item.reward,
        "novelty": item.novelty,
        "primary_score": item.primary_score,
        "secondary": item.secondary,
    }


def _genome_key(genome: dict[str, Any]) -> str:
    return json.dumps(genome, sort_keys=True, separators=(",", ":"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
