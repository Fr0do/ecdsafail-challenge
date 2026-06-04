# ECDSA/T-count Policy Loop

The policy loop is the direct bridge from the Accordion/GFlowNet proposal to
the active ecdsa.fail search:

1. keep the correctness-preserving rewrite/build engine fixed;
2. sample a categorical policy over exposed route knobs and known clean islands;
3. run cheap build-only proxy scoring for a large candidate pool;
4. spend trusted `eval_circuit` only on a small diverse shortlist;
5. update the policy from correctness-gated full rewards;
6. persist JSONL plus `programs.sqlite` for archive queries and future parent
   selection.

This is not full trajectory-balance GFlowNet yet. It is a correctness-gated
terminal-policy loop with GFlowNet-style diversity and reward-proportional
updates. That is the efficient first layer for ECDSA because nearby knobs are
mostly correctness islands: proxy score alone selects many invalid candidates.

The v4 configs also add a light PSO-style layer. At startup, the runner parses
the top public leaderboard commit notes from
`src/point_add/memory/research_graph/commit_notes`, adds explicit reroll values
seen in those diffs, and derives deterministic random prime reroll candidates
from commit/submission hashes. After trusted full evaluations exist, a fraction
of the pool is sampled from top-10 full candidates by moving numeric knobs
toward an EMA of the best full archive, with occasional prime kicks. The EMA is
updated only from full evaluations, so build-only proxy traps do not directly
pull the swarm.

## Commands

Smoke:

```bash
PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_policy_loop.py \
  configs/policy-loop-b343-smoke.yaml
```

Overnight:

```bash
PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_policy_loop.py \
  configs/policy-loop-b343-night.yaml
```

Outputs:

- `events.jsonl`: proxy/full evaluation stream;
- `policy.json`: current categorical probabilities and archives;
- `leaderboard_prior.json`: dry-run dump of top-N leaderboard sources and
  derived prime reroll values;
- `summary.json`: best full and best valid result;
- `programs.sqlite`: queryable run database.

The night config includes an efficiency guard: it stops after enough full
evaluations fail to improve the best valid score. Proxy-only improvements do not
count as progress.

The current night config writes to `artifacts/runs/policy-loop-b343-night-v4`
and uses a separate `artifacts/cache-policy-loop-b343-night-v4` cache, leaving
the v3 baseline untouched.

For TODD/FastTODD T-count optimization, the same runner shape should wrap a
VarTODD-style evaluator: proxy = candidate pool reduction estimate, full =
verified final column count, reward = decreasing function of final T-count.
