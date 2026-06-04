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
- `summary.json`: best full and best valid result;
- `programs.sqlite`: queryable run database.

For TODD/FastTODD T-count optimization, the same runner shape should wrap a
VarTODD-style evaluator: proxy = candidate pool reduction estimate, full =
verified final column count, reward = decreasing function of final T-count.
