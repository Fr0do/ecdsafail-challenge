# 2026-06-04 policy loop v4: PSO/EMA prior

Change:

- Added leaderboard-prime prior to `scripts/run_policy_loop.py`.
- The prior parses top public commit notes from
  `src/point_add/memory/research_graph/commit_notes`.
- It augments `DIALOG_REROLL` and `DIALOG_POST_SUB_REROLL` with explicit
  `set_default_env` values seen in top diffs plus deterministic prime values
  derived from commit/submission hashes.
- Added a PSO-like sampler that activates after full evaluations exist:
  top-10 full candidates define parents, valid full candidates get extra EMA
  weight, and numeric knobs move toward the best candidate plus EMA with
  occasional prime kicks.

Rationale:

v3 showed build-only proxy improvements near `b343_sm5_1434` were mostly
correctness traps. The v4 prior keeps the categorical policy but biases
exploration toward high-leaderboard reroll islands and uses only trusted full
results to move the PSO EMA.

Configs:

- `configs/policy-loop-b343-smoke.yaml` -> `policy-loop-b343-smoke-v3`
- `configs/policy-loop-b343-night.yaml` -> `policy-loop-b343-night-v4`

Verification:

```bash
python3 -m py_compile scripts/run_policy_loop.py
PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_policy_loop.py configs/policy-loop-b343-night.yaml --dry-run
```

The dry run produced `64` candidates and a `leaderboard_prior.json` containing
`10` top sources.

A forced local sampler check with `pso.fraction=1.0` produced an `8`-candidate
pool containing deterministic prime-kick rerolls such as `1307`, `1021`, and
post-sub `1823`.
