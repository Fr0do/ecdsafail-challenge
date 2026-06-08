# GigaEvo-Lite Marathon

This runner keeps the proven local Codex/evomcp evaluator, but adopts the useful
GigaEvo operating pattern:

- long-running supervisor instead of one-shot batches;
- island lanes with different reasoning effort and population shape;
- central SQLite archive plus JSONL events;
- lineage context carried into later prompts;
- stop-file based control for unattended runs.

It intentionally does not vendor full GigaEvo yet. Full GigaEvo needs Redis,
Hydra, Python 3.12+, and API-key LLM routing. For this project, the immediate
goal is sustained Codex subscription use on local hardware with minimal new
infrastructure.

## Start

```bash
screen -dmS ecdsa-gigaevo-lite zsh -lc \
  'cd /Users/mkurkin/experiments/projects/ecdsafail-challenge && \
   PYTHONUNBUFFERED=1 PYTHONPATH=/private/tmp/evomcp:. \
   uv run --with pyyaml python scripts/run_agentic_marathon.py \
   configs/marathon-agentic-gpt55.yaml \
   >> artifacts/runs/agentic-gigaevo-lite-v1/marathon.log 2>&1'
```

## Monitor

```bash
screen -ls
tail -f artifacts/runs/agentic-gigaevo-lite-v1/marathon.log
tail -f artifacts/runs/agentic-gigaevo-lite-v1/events.jsonl
sqlite3 artifacts/runs/agentic-gigaevo-lite-v1/marathon.sqlite \
  "select score, task_id, strategy_profile, prior_id, hypothesis from plans order by score desc, mean_score desc limit 10;"
```

## Stop

```bash
touch artifacts/runs/agentic-gigaevo-lite-v1/STOP
```

The supervisor checks `STOP` between island runs. It does not kill an active
Codex call mid-request.

## Patch Evaluation Worker

The marathon is a hypothesis generator. Promote plans through a separate
ground-truth worker:

```bash
screen -dmS ecdsa-patch-eval zsh -lc \
  'cd /Users/mkurkin/experiments/projects/ecdsafail-challenge && \
   PYTHONUNBUFFERED=1 \
   uv run --with pyyaml python scripts/run_agentic_patch_eval.py \
   configs/agentic-patch-eval-gpt55.yaml \
   >> artifacts/runs/agentic-patch-eval-v1/worker.log 2>&1'
```

It polls the marathon SQLite archive plus live trace bundles, creates one
linked git worktree per candidate under
`/Users/mkurkin/experiments/projects/ecdsafail-patch-worktrees`, asks Codex to
apply a narrow patch, then runs:

```bash
cargo build --release --locked --bin build_circuit --bin eval_circuit
TRACE_PEAK=1 ./target/release/build_circuit
./target/release/eval_circuit --note patch-eval
```

Monitor:

```bash
tail -f artifacts/runs/agentic-patch-eval-v1/events.jsonl
sqlite3 artifacts/runs/agentic-patch-eval-v1/patch_eval.sqlite \
  "select status,count(*) from trials group by status;
   select status,score_delta,score,toffoli,qubits,trial_id,hypothesis
   from trials order by coalesce(score_delta,-999999999) desc, id desc limit 10;"
```

Stop:

```bash
touch artifacts/runs/agentic-patch-eval-v1/STOP
```
