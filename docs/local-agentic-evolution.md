# Local Agentic Evolution Setup

This repository runs local, subscription-friendly agentic evolution. `evomcp`
handles candidate bookkeeping and traces; Claude/Codex CLI calls produce
mutation plans; deterministic Rust scoring remains the final gate.

## Minimal Claude Subscription Harness

```bash
claude -p "$PROMPT" \
  --model haiku \
  --system-prompt "$COMPACT_SYSTEM_PROMPT" \
  --tools "" \
  --disable-slash-commands \
  --strict-mcp-config \
  --setting-sources user \
  --output-format json \
  --max-turns 3 \
  --max-budget-usd 0.03 \
  --no-session-persistence
```

Do not use `--bare` for subscription-backed calls in this environment: it skips
OAuth/keychain auth and reports `Not logged in`. Keep `--bare` for API-key
automation only.

## First-Stage Loop

Stage 0 is `plan_only`: the model returns a JSON hypothesis with allowed files,
commands, expected score effect, and failure controls. The evaluator scores this
contract before any expensive `cargo` run.

The later patch-eval stage should run in isolated worktrees and gate patches
with:

```bash
cargo build --release --locked --bin build_circuit --bin eval_circuit
TRACE_PEAK=1 ./target/release/build_circuit
./target/release/eval_circuit --note agentic
```

## Commands

```bash
PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_agentic_evomcp.py configs/evox-agentic-smoke.yaml

PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_agentic_evomcp.py configs/evox-agentic-claude-smoke.yaml
```

For Codex CLI with ChatGPT auth, omit `--model` unless a known account-compatible
model is required. This local install currently defaults to `gpt-5.5`; explicit
`gpt-5-mini` is rejected by Codex CLI under ChatGPT auth.
