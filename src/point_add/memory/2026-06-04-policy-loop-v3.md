# 2026-06-04 correctness-gated policy loop v3

Run:

```bash
PYTHONPATH=/private/tmp/evomcp:. uv run --with pyyaml \
  python scripts/run_policy_loop.py configs/policy-loop-b343-night.yaml
```

Artifacts:

- `artifacts/runs/policy-loop-b343-night-v3/events.jsonl`
- `artifacts/runs/policy-loop-b343-night-v3/policy.json`
- `artifacts/runs/policy-loop-b343-night-v3/summary.json`
- `artifacts/runs/policy-loop-b343-night-v3/programs.sqlite`

Result:

- `256` build-only proxy evaluations.
- `24` trusted full evaluations.
- `2` valid full evaluations: the two seeds only.
- Best valid remains `b343_sm5_1434`:
  - score `2,479,548,042`
  - Toffoli `1,729,113`
  - qubits `1434`
  - emitted ops `11,114,242`
- `current_1434` remains valid but worse:
  - score `2,480,196,210`
  - Toffoli `1,729,565`
  - qubits `1434`
- Early stop: `3 rounds without valid-score improvement`.

Policy after early stop still correctly favors the b343 island:

- `patch_id=b343_sm5_1434`: probability about `0.69`
- schedule margin `5`: probability about `0.68`
- `DIALOG_REROLL=1844`: probability about `0.55`
- `DIALOG_POST_SUB_REROLL=3532`: probability about `0.58`

Takeaway: build-only proxy improvements in this local reroll/post-sub
neighborhood are mostly correctness traps. Single-slot post-sub/reroll changes
near b343 often reduce emitted ops by a few thousand, but full eval rejects them
with classical mismatches or phase garbage. Future search should spend less full
budget on pure reroll/post-sub neighbors and more on source-level route changes
that create a new clean island, then retune rerolls around that new op stream.
