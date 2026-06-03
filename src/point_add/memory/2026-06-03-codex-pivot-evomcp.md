# 2026-06-03 Codex pivot: evomcp search setup

Current public `main` at `5d2f6cf` validates locally:

- score: `2,480,196,210`
- avg executed Toffoli: `1,729,565`
- peak qubits: `1434`
- emitted ops: `11,126,484`
- validation: `0` classical mismatches, `0` phase-garbage batches, `0` ancilla-garbage batches

The active binders at `1434q` are:

- `dialog_gcd_materialized_special_chunked_raw_sum`
- `dialog_gcd_materialized_special_chunked_raw_difference`
- `dialog_gcd_compressed_block_quotient_reacquire_terminal_u`
- `dialog_gcd_compressed_block_ipmul_reacquire_terminal_u`
- `dialog_gcd_raw_pa_pair1_quotient`
- `dialog_gcd_raw_pa_pair2_product`

An evomcp EvoX integration was added around environment-driven knobs only. The
first search surface should focus on reroll islands and the already-exposed
route constants before source-level rewrites.
