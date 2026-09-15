# KDA parallel-gate experiment

This opt-in experiment is included in
[kernel PR3](https://github.com/zhaozx-cn/sgl-kernel-npu/pull/3) and paired with
[framework PR31](https://github.com/zhaozx-cn/sglang/pull/31). The branch includes
current `a5-k3-0828` at `1ec11e08ea3999a0f84f8b78fa57f19d59892ca5`, including
the chunk-KDA prefill operator required by PR31's framework base. The default
kernel dispatch is unchanged; PR31 defaults to standalone FP32 gate activation.

## What changes

`kda_target_verify_npu(..., precompute_raw_gates=True)` activates all verify
tokens as a `[next_power_of_2(steps), K]` FP32 tile before loading the recurrent
state. The state updates and every intermediate snapshot remain in their
original order. The padded-request early return precedes gate loads.

The activation result stays inside the same kernel; no global gate buffer or
additional kernel launch is introduced. Static `cann.extract_slice` operations
select each token's decay/beta inside the recurrent loop. The caller gets an
explicit error if the installed Triton-Ascend lacks this extension. Steps above
16 are intentionally unsupported by this experiment.

The optional `value_block_size` argument accepts 32, 64, or 128. At B=32,
H_v=3, K=V=128, these yield 384, 192, or 96 logical programs, respectively.
Their FP32 state tiles occupy 16, 32, or 64 KiB. Eight tokens of final FP32
decay occupy another 4 KiB per program, plus 32 bytes of beta. These figures
exclude compiler temporaries, alignment, and buffer duplication, so they are
not measured peak UB usage. Larger V tiles reduce redundant gate/q/k work but
can increase local-memory pressure or reduce available parallelism.

## Run on the serving NPU software stack

Use the matching kernel source checkout and a built library that also includes
the base branch's chunk-KDA prefill operator. The parallel-gate edits themselves
contain no C++ changes. Verify the printed `kernel_path` and revision in the
JSON output; an old library from before kernel PR11 is insufficient for PR31.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_kda_target_verify.py -q

# First isolate token-parallel activation at the original BV=64.
python benchmark/bench_kda_verify_parallel_gates.py \
  --batch 32 --steps 8 --heads 3 --dim 128 \
  --value-blocks 64 --output kda-gates-bv64.json

# Then compare the V tile for all three gate modes.
python benchmark/bench_kda_verify_parallel_gates.py \
  --batch 32 --steps 8 --heads 3 --dim 128 \
  --value-blocks 32 64 128 --output kda-gates-tiles.json

# A captured B=32 layout with one live request.
python benchmark/bench_kda_verify_parallel_gates.py \
  --batch 32 --padded 31 --steps 8 --heads 3 --dim 128 \
  --value-blocks 64 --output kda-gates-padding.json
```

The three modes use the same inputs and kernel revision:

- `separate`: standalone FP32 gate, beta cast/sigmoid, then preactivated verify.
- `raw`: the original per-token activation inside verify.
- `parallel`: token-vectorized activation inside verify.

All timing includes gate preparation where applicable. Compilation/autotuning
is warmed up before graph capture. Events measure repeated captured calls;
host launch overhead is amortized. JSON includes sample timings, output and
snapshot errors, software versions, and the loaded kernel source path.
This fixed-input, single-rank microbenchmark does not reproduce layer-to-layer
cache behavior or distributed TPOT. Repeat the matrix to check variability.

The historical traces showed approximately 59.559 us for separate gate plus
verify and 75.083 us for raw fused verify. Those are old observations, not new
benchmark results. The experiment must beat the same-run `separate` baseline,
not merely improve upon `raw`.

## Serving dispatch and preserved operators

PR31 selects this kernel with these settings before graph capture, on every
rank. Restart the server when changing them:

```bash
# Default/reference: standalone FP32 activation, existing BV selection.
export SGLANG_NPU_KDA_VERIFY_PARALLEL_GATES=0
export SGLANG_NPU_KDA_VERIFY_VALUE_BLOCK_SIZE=0

# Independent experiment: all-token gate activation at the original BV=64.
export SGLANG_NPU_KDA_VERIFY_PARALLEL_GATES=1
export SGLANG_NPU_KDA_VERIFY_VALUE_BLOCK_SIZE=64
```

The value-block override accepts 0 (kernel default), 32, 64, or 128 and applies
to either gate mode, allowing gate and tiling changes to be measured separately.
The removed `SGLANG_NPU_FUSED_KDA_VERIFY_GATES` flag does not select either
experimental path in PR31. The raw per-token kernel API remains available for
the benchmark and older callers, but PR31 does not dispatch it.

PR3 retains its DSpark local/global top1, ragged input/output and output-norm
kernels, packed-QKV strides, padding sentinels, and fixed-width convolution
coverage. These optional paths were not demonstrated to improve TPOT by the
supplied dense/static profiles. Their preservation is not a performance claim.

The fused top1 serving path additionally requires
`SGLANG_DSPARK_FOLDED_PROPOSAL=1`,
`SGLANG_DSPARK_FOLDED_SAMPLING=0`,
`SGLANG_DSPARK_FUSED_LOCAL_TOP1=1`, BF16 base logits and Markov W2, a configured
W2 TP shard, a replayable draft graph, and a greedy batch for the folded result
to be used. W2 BF16 and TP-shard flags default to true in PR31. A launch script
with `FOLDED_PROPOSAL=0` cannot reach the fused top1 sampler.

## Validation status

43 CPU semantic/API tests passed. The harness interprets the actual kernel body
with PyTorch tensor/pointer adapters. The tests cover strided raw gates and packed QKV,
grouped key/value heads, 3/8/16 steps, K/V tails, bounded/unbounded activation,
V tiles 32/64/128, padding NaNs, valid cache slot 0, and disabled snapshot writes.
The existing state/gate formulas and FP32 intermediate precision are retained.

The local checks did not compile Triton, execute CANN extensions, measure UB,
or run on an NPU. Device tests, graph output/snapshot parity, compiler memory
inspection, and performance measurement remain necessary. A serving experiment
must additionally compare acceptance length and full-round TPOT on identical
framework/kernel baselines. Do not promote this opt-in path based on CPU tests.

The implementation uses the static slicing pattern shown in the
[Ascend gather/scatter example](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py).
`tl.static_range` is a loop-unrolling hint, not a token-parallel launch grid;
see the [Triton API](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html).
