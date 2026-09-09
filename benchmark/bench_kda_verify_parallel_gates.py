"""Same-revision NPU graph benchmark of gate preparation plus KDA verify.

Run from a kernel checkout containing PR3 and the parallel-gate experiment.
No serving flags or defaults are changed. Reported times are microbenchmark
times, not distributed TPOT. Compile and warm up every mode before timing.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch_npu
import triton
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


def measure_graph(function, *, capture_calls, replays, samples):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(capture_calls):
            graph_output = function()
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    times_us = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000 / (replays * capture_calls))
    # Keep the captured output live until after all replays.
    assert graph_output is not None
    return {
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "samples_us": times_us,
    }, graph_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--padded", type=int, default=0)
    parser.add_argument("--value-blocks", type=int, nargs="+", default=[64])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--capture-calls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, default=Path("kda-gate-results.json"))
    args = parser.parse_args()
    if not 0 <= args.padded < args.batch:
        parser.error("padded must be in [0, batch)")
    if not 1 <= args.steps <= 16:
        parser.error("steps must be in [1, 16] for the parallel-gate experiment")
    if any(
        x <= 0
        for x in (
            args.batch,
            args.heads,
            args.dim,
            args.samples,
            args.replays,
            args.capture_calls,
        )
    ):
        parser.error("shape and timing counts must be positive")
    if any(x not in (32, 64, 128) for x in args.value_blocks):
        parser.error("value-blocks must contain only 32, 64, or 128")

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    batch, steps, heads, dim = args.batch, args.steps, args.heads, args.dim
    tokens = batch * steps
    packed = torch.randn(
        1, tokens, 3 * heads * dim, dtype=torch.bfloat16, device=device
    )
    q, k, v = [x.view(1, tokens, heads, dim) for x in packed.chunk(3, dim=-1)]
    raw_a = torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    raw_b = torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(1, 1, heads, 1, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(heads * dim, dtype=torch.float32, device=device) * 0.1
    initial = (
        torch.randn(batch, heads, dim, dim, dtype=torch.bfloat16, device=device) * 0.1
    )
    initial_before = initial.clone()
    initial_indices = torch.arange(batch, dtype=torch.int64, device=device)
    if args.padded:
        initial_indices[-args.padded :] = -1
    snapshot_indices = torch.arange(batch, dtype=torch.int64, device=device)
    scratch = torch.full(
        (batch, steps, heads, dim, dim), 3.0, dtype=torch.bfloat16, device=device
    )
    common = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state_source=initial,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=scratch,
        intermediate_state_indices=snapshot_indices,
        cache_steps=steps,
    )

    def make_call(mode, bv):
        def call():
            if mode == "separate":
                gate_a = fused_kda_gate_npu(
                    raw_a.flatten(-2), A_log, dim, gate_bias=dt_bias, lower_bound=-5.0
                )
                gate_b = raw_b.float().sigmoid()
                return kda_target_verify_npu(
                    **common,
                    a=gate_a,
                    b=gate_b,
                    gates_are_preactivated=True,
                    value_block_size=bv,
                )
            return kda_target_verify_npu(
                **common,
                a=raw_a,
                b=raw_b,
                gates_are_preactivated=False,
                lower_bound=-5.0,
                precompute_raw_gates=(mode == "parallel"),
                value_block_size=bv,
            )

        return call

    expected = make_call("separate", 64)().clone()
    expected_scratch = scratch.clone()
    kernel_path = Path(inspect.getfile(kda_target_verify_npu)).resolve()
    revision = subprocess.run(
        ["git", "-C", str(kernel_path.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    report = {
        "scope": "single-rank NPU graph, gate preparation plus recurrence, fixed inputs",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "device": torch.npu.get_device_name(args.device),
        "kernel_path": str(kernel_path),
        "kernel_revision": revision,
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "compiler_environment": {
            name: os.environ.get(name)
            for name in ("TRITON_ALL_BLOCKS_PARALLEL", "ASCEND_LAUNCH_BLOCKING")
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [],
    }
    failed = False
    for bv in args.value_blocks:
        for mode in ("separate", "raw", "parallel"):
            case = {"mode": mode, "value_block_size": bv}
            try:
                call = make_call(mode, bv)
                scratch.fill_(3.0)
                actual = call()
                torch.npu.synchronize()
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(
                    scratch, expected_scratch, rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)
                if args.padded:
                    torch.testing.assert_close(
                        actual[:, -args.padded * steps :],
                        torch.zeros_like(actual[:, -args.padded * steps :]),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scratch[-args.padded :],
                        torch.full_like(scratch[-args.padded :], 3.0),
                        rtol=0,
                        atol=0,
                    )
                case["output_max_abs_error"] = (
                    (actual.float() - expected.float()).abs().max().item()
                )
                case["snapshot_max_abs_error"] = (
                    (scratch.float() - expected_scratch.float()).abs().max().item()
                )
                timing, graph_output = measure_graph(
                    call,
                    capture_calls=args.capture_calls,
                    replays=args.replays,
                    samples=args.samples,
                )
                torch.testing.assert_close(graph_output, expected, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(
                    scratch, expected_scratch, rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)
                case.update(timing)
                case["status"] = "ok"
            except Exception as error:
                case.update(status="failed", error=f"{type(error).__name__}: {error}")
                failed = True
            report["cases"].append(case)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(case), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
