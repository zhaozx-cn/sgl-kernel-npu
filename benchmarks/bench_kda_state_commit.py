"""Compare KDA state commits with the current fallback using alternating graphs.

Run on an otherwise idle NPU for acceptance. Example:
  ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/bench_kda_state_commit.py \
    --mode conv --output conv.json --memory-fraction 0.1

Conv uses the full recorded [69,161,10,1152] BF16 shape and BS32. Temporal
defaults to memory-limited screening cases; --full-temporal adds the full
L69/BS32 source and persistent pool. Do not extrapolate screening timings to
full-shape or end-to-end performance. --baseline-kernel-file compares with an
earlier revision of kda_state_commit.py instead of the framework fallbacks.
"""

import argparse
import faulthandler
import gc
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
from sgl_kernel_npu.mamba import kda_state_commit as installed_candidate
from sgl_kernel_npu.mamba import mamba_state_update_triton as fallback
from sgl_kernel_npu.mamba import speculative_state_scatter as generic


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(operations, *, repetitions=20):
    graphs = {}
    samples = {name: [] for name in operations}
    for name, call in operations.items():
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(repetitions):
                call()
        graphs[name] = graph
        graph.replay()
        torch.npu.synchronize()
    names = list(operations)
    for trial in range(7):
        for name in names if trial % 2 == 0 else names[::-1]:
            graph = graphs[name]
            graph.replay()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (3 * repetitions))
    return {
        "samples_us": samples,
        "median_us": {n: statistics.median(v) for n, v in samples.items()},
        "operations_per_graph": repetitions,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("conv", "temporal", "snapshot"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-file", type=Path)
    parser.add_argument("--baseline-kernel-file", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.1)
    parser.add_argument("--full-temporal", action="store_true")
    parser.add_argument(
        "--conv-requests", type=int, choices=(1, 4, 8, 16, 32), default=32
    )
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(53)
    faulthandler.dump_traceback_later(45, repeat=True)
    old = (
        load("state_old", args.baseline_kernel_file)
        if args.baseline_kernel_file
        else None
    )
    new = (
        load("state_new", args.kernel_file) if args.kernel_file else installed_candidate
    )
    results = {
        "mode": args.mode,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "screening_only": True,
        "candidate_file": str(args.kernel_file or installed_candidate.__file__),
        "candidate_sha256": hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            Path(
                old.__file__
                if old is not None
                else (
                    generic.__file__ if args.mode == "snapshot" else fallback.__file__
                )
            ).read_bytes()
        ).hexdigest(),
        "baseline_file": (
            str(args.baseline_kernel_file)
            if args.baseline_kernel_file
            else "framework fallback"
        ),
        "cases": [],
    }
    if args.mode == "conv":
        # Exact profile shape: 69,161,10,1152, BF16; BS32.
        layers, pool, window, channels, steps = 69, 161, 10, 1152, 8
        requests = args.conv_requests
        original = torch.randn(layers, pool, window, channels, dtype=torch.bfloat16)
        dst = (torch.arange(requests, dtype=torch.int32) * 3 + 5).to("npu")
        a = original.to("npu")
        b = original.to("npu")
        accepted = torch.empty(requests, dtype=torch.int32, device="npu")
        for pattern in ("mixed", "step0", "step3", "step6", "step7"):
            step = torch.tensor(
                (
                    ([0, 1, 2, 3, 4, 5, 6, 7] * ((requests + 7) // 8))[:requests]
                    if pattern == "mixed"
                    else [int(pattern[-1])] * requests
                ),
                dtype=torch.int32,
            )
            if pattern == "mixed":
                step[0] = -1
            accepted.copy_(step)
            a.copy_(original)
            b.copy_(original)

            def baseline():
                if old is None or not old.commit_kda_extended_conv_state(
                    a, dst, dst, accepted, steps
                ):
                    fallback.conv_state_rollback(a, dst, accepted, steps)

            def candidate():
                assert new.commit_kda_extended_conv_state(b, dst, dst, accepted, steps)

            baseline()
            candidate()
            torch.npu.synchronize()
            torch.testing.assert_close(
                a[:, :, -3:].cpu(), b[:, :, -3:].cpu(), rtol=0, atol=0
            )
            result = measure({"baseline": baseline, "candidate": candidate})
            result.update(
                pattern=pattern,
                shape=list(original.shape),
                requests=requests,
                persistent_tail_bitwise_equal=True,
            )
            results["cases"].append(result)
            args.output.write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
        del a, b, baseline, candidate
    elif args.mode == "temporal":
        # Full L69/BS1 and BS32/L8 fit the test allocator cap. Do not extrapolate to L69/BS32.
        for layers, requests in [(69, 1), (8, 32)] + (
            [(69, 32)] if args.full_temporal else []
        ):
            src_pool = requests + 1
            dst_pool = 3 if requests == 1 else (161 if layers == 69 else 64)
            original = torch.randn(layers, dst_pool, 3, 128, 128, dtype=torch.bfloat16)
            source = torch.randn(
                layers, src_pool, 8, 3, 128, 128, dtype=torch.bfloat16
            ).to("npu")
            src = torch.arange(requests, dtype=torch.int32, device="npu")
            dst = (torch.arange(requests, dtype=torch.int32) + 1).to("npu")
            accepted = (torch.arange(requests, dtype=torch.int32) % 7).to("npu")
            for transposed in (False, True):
                a = (
                    original.transpose(-1, -2).contiguous().to("npu").transpose(-1, -2)
                    if transposed
                    else original.to("npu")
                )
                b = (
                    original.transpose(-1, -2).contiguous().to("npu").transpose(-1, -2)
                    if transposed
                    else original.to("npu")
                )

                def baseline():
                    if old is None or not old.move_kda_temporal_snapshot(
                        a, source, dst, src, accepted
                    ):
                        fallback.move_intermediate_cache_kda(
                            a, source, dst, src, accepted, h_block_size=1
                        )

                def candidate():
                    if not new.move_kda_temporal_snapshot(
                        b, source, dst, src, accepted
                    ):
                        fallback.move_intermediate_cache_kda(
                            b, source, dst, src, accepted, h_block_size=1
                        )

                baseline()
                candidate()
                torch.npu.synchronize()
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
                result = measure({"baseline": baseline, "candidate": candidate})
                result.update(
                    src_shape=list(source.shape),
                    dst_shape=list(a.shape),
                    dst_stride=list(a.stride()),
                    requests=requests,
                    transposed=transposed,
                    bitwise_equal=True,
                    full_profile_shape=(layers == 69 and requests == 32),
                )
                results["cases"].append(result)
                args.output.write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
                del a, b, baseline, candidate
                gc.collect()
                torch.npu.empty_cache()
            del source
            gc.collect()
            torch.npu.empty_cache()
    else:
        for requests in (1, 32):
            for channels in (1152, 2304):
                layers = 69
                pool = 3 if requests == 1 else 161
                original = torch.randn(layers, pool, channels, 3, dtype=torch.bfloat16)
                source = torch.randn(
                    layers, requests + 1, 8, channels, 3, dtype=torch.bfloat16
                ).to("npu")
                a = original.to("npu")
                b = original.to("npu")
                src = torch.arange(requests, dtype=torch.int32, device="npu")
                dst = (torch.arange(requests, dtype=torch.int32) + 1).to("npu")
                accepted = (torch.arange(requests, dtype=torch.int32) % 8).to("npu")

                def baseline():
                    if old is None or not old.scatter_kda_conv_snapshot(
                        a, source, dst, src, accepted
                    ):
                        generic.speculative_state_scatter_npu(
                            a, source, dst, src, accepted
                        )

                def candidate():
                    assert new.scatter_kda_conv_snapshot(b, source, dst, src, accepted)

                baseline()
                candidate()
                torch.npu.synchronize()
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
                result = measure({"baseline": baseline, "candidate": candidate})
                result.update(
                    requests=requests,
                    channels=channels,
                    src_shape=list(source.shape),
                    dst_shape=list(a.shape),
                    bitwise_equal=True,
                )
                results["cases"].append(result)
                args.output.write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
                del source, a, b, baseline, candidate
                gc.collect()
                torch.npu.empty_cache()
    results["peak_tensor_bytes"] = torch.npu.max_memory_allocated()
    args.output.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {"mode": args.mode, "peak_tensor_bytes": results["peak_tensor_bytes"]}
        ),
        flush=True,
    )
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
