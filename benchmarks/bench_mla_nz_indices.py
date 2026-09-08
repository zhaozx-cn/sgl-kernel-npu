"""Alternating graph A/B for paired NZ indices plus both CANN cache scatters."""

import argparse
import gc
import json
import statistics
import time

import torch
import torch_npu
from sgl_kernel_npu.mem_cache.mla_nz_indices import build_mla_nz_scatter_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", type=int, default=1563)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(17)
    results = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cases": [],
    }
    for tokens in args.tokens:
        for strided in (False, True):
            storage = torch.randn(tokens, 2112, device="npu", dtype=torch.bfloat16)
            values = [storage[:, :512], storage[:, 512:576]]
            if not strided:
                values = [value.contiguous() for value in values]
            loc = (torch.randperm(args.pages * 128 - 1)[:tokens] + 1).to(
                "npu", torch.int32
            )
            caches = [
                [
                    torch.zeros(
                        args.pages, 128, 1, dim, device="npu", dtype=torch.bfloat16
                    )
                    for dim in (512, 64)
                ]
                for _ in range(2)
            ]

            def write(fused, buffers=caches):
                if fused:
                    indices = build_mla_nz_scatter_indices(loc, 512, 64, 128)
                else:
                    indices = []
                    for dim in (512, 64):
                        page = torch.div(loc, 128, rounding_mode="floor")
                        slot = torch.remainder(loc, 128)
                        tiles = torch.arange(
                            dim // 16, dtype=loc.dtype, device=loc.device
                        )
                        indices.append(
                            (
                                (page[:, None] * (dim // 16) + tiles) * 128
                                + slot[:, None]
                            ).reshape(-1, 1)
                        )
                for cache, value, index in zip(buffers[int(fused)], values, indices):
                    torch_npu.npu_scatter_nd_update_(
                        cache.view(-1, 16), index, value.contiguous().view(-1, 16)
                    )

            graphs = []
            for fused in (False, True):
                for _ in range(3):
                    write(fused)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    write(fused)
                graphs.append(graph)
                for _ in range(20):
                    graph.replay()
            torch.npu.synchronize()
            for a, b in zip(*caches):
                for start in range(0, args.pages, 64):
                    assert torch.equal(
                        a[start : start + 64].cpu().view(torch.int16),
                        b[start : start + 64].cpu().view(torch.int16),
                    )
            samples = [[], []]
            for trial in range(args.trials):
                for variant in (0, 1) if trial % 2 == 0 else (1, 0):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(args.repeats):
                        graphs[variant].replay()
                    torch.npu.synchronize()
                    samples[variant].append(
                        (time.perf_counter_ns() - start) / args.repeats / 1000
                    )
            result = {
                "tokens": tokens,
                "strided": strided,
                "samples_us": samples,
                "baseline_us": statistics.median(samples[0]),
                "candidate_us": statistics.median(samples[1]),
            }
            results["cases"].append(result)
            with open(args.output, "w") as file:
                json.dump(results, file, indent=2)
            print(json.dumps(result), flush=True)
            del graphs, graph, caches, write, a, b
            gc.collect()
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
