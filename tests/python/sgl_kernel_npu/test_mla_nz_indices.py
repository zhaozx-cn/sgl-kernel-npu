import unittest

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.mla_nz_indices import build_mla_nz_scatter_indices


def _reference(loc, dim, page_size):
    tiles = dim // 16
    return (
        (loc[:, None] // page_size * tiles + torch.arange(tiles, dtype=loc.dtype))
        * page_size
        + loc[:, None] % page_size
    ).reshape(-1, 1)


class TestMLANZIndices(unittest.TestCase):
    def test_dtypes_strides_and_replay(self):
        torch.manual_seed(17)
        for dtype in (torch.int32, torch.int64):
            for page_size in (16, 128):
                for rows in (0, 1, 7, 32, 256, 2048):
                    with self.subTest(dtype=dtype, page_size=page_size, rows=rows):
                        cpu = torch.randint(0, 200000, (rows,), dtype=dtype)
                        cpu[::7] = 0
                        storage = torch.empty(rows * 2, device="npu", dtype=dtype)
                        loc = storage[::2]
                        loc.copy_(cpu)
                        actual = build_mla_nz_scatter_indices(loc, 512, 64, page_size)
                        for value, dim in zip(actual, (512, 64)):
                            self.assertEqual(value.dtype, dtype)
                            self.assertTrue(
                                torch.equal(
                                    value.cpu(), _reference(cpu, dim, page_size)
                                )
                            )
                        if not rows:
                            continue
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            actual = build_mla_nz_scatter_indices(
                                loc, 512, 64, page_size
                            )
                        cpu = cpu.flip(0) + 129
                        if dtype == torch.int64:
                            cpu += 2**31
                        loc.copy_(cpu)
                        graph.replay()
                        torch.npu.synchronize()
                        for value, dim in zip(actual, (512, 64)):
                            self.assertTrue(
                                torch.equal(
                                    value.cpu(), _reference(cpu, dim, page_size)
                                )
                            )
                        del graph

    def test_invalid_metadata(self):
        loc = torch.empty(0, device="npu", dtype=torch.int32)
        with self.assertRaises(ValueError):
            build_mla_nz_scatter_indices(loc.float(), 512, 64, 128)
        with self.assertRaises(ValueError):
            build_mla_nz_scatter_indices(loc, 512, 63, 128)
        with self.assertRaises(ValueError):
            build_mla_nz_scatter_indices(loc, 512, 64, 0)


if __name__ == "__main__":
    unittest.main()
