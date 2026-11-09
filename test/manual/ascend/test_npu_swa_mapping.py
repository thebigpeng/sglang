"""Isolate NPU KV allocation and SWA mapping without loading a model or ZBAL.

Run in a fresh process: python test/manual/ascend/test_npu_swa_mapping.py
This synchronizes each stage; it does not cover graph replay or allocator overlap.
"""

import inspect
import logging
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sgl_kernel_npu.mem_cache import allocator as kernel_module

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)
PAGE_SIZE = 128
FULL_SIZE = 19_455_104  # Capacity in the 2026-11-09 IndexPutV2 failure log.
SWA_SIZE = 39_040


def allocate_and_check(allocator, tokens):
    page = int(allocator.free_pages[0].cpu())
    prefix_cpu = torch.tensor([0], dtype=torch.int64)
    seq_cpu = torch.tensor([tokens], dtype=torch.int64)
    indices = allocator.alloc_extend(
        prefix_cpu.to(allocator.device),
        prefix_cpu,
        seq_cpu.to(allocator.device),
        seq_cpu,
        torch.tensor([-1], dtype=torch.int64, device=allocator.device),
        tokens,
        num_new_pages=1,
    )
    actual = indices.cpu().to(torch.int64)
    expected = torch.arange(page * PAGE_SIZE, page * PAGE_SIZE + tokens)
    logger.info("ALLOC size=%d tokens=%d page=%d indices=%s", allocator.size, tokens, page, actual.tolist())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool(((actual >= PAGE_SIZE) & (actual < allocator.size + PAGE_SIZE)).all()), actual.tolist()
    return indices


def main():
    backend = torch.npu.memory.get_allocator_backend()
    if backend != "native":
        raise RuntimeError(f"This isolation test requires the native NPU allocator, got {backend}")
    torch.npu.set_device(0)
    logger.info("torch=%s torch_npu=%s allocator=%s", torch.__version__, torch_npu.__version__, backend)
    logger.info("SGLang allocator=%s", inspect.getfile(NPUPagedTokenToKVPoolAllocator))
    logger.info("alloc_extend_kernel=%s", kernel_module.__file__)
    pools = [
        NPUPagedTokenToKVPoolAllocator(size, PAGE_SIZE, torch.bfloat16, "npu:0", None, False)
        for size in (FULL_SIZE, SWA_SIZE)
    ]
    mapping = torch.cat(
        [
            torch.zeros(FULL_SIZE + PAGE_SIZE, dtype=torch.int64, device="npu:0"),
            torch.tensor([-1], dtype=torch.int64, device="npu:0"),
        ]
    )
    owner = SimpleNamespace(full_to_swa_index_mapping=mapping)
    # Advancing free_pages by one page also exercises its alternate pointer alignment.
    for tokens in (6, 1, 1, 6):
        full_indices, swa_indices = [allocate_and_check(pool, tokens) for pool in pools]
        logger.info("WRITE tokens=%d mapping_numel=%d", tokens, mapping.numel())
        SWATokenToKVPoolAllocator.set_full_to_swa_mapping(owner, full_indices, swa_indices)
        torch.npu.synchronize()
        logger.info("WRITE PASS tokens=%d", tokens)
        # Match the previous Index_int64_int32 failure as well as IndexPutV2.
        actual = mapping[full_indices].cpu()
        torch.testing.assert_close(actual, swa_indices.cpu().to(torch.int64), rtol=0, atol=0)
        logger.info("READ PASS tokens=%d", tokens)
    assert int(mapping[-1].cpu()) == -1
    logger.info("NPU_SWA_MAPPING PASS (native allocator, no model or DFC)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("NPU_SWA_MAPPING FAILED device=0 full_size=%d page_size=%d", FULL_SIZE, PAGE_SIZE)
        raise
