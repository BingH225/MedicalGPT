"""Compatibility helpers for TRL on older torch releases."""

import torch.distributed.fsdp as fsdp


def patch_fsdp_module_alias() -> bool:
    """Backfill torch.distributed.fsdp.FSDPModule when TRL expects it."""
    if hasattr(fsdp, "FSDPModule"):
        return False
    if hasattr(fsdp, "FullyShardedDataParallel"):
        fsdp.FSDPModule = fsdp.FullyShardedDataParallel
        return True
    return False
