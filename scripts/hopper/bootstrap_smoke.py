#!/usr/bin/env python
"""Smoke-test the Hopper training environment after bootstrap."""

import torch
import torch.distributed.fsdp as fsdp

if not hasattr(fsdp, "FSDPModule") and hasattr(fsdp, "FullyShardedDataParallel"):
    fsdp.FSDPModule = fsdp.FullyShardedDataParallel

import trl
from trl import DPOConfig, DPOTrainer, RLOOConfig, RLOOTrainer


def main() -> None:
    print(f"[bootstrap_remote] torch_version={torch.__version__}")
    print(f"[bootstrap_remote] torch_cuda={torch.version.cuda}")
    print(f"[bootstrap_remote] torch_path={torch.__file__}")
    print(f"[bootstrap_remote] trl_version={trl.__version__}")
    print(f"[bootstrap_remote] fsdp_module_alias={hasattr(fsdp, 'FSDPModule')}")
    print(f"[bootstrap_remote] trl_imports_ok={all([DPOConfig, DPOTrainer, RLOOConfig, RLOOTrainer])}")


if __name__ == "__main__":
    main()
