# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# mla_decode_reduce — MLA decode LSE-reduce (CSV-tuned).
#
# mla_decode_reduce replaces aiter.mla_reduce_v1's LSE-weighted reduce across
# the KV-splits (fp32 → bf16) on the plain-TP MLA decode post-attention path.
#
# It is a thin @compile_ops binding to a C++ host entry in
# csrc/kernels/mla/decode_reduce/mla_decode_reduce.cu. The C++ side switches on
# the tuning knobs (num_splits, batch, vec) to pick the right template
# instantiation.
#
# Adding a new shape: extend the dispatch switch in C++ + add a row to
# aiter/configs/tuned_mla_decode_reduce.csv.

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_mla_decode_reduce")
def mla_decode_reduce(
    partial_output: Tensor,  # [T*N_SPLITS, H, K]  fp32
    partial_lse: Tensor,     # [T*N_SPLITS, H]     fp32
    reduced: Tensor,         # [T, H, K]           bf16 (out)
    kv_indptr: Tensor,       # [T+1]               int32
    T: int,
    batch: int,
    vec: int = 1,            # fp32/thread along D_V (1/2/4 → b32/b64/b128)
    num_splits: int = 16,    # KV-split count (concurrency-adaptive); buffer is
                             # [T*num_splits, H, K]
) -> None: ...
