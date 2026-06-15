# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Hand-written HIP ops for the prezero co-design of the Kimi-K2.5 MLA decode input
# stage: an upstream fused RMSNorm pre-zeroes a downstream split-K GEMM output buffer
# (a "free-rider"), so the GEMM runs zero_init=false (pure atomic-add, no self-zero
# semaphore / FillFunctor). See csrc/kernels/prezero_gemm/{mla_rmsnorm,splitk_gemm_a16w16}.cuh.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional

MD_NAME = "module_mla_rmsnorm"


@compile_ops("module_mla_rmsnorm")
def mla_add_rmsnorm(
    out: Tensor,
    residual_out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    eps: float,
    gemm_zero: Optional[Tensor] = None,
) -> None:
    """op1: x = input + residual_in; out = rmsnorm(x)*weight; residual_out = x.
    If gemm_zero is given (the qkv_a-GEMM output, [m, 2112]), also zero it (prezero co-design)."""
    ...


@compile_ops("module_mla_rmsnorm")
def mla_qk_rmsnorm(
    q_out: Tensor,
    k_out: Tensor,
    q_in: Tensor,
    k_in: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    q_eps: float,
    k_eps: float,
    gemm_zero: Optional[Tensor] = None,
    k_pe_out: Optional[Tensor] = None,
) -> None:
    """op3: q_out = rmsnorm(q_in)*q_weight; k_out = rmsnorm(k_in)*k_weight (QN=1536, KN=512).
    q_in/k_in may be column-slices of qkv_lora (row strides honored). If gemm_zero is given
    (the q_b-GEMM output, [m, 3072]), also zero it (prezero co-design). If k_pe_out [m, rope] is
    given, the K-plane also copies each row's rope cols (k_in[row]+KN) into it (k_pe free-rider,
    replaces a separate torch .contiguous() of the strided rope slice)."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_with_prezero(C: Tensor, A: Tensor, B: Tensor) -> None:
    """C[M,N] += A[M,K] @ B[N,K]^T (bf16, TN, split-K packed-atomic). C MUST be pre-zeroed
    (by the upstream mla_*_rmsnorm GZ prezero) — this GEMM runs zero_init=false. Wired (M=64):
    qkv_a (N=2112,K=7168), q_b (N=3072,K=1536)."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_prezero_tuned(
    C: Tensor, A: Tensor, B: Tensor, BN: int, SPLITK: int, BM: int
) -> None:
    """CSV-tuned prezero split-K GEMM: same kernel as splitk_gemm_with_prezero but the tile
    params (BN, SPLITK, BM) are passed at runtime (read from a16w16_prezero_tuned_gemm.csv by the
    tgemm_prezero dispatcher) and mapped to a precompiled launch<> instance. C MUST be
    pre-zeroed (zero_init=false). BK is fixed 128. See mla_gemm_host.cu for the instance set."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_bench(C: Tensor, A: Tensor, B: Tensor, BN: int, SPLITK: int) -> None:
    """BENCH-ONLY (router N=384,K=7168): same kernel with runtime (BN,SPLITK) to sweep CU fill.
    C pre-zeroed (zero_init=false). M in {32,64,128}; BK=128; SPLITK divides 56; BN in {16,32,64}."""
    ...
