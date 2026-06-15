// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// =============================================================================
// device_prims.cuh — reusable device primitives for the hand-written Kimi-K2.5 MLA co-design kernels
// (mla_rmsnorm, tiny_pre_zero_splitk_gemm, ...). Self-contained plain HIP (no opus dependency).
//
//   * bf16 vector types
//   * bounded b128 buffer load/store: a per-region buffer descriptor whose bound is the region size,
//     so out-of-bounds loads return 0 and OOB stores are dropped. This lets a chunk loop be FULLY
//     UNROLLED with UNCONDITIONAL b128 loads/stores (no per-chunk OOB branch) — the loads then issue
//     up front and hide memory latency, which is the decisive win for the low-occupancy decode regime.
//   * DPP wave/block reduce (sum): the 64-lane wavefront reduce runs cross-lane in registers via DPP
//     (NO LDS / no ds_bpermute, which is what __shfl lowers to), then combines the wave partials.
// =============================================================================
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace aiter::prezero_gemm {

// identical `using` aliases may repeat across TUs, so these don't clash with the GEMM kernel's defs.
using bf16   = __hip_bfloat16;
using bf16x8 = __attribute__((ext_vector_type(8))) __bf16;
using i32x4  = __attribute__((ext_vector_type(4))) int;
using buffer_rsrc_t = __amdgpu_buffer_rsrc_t;

// ---- bounded b128 buffer IO ------------------------------------------------
// Descriptor over [p, p+nbytes): loads past nbytes return 0, stores past it are dropped.
__device__ __forceinline__ buffer_rsrc_t make_bounded_rsrc(const void* p, unsigned nbytes){
    return __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, nbytes, 0x00020000);
}
// AUX=0 cacheable (default — the consumer's input is HOT from upstream, so cacheable wins; measured);
// AUX=3 emits `sc0 nt` (streaming). voff is a BYTE offset into the bound region.
template<int AUX = 0>
__device__ __forceinline__ bf16x8 buffer_load_b128(buffer_rsrc_t r, int voff){
    return __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, AUX));
}
template<int AUX = 0>   // AUX=0 cacheable; AUX=3 -> `sc0 nt` (streaming, no L2 write-allocate)
__device__ __forceinline__ void buffer_store_b128(bf16x8 v, buffer_rsrc_t r, int voff){
    __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(i32x4, v), r, voff, 0, AUX);
}

// ---- DPP block reduce (sum) ------------------------------------------------
// gfx9 wave64 sum: row_shr 1/2/4/8 (reduce within each 16-lane row) + row_bcast 15/31 (combine the
// four rows) -> lane 63 holds the wave sum; broadcast it, then combine the BLOCK/64 wave partials
// through one shared round. Returns the full block sum to every thread. 2 ds / 1 barrier (or 0/0 for
// a single-wave block) vs ~12 ds for a __shfl tree.
template<int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v){
    #define DPP_ADD(ctrl) v += __builtin_bit_cast(float, \
        __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), (ctrl), 0xf, 0xf, false))
    DPP_ADD(0x111); DPP_ADD(0x112); DPP_ADD(0x114); DPP_ADD(0x118);   // row_shr:1,2,4,8
    DPP_ADD(0x142); DPP_ADD(0x143);                                   // row_bcast:15,31
    #undef DPP_ADD
    v = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
    constexpr int NW = BLOCK / 64;
    if constexpr(NW == 1) return v;                          // single wave: already done
    __shared__ float sh[NW];
    if((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
    __syncthreads();
    float s = 0.f;
    #pragma unroll
    for(int i = 0; i < NW; i++) s += sh[i];                  // all threads get the same block sum
    return s;
}

}  // namespace aiter::prezero_gemm
