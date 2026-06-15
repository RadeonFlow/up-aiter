// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// =============================================================================
// mla_rmsnorm — dedicated bf16 rmsnorm kernels for the Kimi-K2.5 MLA decode input stage.
// These are the upstream producers in the co-design pipeline that feeds tiny_pre_zero_splitk_gemm:
//   A) mla_add_rmsnorm : residual-add + rmsnorm (-> bf16). The decoder-layer input norm; its bf16
//      output feeds the qkv_a GEMM. (== aiter::add_rmsnorm_quant_kernel bf16 / FUSE_QUANT=false path)
//   B) mla_qk_rmsnorm  : q & k rmsnorm in one 2D-grid launch (blockIdx.y=0->Q, 1->K). Feeds the q_b
//      GEMM. (== aiter::fused_qk_rmsnorm_kernel bf16 path)
// Both share the core:  rcp = rsqrtf( blockReduce(sum x^2) / n + eps );  out = x * rcp * weight.
// A first forms x = input + residual_in and writes residual_out = x.
//
// Validated bit-exact vs the aiter originals (same DPP reduce order). Perf (m=64 decode, gfx950,
// same-session rocprof GPU-dur min-of-N/p50, min-of-N to filter MI350X VF clock noise):
//   WARM (serving regime, input HOT from upstream): A 2.12/3.08 vs aiter 2.64/3.48 (FASTER);
//        B 1.60/2.60 vs aiter 2.12/2.76 (FASTER).
//   COLD (input flushed from cache): A 4.16 vs 4.16 (TIED), B 3.56 vs 4.08 (faster). Cold is
//        HBM-read-bound; hoisting the weight load up front (see below) closed an earlier ~0.24us gap.
// Decisive warm win: full loop unroll (template N/QN/KN + bounded buffer loads -> all b128 loads
// unconditional & issued up front, no per-chunk OOB branch) + DPP block-reduce (ds 12->2).
//
// Load policy = CACHEABLE for input/residual (RMS_LOAD_AUX=0). aiter instead uses NT (`sc0 nt`) on
// input/residual + cacheable weight. Full M-sweep (4..256) + truly-hot test settles it:
//   - TRULY-HOT input (producer writes it right before — the real serving case, residual stream fresh
//     from upstream): CACHEABLE beats NT-input (M64 2.32 vs 2.40, M256 3.28 vs 3.56) — NT bypasses the
//     hot copy and re-reads HBM = waste. So cacheable wins serving at every M.
//   - COLD + large M (M=256, input evicted): aiter's NT-input wins (avoids polluting a thrashed L2),
//     our cacheable ~0.5us slower. But cold-large-M is NOT the serving regime.
//   => aiter trades hot-serving perf for the cold/cache-hygiene corner; we optimize hot-serving, so
//      cacheable is correct. (Interleave layout + NT-store also tried: no help / worse.)
// The shared WEIGHT, if ever streamed, must stay CACHEABLE (RMS_WEIGHT_AUX=0): all M CTAs reuse it;
// NT-ing it re-reads 14KB x M from HBM and erases any NT-input gain.
// Key optimizations:
//   * full loop unroll (compile-time N/QN/KN + bounded buffer loads, OOB->0) — the decisive warm win
//   * cache x in registers (read input+residual ONCE, not twice) + b128-vectorized load/store
//   * load weight UP FRONT (pass 1, not pass 2) so its cold-HBM latency overlaps input's instead of
//     being exposed after the reduce — closed the cold gap; free in warm (weight already cached)
//   * DPP wave reduce (no ds_bpermute) -> 2 ds / 1 barrier, and bit-exact vs aiter's reduce order
//
// PREZERO (co-design): mla_add_rmsnorm<N,GZ> with GZ>0 zeroes the downstream qkv_a-GEMM output [m*GZ]
// so that GEMM runs zero_init=false (pure atomic-add, no self-zero sema). Done via DEDICATED CTAs: the
// launcher uses grid = m + ZCTA; CTAs with blockIdx.x>=m ONLY zero (one 2048-elem chunk each, then
// return), running on the CUs the occupancy-starved m rmsnorm CTAs leave idle -> the prezero overlaps
// the rmsnorm rather than adding work to every rmsnorm CTA. Cost (GZ=2112): WARM +0.08us(M64)/+0.16us
// (M256) — ~33% cheaper than an inline-per-CTA zero; the residual is just the irreducible HBM write BW
// for the zeros. COLD ~free (overlaps the HBM read). Buys ~3-4us on the GEMM side => big net win.
// GZ=0 default => grid m, kernel byte-identical. B (mla_qk_rmsnorm<QN,KN,GZ>) does the same for the
// q_b-GEMM out [m*GZ] (GZ=3072), its prezero CTAs split across BOTH grid.y planes — cost at m=64 is
// FREE (warm +0.00us; the qk-norm is tiny + more occupancy-starved, so the zeros fully hide). Both
// kernels also honor runtime row strides (free; q_c/kv_c are column-slices of qkv_lora). TODO: wire
// mla_add_rmsnorm + mla_qk_rmsnorm into ATOM _fuse_rmsnorm_quant + the tiny_pre_zero_splitk_gemm op.
// =============================================================================
#pragma once

#include "device_prims.cuh"     // bf16 vector types, bounded b128 buffer IO, DPP block_reduce_sum

namespace aiter::prezero_gemm {

constexpr int RMS_BS = 256;     // threads/block (one block per token row)

#ifndef RMS_LOAD_AUX
#define RMS_LOAD_AUX 0          // 0 = cacheable (input is HOT in serving); -DRMS_LOAD_AUX=3 = nt/streaming
#endif
#ifndef RMS_WEIGHT_AUX
#define RMS_WEIGHT_AUX 0       // shared weight -> cacheable (caches across all CTAs)
#endif
#ifndef RMS_STORE_AUX
#define RMS_STORE_AUX 0        // 0 = cacheable; 3 = nt streaming store (no L2 write-allocate)
#endif
#ifndef RMS_INTERLEAVE
#define RMS_INTERLEAVE 0        // 0 = contiguous-segment chunks; 1 = warp-interleave (waves spread across the row)
#endif

// ---- A: residual-add + rmsnorm (bf16). grid = m blocks (one row each). Templated on N (the hidden
// dim is fixed at build time) so the chunk loop is FULLY UNROLLED -> all b128 loads issue up front
// (latency hiding under the m=64 low-occupancy regime). N must be a multiple of 8. ----
template<int N, int GZ = 0>     // GZ>0: also zero GZ bf16 of this token's downstream splitK-GEMM out row
__global__ void __launch_bounds__(RMS_BS)
mla_add_rmsnorm(bf16* __restrict__ out, bf16* __restrict__ residual_out,
                const bf16* __restrict__ input, const bf16* __restrict__ residual_in,
                const bf16* __restrict__ weight, double eps, int m,   // eps double: matches aiter's add_rmsnorm_quant_kernel
                bf16* __restrict__ gemm_zero = nullptr,
                int in_stride = N, int rin_stride = N, int rout_stride = N, int out_stride = N){
    int t = threadIdx.x;
    constexpr int V = 8;
    // co-design PREZERO via DEDICATED CTAs (blockIdx.x >= m): they zero the qkv_a-GEMM output buffer
    // [m*GZ] so the GEMM runs zero_init=false (pure atomic-add, no self-zero sema). Launched as extra
    // grid (m + ZCTA); at decode the m rmsnorm CTAs leave most CUs idle, so these run CONCURRENTLY on
    // the idle CUs -> the prezero overlaps the rmsnorm instead of adding work to every rmsnorm CTA.
    // Each prezero CTA zeros one RMS_BS*V (=2048) elem chunk; OOB tail dropped by the buffer bound.
    if constexpr(GZ > 0){
        if(blockIdx.x >= m){
            buffer_rsrc_t Zr = make_bounded_rsrc(gemm_zero, (unsigned)((size_t)m*GZ*sizeof(bf16)));
            const bf16x8 zero8 = {};
            buffer_store_b128<RMS_STORE_AUX>(zero8, Zr, ((blockIdx.x - m)*RMS_BS + t)*V*(int)sizeof(bf16));
            return;
        }
    }
    int row = blockIdx.x;   // rmsnorm CTA: one token
    constexpr int NCHUNK = (N + RMS_BS*V - 1) / (RMS_BS*V);   // b128 chunks/thread (compile-time)
    constexpr unsigned RB = (unsigned)N * sizeof(bf16);       // row size in bytes (buffer bound)
    int wave = t >> 6, lane = t & 63;
    // byte offset of thread t's chunk c. INTERLEAVE: the NW waves each own a different N/NW segment of
    // the row and read them simultaneously (HBM channels spread); CONTIGUOUS: chunk c is one 2048-elem
    // segment read by all threads at once (channels concentrated). Same element set, just reassigned.
    auto voff = [&](int c) -> int {
#if RMS_INTERLEAVE
        return (wave*(64*NCHUNK*V) + lane*V + c*(64*V)) * (int)sizeof(bf16);
#else
        return (t + c*RMS_BS) * V * (int)sizeof(bf16);
#endif
    };
    // honor runtime row strides (measured FREE vs compile-time row*N; needed for sliced inputs, e.g.
    // q_c/kv_c being column-slices of qkv_lora). Default strides = N => contiguous. weight has no row stride.
    buffer_rsrc_t Ir  = make_bounded_rsrc(input       + (size_t)row*in_stride,   RB);
    buffer_rsrc_t Rr  = make_bounded_rsrc(residual_in + (size_t)row*rin_stride,  RB);
    buffer_rsrc_t Wr  = make_bounded_rsrc(weight, RB);
    buffer_rsrc_t ROr = make_bounded_rsrc(residual_out + (size_t)row*rout_stride, RB);
    buffer_rsrc_t Or  = make_bounded_rsrc(out          + (size_t)row*out_stride,  RB);
    float xc[NCHUNK*V]; bf16x8 wc[NCHUNK]; float partial = 0.f;
    // pass 1: load input+residual+weight UP FRONT (so weight's cold-HBM latency overlaps input's, not
    // exposed after the reduce), form x, cache x and weight, write residual_out.
    #pragma unroll
    for(int c = 0; c < NCHUNK; c++){ int vo_b = voff(c);
        bf16x8 vi = buffer_load_b128<RMS_LOAD_AUX>(Ir, vo_b), vr = buffer_load_b128<RMS_LOAD_AUX>(Rr, vo_b), vo;
        wc[c] = buffer_load_b128<RMS_WEIGHT_AUX>(Wr, vo_b);
        #pragma unroll
        for(int j=0;j<V;j++){ float x=(float)vi[j]+(float)vr[j]; xc[c*V+j]=x; vo[j]=(__bf16)x; partial += x*x; }
        buffer_store_b128<RMS_STORE_AUX>(vo, ROr, vo_b);
    }
    float rcp = rsqrtf(block_reduce_sum<RMS_BS>(partial) / N + eps);   // OOB lanes added 0, so partial is exact
    // pass 2: normalize the cached x with the cached weight -> out (no loads here -> nothing to wait on).
    #pragma unroll
    for(int c = 0; c < NCHUNK; c++){ int vo_b = voff(c);
        bf16x8 vo;
        #pragma unroll
        for(int j=0;j<V;j++) vo[j] = (__bf16)(xc[c*V+j] * rcp * (float)wc[c][j]);
        buffer_store_b128<RMS_STORE_AUX>(vo, Or, vo_b);
    }
}

// ---- B: q/k rmsnorm (bf16). 2D grid: blockIdx.x = row, blockIdx.y = 0(Q)/1(K) run in parallel.
// Templated on QN/KN (compile-time) + bounded buffer loads -> fully unrolled, same as A. ----
template<int QN, int KN, int GZ = 0>    // GZ>0: also zero the downstream q_b-GEMM output [m*GZ]
__global__ void __launch_bounds__(RMS_BS)
mla_qk_rmsnorm(bf16* __restrict__ q_out, bf16* __restrict__ k_out,
               const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
               const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight,
               float q_eps, float k_eps, int m,
               bf16* __restrict__ gemm_zero = nullptr,
               int q_in_stride = QN, int k_in_stride = KN, int q_out_stride = QN, int k_out_stride = KN,
               bf16* __restrict__ k_pe_out = nullptr, int rope = 0){
    // PREZERO via dedicated CTAs (blockIdx.x >= m), split across BOTH grid.y planes so the q & k planes'
    // spare CTAs both help zero the q_b-GEMM out [m*GZ]. Run on idle CUs -> overlap the qk-norm.
    if constexpr(GZ > 0){
        if(blockIdx.x >= m){
            int chunk = (blockIdx.x - m) * gridDim.y + blockIdx.y;   // enumerate extras over both planes
            buffer_rsrc_t Zr = make_bounded_rsrc(gemm_zero, (unsigned)((size_t)m*GZ*sizeof(bf16)));
            const bf16x8 zero8 = {};
            buffer_store_b128<RMS_STORE_AUX>(zero8, Zr, (chunk*RMS_BS + threadIdx.x)*8*(int)sizeof(bf16));
            return;
        }
    }
    int row = blockIdx.x; if(row >= m) return;
    bool is_q = (blockIdx.y == 0);
    int n = is_q ? QN : KN; float eps = is_q ? q_eps : k_eps;
    int in_stride  = is_q ? q_in_stride  : k_in_stride;     // q_in/k_in may be column-slices of qkv_lora
    int out_stride = is_q ? q_out_stride : k_out_stride;
    const bf16* in = (is_q ? q_in : k_in) + (size_t)row*in_stride;
    bf16* o        = (is_q ? q_out : k_out) + (size_t)row*out_stride;
    const bf16* w  = is_q ? q_weight : k_weight;
    int t = threadIdx.x;
    constexpr int V = 8;
    constexpr int NCHUNK = ((QN > KN ? QN : KN) + RMS_BS*V - 1) / (RMS_BS*V);   // compile-time
    unsigned RB = (unsigned)n * sizeof(bf16);
    buffer_rsrc_t Ir = make_bounded_rsrc(in, RB), Wr = make_bounded_rsrc(w, RB), Or = make_bounded_rsrc(o, RB);
    float xc[NCHUNK*V]; bf16x8 wc[NCHUNK]; float partial = 0.f;
    // load input + weight UP FRONT (hide weight's cold latency with input's), cache both.
    #pragma unroll
    for(int c = 0; c < NCHUNK; c++){ int vb = (t + c*RMS_BS) * V * (int)sizeof(bf16);
        bf16x8 vi = buffer_load_b128<RMS_LOAD_AUX>(Ir, vb);
        wc[c] = buffer_load_b128<RMS_WEIGHT_AUX>(Wr, vb);
        #pragma unroll
        for(int j=0;j<V;j++){ float x=(float)vi[j]; xc[c*V+j]=x; partial += x*x; }
    }
    float rcp = rsqrtf(block_reduce_sum<RMS_BS>(partial) / n + eps);
    #pragma unroll
    for(int c = 0; c < NCHUNK; c++){ int vb = (t + c*RMS_BS) * V * (int)sizeof(bf16);
        bf16x8 vo;
        #pragma unroll
        for(int j=0;j<V;j++) vo[j] = (__bf16)(xc[c*V+j] * rcp * (float)wc[c][j]);
        buffer_store_b128<RMS_STORE_AUX>(vo, Or, vb);
    }
    // k_pe free-rider: the K-plane row CTA also copies this row's `rope` rope-cols — which sit right
    // after the KN kv cols in the SAME strided qkv row (k_in[row] + KN) — into a contiguous k_pe_out.
    // Fuses what was a separate torch direct_copy of qkv_zeroed[:, KN:KN+rope].contiguous() into op3.
    if(!is_q && k_pe_out != nullptr && rope > 0){
        const bf16* kpe = in + KN;
        for(int i = t; i < rope; i += RMS_BS) k_pe_out[(size_t)row*rope + i] = kpe[i];
    }
}

// ---- host launchers ----
template<int N, int GZ = 0>
inline void launch_mla_add_rmsnorm(bf16* out, bf16* residual_out, const bf16* input,
                                   const bf16* residual_in, const bf16* weight, double eps,
                                   int m, hipStream_t stream, bf16* gemm_zero = nullptr,
                                   int in_stride = N, int rin_stride = N, int rout_stride = N, int out_stride = N){
    int z = (GZ > 0) ? (m*GZ + RMS_BS*8 - 1) / (RMS_BS*8) : 0;   // dedicated prezero CTAs
    mla_add_rmsnorm<N, GZ><<<dim3(m + z), dim3(RMS_BS), 0, stream>>>(out, residual_out, input, residual_in, weight, eps, m, gemm_zero, in_stride, rin_stride, rout_stride, out_stride);
}
template<int QN, int KN, int GZ = 0>
inline void launch_mla_qk_rmsnorm(bf16* q_out, bf16* k_out, const bf16* q_in, const bf16* k_in,
                                  const bf16* q_weight, const bf16* k_weight, float q_eps, float k_eps,
                                  int m, hipStream_t stream, bf16* gemm_zero = nullptr,
                                  int q_in_stride = QN, int k_in_stride = KN, int q_out_stride = QN, int k_out_stride = KN,
                                  bf16* k_pe_out = nullptr, int rope = 0){
    int zx = (GZ > 0) ? (((size_t)m*GZ + RMS_BS*8 - 1)/(RMS_BS*8) + 1) / 2 : 0;   // prezero CTAs split over the 2 planes
    mla_qk_rmsnorm<QN,KN,GZ><<<dim3(m + zx, 2), dim3(RMS_BS), 0, stream>>>(q_out, k_out, q_in, k_in, q_weight, k_weight, q_eps, k_eps, m, gemm_zero, q_in_stride, k_in_stride, q_out_stride, k_out_stride, k_pe_out, rope);
}

}  // namespace aiter::prezero_gemm
