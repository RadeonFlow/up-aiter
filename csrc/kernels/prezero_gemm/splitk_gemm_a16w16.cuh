// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// =============================================================================
// splitk_gemm_a16w16 — hand-written HIP a16w16 (bf16) split-K GEMM for the Kimi-K2.5 MLA projections.
//   C[M,N] = A[M,K] * B[N,K]^T      (TN layout; bf16 in / fp32 accumulate / bf16 out)
//   gfx950 / MI350X. The HIP counterpart to the asm bf16gemm_fp32bf16_tn_*_splitk_clean
//   kernels (hsa/gfx950/bf16gemm/*.co); measured ~1.4-1.6x faster on the 64x64/32x64 shapes.
//
// Strategy
//   * splitK — K is cut into SPLITK slices; one threadblock per (N-tile, K-slice).
//     grid = (N/BN, SPLITK). Each block computes a partial C tile over its K-slice and
//     atomically adds it into the (pre-zeroed) global C.
//   * MFMA v_mfma_f32_16x16x32_bf16 (K=32): each lane feeds 8 CONTIGUOUS bf16 (= one b128)
//     per instruction, so a PLAIN A[BM][BK] LDS tile (identical to global, stored as-is) gives
//     BOTH b128 LDS writes AND b128 LDS reads. An 8-block XOR swizzle (phys()) makes both
//     0 bank-conflict.
//   * Reduction — PACKED bf16 atomics (global_atomic_pk_add_bf16, 2 bf16/op) straight into bf16
//     C: half the atomic traffic of fp32 atomicAdd, no separate fp32->bf16 pass. Per-slice K
//     accumulates in fp32; only the cross-slice sum rounds in bf16 (== the asm is_out_b16 path).
//
// Tile / wave layout (4 waves = 256 threads). The BMxBN output tile is split 2(M) x 2(N):
//   nchunk = wave % (BN/16)         which 16-col half (N-chunk)
//   mbase  = (wave/(BN/16)) * MCH_W which 32-row half (first of this wave's MCH_W m-chunks)
//
// MFMA 16x16x32 per-lane mapping (lane = g*16 + e;  g = lane>>4 in[0,4), e = lane&15 in[0,16)):
//   INPUT  operand: lane supplies the 8-bf16 K-octet at k = kk*32 + g*8, for row = chunk*16 + e
//   OUTPUT acc[v]:  lane owns 4 C rows (g*4 + v, v in[0,4)) at C column = e
//   => inputs: g selects the K-octet, e the row; outputs: g the row-quad, e the column (the MFMA
//      register transpose). That transpose is why the epilogue cshuffles the accumulator through
//      LDS before packing 2 *adjacent columns* (the only 32-bit-contiguous unit) for the atomic.
// =============================================================================
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "opus/opus.hpp"   // shared device infra (make_buffer_rsrc, fp32 vector types); the layer a4w4 builds on

namespace aiter::prezero_gemm {

// Vector types + rsrc maker are reused from opus (see make_buffer_rsrc below). gfx950 = ROCm 7+ =
// clang 20+, where opus's REGISTER_DTYPE backs bf16_t with __bf16, so opus::bf16x*_t feed the bf16
// MFMA / packed-atomic builtins directly. (The static_assert fails loudly if ever built where it
// isn't __bf16 — e.g. clang<20, which gfx950 never uses.)
static_assert(std::is_same_v<opus::bf16_t, __bf16>, "opus bf16_t must be __bf16 (ROCm 7+/clang 20+)");
using bf16     = __hip_bfloat16;       // external (host/global) bf16 element; torch bf16 ABI
using bf16x4   = opus::bf16x4_t;       // __bf16 ext_vector_type(4)
using bf16x8   = opus::bf16x8_t;       // MFMA operand (8 contiguous bf16 = one b128)
using bhalf2_t = opus::bf16x2_t;       // packed-atomic operand
using floatx4  = opus::fp32x4_t;
using buffer_rsrc_t = __amdgpu_buffer_rsrc_t;

// raw_buffer_store aux bit4 = SC1 (push-to-LLC, cross-XCD coherent) — used for the splitK zeroing.
constexpr int AUXZ = 16;
// MFMA-fixed constants (independent of shape/tile).
constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 32;  // one MFMA computes MFMA_M x MFMA_N x MFMA_K
constexpr int KOCT     = 8;            // bf16 per lane per MFMA operand (= one b128)
constexpr int ACC_ROWS = 4;            // C rows held per lane in the fp32 accumulator (floatx4)
constexpr int BF16B    = sizeof(bf16); // bf16 size in bytes (= 2), for global byte offsets

// ---- device helpers: make_buffer_rsrc reuses opus; the rest are kernel-specific (opus has no
//      packed-bf16 atomic, and rot021 / the b64 load are local — same split as a4w4). ----
__device__ __forceinline__ buffer_rsrc_t make_buffer_rsrc(const void* base, unsigned nbytes){
    return opus::make_buffer_rsrc(base, nbytes, 0x00020000);
}
__device__ __forceinline__ bf16x4 buf_load_b64(buffer_rsrc_t r, int voff, int soff){
    return __builtin_bit_cast(bf16x4, __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0));
}
__device__ __forceinline__ void atomic_pk_add_bf16(bf16* addr, bhalf2_t val){
    __builtin_amdgcn_global_atomic_fadd_v2bf16(reinterpret_cast<bhalf2_t*>(addr), val);
}
// 8-block XOR swizzle helper (see phys() inside the kernel). rot021 is tile-independent.
__device__ __forceinline__ int rot021(int e){ return ((e & 6) >> 1) | ((e & 1) << 2) | (e & 8); }

// sema: per-(device,stream) workspace of N/BN ints, init to -1 once; caller increments `epoch` per
// launch (the asm semaphore protocol). ZERO_INIT==false skips zeroing (caller pre-zeroed C).
template<int M, int N, int K, int BM, int BN, int BK, int SPLITK, bool ZERO_INIT>
__global__ void __launch_bounds__(256, 2)
bf16gemm_mfma32_splitk_pk(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
                          int* __restrict__ sema, int epoch){
    constexpr int KSLICE   = K / SPLITK;                    // K handled by one split
    static_assert(K % SPLITK == 0 && KSLICE % BK == 0, "bad splitK");
    constexpr int NCH      = BN / MFMA_N;                   // n-chunks per tile
    constexpr int MGROUPS  = 4 / NCH;                       // wave M-groups
    constexpr int MCH_W    = BM / (MGROUPS * MFMA_M);       // m-chunks per wave (= 4/MGROUPS when BM=64;
                                                            // BM=128 doubles it -> acc[] & A-LDS double)
    constexpr int KK       = BK / MFMA_K;                   // MFMA K-steps per tile
    constexpr int A_CHUNKS = BM * BK / KOCT / 256;          // b128 A chunks loaded per thread
    constexpr int B_TOTAL  = BN * BK / KOCT;                // b128 B chunks total
    constexpr int B_CHUNKS = (B_TOTAL + 255) / 256;         // b128 B chunks loaded per thread

    // LDS address of [row][k] in a [rows][BK] bf16 tile. 8-block XOR swizzle: the in-block offset
    // (k&7) stays contiguous (reads/writes remain b128); the block index (k>>3) is XOR-scattered by
    // rot021(row&15) so the 16 MFMA rows hit distinct banks -> 0 conflict for both b128 read & write.
    auto phys = [](int row, int k){ return row * BK + (((k >> 3) ^ rot021(row & 15)) << 3) + (k & 7); };

    const int n_tile = blockIdx.x;   // which BN-column tile of C this block owns
    const int ksplit = blockIdx.y;   // which K-slice this block accumulates
    const int mtile  = blockIdx.z;   // which BM-row tile of C (M-tiling; grid.z = ceil(M/BM))
    const int mrow0  = mtile * BM;   // first global A/C row this block owns (0 when M<=BM)

    // splitK zero-init: the ksplit==0 block zeroes this tile's bf16 C slab, then flags sema[n_tile];
    // the other splits spin on that flag before accumulating (epilogue). ZERO_INIT==false strips it.
    if constexpr(ZERO_INIT) if(ksplit == 0){
        buffer_rsrc_t Cr = make_buffer_rsrc(C, (unsigned)M * N * BF16B);
        for(int idx = threadIdx.x; idx < M * (BN / 2); idx += 256){
            int m = idx / (BN / 2), col = (idx % (BN / 2)) * 2;     // (row, even column) in slab
            __builtin_amdgcn_raw_buffer_store_b32(0, Cr, (m * N + n_tile * BN + col) * BF16B, 0, AUXZ);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");           // drain stores (syncthreads has no vmcnt)
        __syncthreads();
        if(threadIdx.x == 0)
            __hip_atomic_store(&sema[n_tile], epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    // s_A/s_B are live across the main loop; the fp32 cshuffle image is live only in the epilogue.
    // Disjoint lifetimes -> the epilogue overlays its scratch onto s_A (see lds_f) so total LDS stays
    // 48KB (the double-buffered A+B floor, OCC3). NB: a literal `union {struct{s_A,s_B}; float[]}` was
    // tried and rejected: even with alignas(16) it makes the compiler roll the K-loop (~6% slower).
    __shared__ bf16 s_A[2][BM * BK];     // double-buffered A tile (epilogue reuses it as cshuffle scratch)
    __shared__ bf16 s_B[2][BN * BK];     // double-buffered B tile
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63, g = lane >> 4, e = lane & 15;
    const int nchunk = wave % NCH, mbase = (wave / NCH) * MCH_W;   // this wave's N-chunk / first m-chunk
    const int n0 = n_tile * BN, kbeg = ksplit * KSLICE, ntiles = KSLICE / BK;
    const buffer_rsrc_t Ar = make_buffer_rsrc(A, (unsigned)M * K * BF16B);
    const buffer_rsrc_t Br = make_buffer_rsrc(B, (unsigned)N * K * BF16B);

    floatx4 acc[MCH_W];                  // fp32 accumulators, one floatx4 (= 4 C rows) per m-chunk
    #pragma unroll
    for(int i = 0; i < MCH_W; i++) acc[i] = {0, 0, 0, 0};

    // Per-thread global<->LDS coordinates for the A / B tile loads (each thread moves
    // A_CHUNKS/B_CHUNKS b128s; a_row/a_k are the tile-local (row, k) of chunk c).
    int a_row[A_CHUNKS], a_k[A_CHUNKS];
    #pragma unroll
    for(int c = 0; c < A_CHUNKS; c++){ int j = tid + c * 256; a_row[c] = j / (BK / KOCT); a_k[c] = (j % (BK / KOCT)) * KOCT; }
    int b_row[B_CHUNKS], b_k[B_CHUNKS]; bool b_valid[B_CHUNKS];
    #pragma unroll
    for(int c = 0; c < B_CHUNKS; c++){ int j = tid + c * 256; b_valid[c] = j < B_TOTAL; b_row[c] = j / (BK / KOCT); b_k[c] = (j % (BK / KOCT)) * KOCT; }

    // ==== pipeline stages (one lambda each); data flow of a K-tile is loadX -> storeX -> compute ====
    //   loadA/loadB : global -> registers  (coalesced buf_load_b64 x2 = one b128 chunk/thread)
    //   storeA/storeB: registers -> LDS     (two bf16x4 merge into one swizzled ds_write_b128)
    //   compute     : LDS -> MFMA          (lane reads its 8-K-octet b128, one mfma per m-chunk)
    auto loadA = [&](int kt, bf16x4* r){
        #pragma unroll
        for(int c = 0; c < A_CHUNKS; c++){ int o = ((mrow0 + a_row[c]) * K + kbeg + kt + a_k[c]) * BF16B; r[2*c] = buf_load_b64(Ar, o, 0); r[2*c+1] = buf_load_b64(Ar, o + KOCT, 0); }
    };
    auto loadB = [&](int kt, bf16x4* r){
        #pragma unroll
        for(int c = 0; c < B_CHUNKS; c++) if(b_valid[c]){ int o = ((n0 + b_row[c]) * K + kbeg + kt + b_k[c]) * BF16B; r[2*c] = buf_load_b64(Br, o, 0); r[2*c+1] = buf_load_b64(Br, o + KOCT, 0); }
    };
    // the two bf16x4 sit at adjacent swizzled LDS addrs -> the compiler merges them to one ds_write_b128
    auto storeA = [&](int buf, const bf16x4* r){
        #pragma unroll
        for(int c = 0; c < A_CHUNKS; c++){
            *reinterpret_cast<bf16x4*>(&s_A[buf][phys(a_row[c], a_k[c])])     = r[2*c];
            *reinterpret_cast<bf16x4*>(&s_A[buf][phys(a_row[c], a_k[c] + 4)]) = r[2*c+1];
        }
    };
    auto storeB = [&](int buf, const bf16x4* r){
        #pragma unroll
        for(int c = 0; c < B_CHUNKS; c++) if(b_valid[c]){
            *reinterpret_cast<bf16x4*>(&s_B[buf][phys(b_row[c], b_k[c])])     = r[2*c];
            *reinterpret_cast<bf16x4*>(&s_B[buf][phys(b_row[c], b_k[c] + 4)]) = r[2*c+1];
        }
    };
    auto compute = [&](int buf){
        #pragma unroll
        for(int kk = 0; kk < KK; kk++){ int k = kk * MFMA_K + g * KOCT;   // lane (g,e)'s 8-bf16 K-octet
            bf16x8 b = *reinterpret_cast<bf16x8*>(&s_B[buf][phys(nchunk * MFMA_N + e, k)]);
            #pragma unroll
            for(int mc = 0; mc < MCH_W; mc++){
                bf16x8 a = *reinterpret_cast<bf16x8*>(&s_A[buf][phys((mbase + mc) * MFMA_M + e, k)]);
                acc[mc] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[mc], /*cbsz,abid,blgp=*/0, 0, 0);
            }
        }
    };

    // Main loop: double-buffered LDS, one tile of K-prefetch ahead of compute.
    constexpr int AR = 2 * A_CHUNKS, BR = 2 * B_CHUNKS;
    bf16x4 a_reg[AR], a_next[AR], b_reg[BR], b_next[BR];
    loadA(0, a_reg); loadB(0, b_reg); storeA(0, a_reg); storeB(0, b_reg); __builtin_amdgcn_s_waitcnt(0); __syncthreads();
    for(int t = 0; t < ntiles; t++){ int buf = t & 1;
        if(t + 1 < ntiles){ loadA((t + 1) * BK, a_next); loadB((t + 1) * BK, b_next); }
        compute(buf);
        if(t + 1 < ntiles){ storeA(buf ^ 1, a_next); storeB(buf ^ 1, b_next); __syncthreads();
            #pragma unroll
            for(int c = 0; c < AR; c++) a_reg[c] = a_next[c];
            #pragma unroll
            for(int c = 0; c < BR; c++) b_reg[c] = b_next[c];
        }
    }

    // ---- Epilogue: cshuffle fp32 acc (4 rows/lane) through LDS -> packed bf16 atomic (2 cols/lane).
    //   1. write each acc lane to a plain [BM][BN] fp32 image in LDS (overlays s_A; 8KB <= s_A's 32KB)
    //   2. re-read it as adjacent column-pairs and packed-atomic-add into bf16 C.
    float* lds_f = reinterpret_cast<float*>(&s_A[0][0]);
    __syncthreads();                                   // all lanes done reading s_A in compute()
    #pragma unroll
    for(int mc = 0; mc < MCH_W; mc++)
        #pragma unroll
        for(int v = 0; v < ACC_ROWS; v++){
            int c_row = (mbase + mc) * MFMA_M + g * ACC_ROWS + v;   // C row this lane owns
            int c_col = nchunk * MFMA_N + e;                        // C column this lane owns
            lds_f[c_row * BN + c_col] = acc[mc][v];
        }
    __syncthreads();
    // gate the global atomics on the zero-init being visible (other splits wait on sema)
    if constexpr(ZERO_INIT) if(ksplit != 0){
        if(tid == 0) while(__hip_atomic_load(&sema[n_tile], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != epoch){}
        __syncthreads();
    }
    // BM*BN fp32 = BM*BN/2 column-pairs; 256 threads each handle (BM*BN/2)/256 pairs.
    #pragma unroll
    for(int p4 = 0; p4 < (BM * BN / 2) / 256; p4++){
        int p = tid + p4 * 256, row = p / (BN / 2), c0 = (p % (BN / 2)) * 2;   // c0 = even column in slab
        bhalf2_t v{ (__bf16)lds_f[row * BN + c0], (__bf16)lds_f[row * BN + c0 + 1] };
        int global_m = mrow0 + row;
        if(global_m < M)
            atomic_pk_add_bf16(&C[global_m * N + n0 + c0], v);
    }
}

// Host launcher. Compiles BOTH zero_init instantiations; picks at the call site (host branch, no GPU
// cost). BM is fixed to M (M<=64 fully covered by one tile). sema/epoch as documented on the kernel.
template<int M, int N, int K, int BN, int BK, int SPLITK, int BM = 64>
inline void launch(const bf16* A, const bf16* B, bf16* C, int* sema, int epoch,
                   bool zero_init, hipStream_t stream){
    // grid.z = ceil(M/BM) tiles the M dimension (each block owns BM rows at mrow0=blockIdx.z*BM).
    // NB: the per-M-tile blocks each read their B (weight) slice, so weight traffic scales with
    // ceil(M/BM) — fine for M<=64 (1 tile) and the concurrent tiles share it via L2/LLC.
    dim3 grid(N / BN, SPLITK, (M + BM - 1) / BM), block(256);
    if(zero_init)
        bf16gemm_mfma32_splitk_pk<M, N, K, BM, BN, BK, SPLITK, true >
            <<<grid, block, 0, stream>>>(A, B, C, sema, epoch);
    else
        bf16gemm_mfma32_splitk_pk<M, N, K, BM, BN, BK, SPLITK, false>
            <<<grid, block, 0, stream>>>(A, B, C, sema, epoch);
}

}  // namespace aiter::prezero_gemm
