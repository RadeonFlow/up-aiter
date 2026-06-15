// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch host wrapper + pybind for the hand-written bf16 split-K GEMM (splitk_gemm_a16w16.cuh),
// PREZERO variant: C is assumed already zeroed (by the upstream mla_*_rmsnorm GZ prezero), so the
// GEMM runs zero_init=false — pure packed-bf16 atomic-add, no in-kernel zeroing / no sema.
//   C[M,N] = A[M,K] @ B[N,K]^T   (TN, bf16 in / fp32 accumulate / bf16 out)
// Wired shapes (Kimi-K2.5 MLA, decode conc M=64):
//   op2 qkv_a : N=2112, K=7168     op4 q_b : N=3072, K=1536

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "splitk_gemm_a16w16.cuh"

namespace py = pybind11;
using aiter::prezero_gemm::bf16;

namespace aiter {

// C must be pre-zeroed (prezero co-design). zero_init=false => no sema needed.
void splitk_gemm_with_prezero(torch::Tensor& C, torch::Tensor& A, torch::Tensor& B) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "splitk_gemm_with_prezero: A[M,K] @ B[N,K]^T, K mismatch");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must be [M,N]");

    auto* a = reinterpret_cast<const bf16*>(A.data_ptr());
    auto* b = reinterpret_cast<const bf16*>(B.data_ptr());
    auto* c = reinterpret_cast<bf16*>(C.data_ptr());
    using namespace aiter::prezero_gemm;
    // M is a compile-time template (A/C buffer bounds + M-tiling). Precompiled buckets {4..256}; caller
    // pads the decode batch to the nearest bucket. (BM,BN,SPLITK) tuned PER (shape,M) on MI355X (256 CU)
    // — VALIDATED sweep /tmp/tune_{gemm,bm}*.hip (each config checked vs a reference). Lessons:
    //   - BK MUST be 128: the rot021 swizzle assumes 16 k-blocks; BK=64 is fast-but-NUMERICALLY-WRONG.
    //   - best BM = the SMALLEST BM that still covers M in ONE M-tile (grid.z=1): smaller BM -> smaller
    //     s_A LDS -> higher occupancy -> better hides the weight read (M=4 qkv_a BM16 4.9us vs BM64 7.4
    //     = -34%). Below that (extra M-tiles) re-reads the weight; above (BM128) drops occupancy. So
    //     M<=16 -> BM16, M=32 -> BM32, M>=64 -> BM64. (BM16 needs BN64; BM32 ok with BN32/BN64.)
    //   - SPLITK targets ~512 blocks = (N/BN)*SK*ceil(M/BM) on 256 CUs.
    // rocprof us (M4/16/32/64/128/256): qkv_a ~4.9/5.1/6.0/7.0/13.8/24.2 ; q_b ~2.9/2.9/3.2/4.0/4.6/7.9.
    if (N == 2112 && K == 7168) {            // qkv_a (N=2112, K=7168)   <M, N, K, BN, BK, SPLITK, BM>
        switch (M) {
            case 4:   launch<4,   2112, 7168, 64, 128, 14, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 8:   launch<8,   2112, 7168, 64, 128, 14, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 16:  launch<16,  2112, 7168, 64, 128, 14, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 32:  launch<32,  2112, 7168, 64, 128, 14, 32>(a, b, c, nullptr, 0, false, stream); break;
            case 64:  launch<64,  2112, 7168, 32, 128,  7, 64>(a, b, c, nullptr, 0, false, stream); break;
            case 128: launch<128, 2112, 7168, 32, 128,  8, 64>(a, b, c, nullptr, 0, false, stream); break;
            case 256: launch<256, 2112, 7168, 32, 128,  8, 64>(a, b, c, nullptr, 0, false, stream); break;
            default: TORCH_CHECK(false, "M must be in {4,8,16,32,64,128,256}, got ", M);
        }
    } else if (N == 3072 && K == 1536) {     // q_b (N=3072, K=1536)     <M, N, K, BN, BK, SPLITK, BM>
        switch (M) {
            case 4:   launch<4,   3072, 1536, 64, 128,  6, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 8:   launch<8,   3072, 1536, 64, 128,  6, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 16:  launch<16,  3072, 1536, 64, 128,  6, 16>(a, b, c, nullptr, 0, false, stream); break;
            case 32:  launch<32,  3072, 1536, 32, 128,  6, 32>(a, b, c, nullptr, 0, false, stream); break;
            case 64:  launch<64,  3072, 1536, 32, 128,  4, 64>(a, b, c, nullptr, 0, false, stream); break;
            case 128: launch<128, 3072, 1536, 32, 128,  4, 64>(a, b, c, nullptr, 0, false, stream); break;
            case 256: launch<256, 3072, 1536, 32, 128,  2, 64>(a, b, c, nullptr, 0, false, stream); break;
            default: TORCH_CHECK(false, "M must be in {4,8,16,32,64,128,256}, got ", M);
        }
    } else if (N == 384 && K == 7168) {      // MoE router/gate (N=384, K=7168)  <M,N,K,BN,BK,SPLITK,BM>
        // Thin N (384). SK must divide 56 (=K/128). SWEEP (_nz_test/sweep_router_cu_fill.py, MI355X 256 CU):
        // this GEMM hits its latency FLOOR (~4.6us) at only ~84-96 blocks (~35% occ) — it's mem/latency
        // bound, NOT occupancy bound. Pushing SK to occ=100% does NOT help and only grows bf16 atomic-add
        // error (more splits contend on the tiny C). So pick the LOWEST SK that crosses the floor (lowest
        // error). amortized hot us @ MI355X: SK=8 is the sweet spot (8 | 56, zero extra error vs SK=7).
        switch (M) {
            case 32:  launch<32,  384, 7168, 64, 128,  8, 32>(a, b, c, nullptr, 0, false, stream); break;  // 48 blk, ~4.63us
            case 64:  launch<64,  384, 7168, 32, 128,  8, 64>(a, b, c, nullptr, 0, false, stream); break;  // 96 blk, ~4.61us (was BN64/SK7=6.69us, -31%)
            case 128: launch<128, 384, 7168, 32, 128,  8, 64>(a, b, c, nullptr, 0, false, stream); break;  // 192 blk, ~4.76us
            case 256: launch<256, 384, 7168, 32, 128,  4, 64>(a, b, c, nullptr, 0, false, stream); break;  // (not in M<=128 sweep)
            default: TORCH_CHECK(false, "router: M must be in {32,64,128,256}, got ", M);
        }
    } else {
        TORCH_CHECK(false, "splitk_gemm_with_prezero: unsupported (N,K)=(", N, ",", K, ")");
    }
}

// ---------------------------------------------------------------------------
// CSV-tuned variant: tile params (BN, SPLITK, BM) come from the prezero tuning CSV
// (aiter/configs/a16w16_prezero_tuned_gemm.csv) via the python tgemm_prezero dispatcher, instead
// of the hardcoded per-(shape,M) switch above. (M,N,K) are still compile-time (buffer bounds
// + tiling), so this maps the runtime (M,N,K,BN,SPLITK,BM) tuple to a PRECOMPILED launch<>
// instance. The instance set below = exactly the rows seeded in the CSV (qkv_a / q_b / router
// x M buckets). BK is fixed 128 (rot021 swizzle assumes 16 k-blocks). C must be pre-zeroed.
// To tune a brand-new (BN,SPLITK,BM) combo, add the matching INST() line here (and the CSV row).
void splitk_gemm_prezero_tuned(torch::Tensor& C, torch::Tensor& A, torch::Tensor& B,
                               int BN, int SPLITK, int BM) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "splitk_gemm_prezero_tuned: A[M,K] @ B[N,K]^T, K mismatch");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must be [M,N]");
    auto* a = reinterpret_cast<const bf16*>(A.data_ptr());
    auto* b = reinterpret_cast<const bf16*>(B.data_ptr());
    auto* c = reinterpret_cast<bf16*>(C.data_ptr());
    using namespace aiter::prezero_gemm;
#define INST(MM, NN, KK, BNV, SKV, BMV) \
    if (M == MM && N == NN && K == KK && BN == BNV && SPLITK == SKV && BM == BMV) { \
        launch<MM, NN, KK, BNV, 128, SKV, BMV>(a, b, c, nullptr, 0, false, stream); return; }
    // qkv_a (N=2112, K=7168)
    INST(4,   2112, 7168, 64, 14, 16) INST(8,   2112, 7168, 64, 14, 16)
    INST(16,  2112, 7168, 64, 14, 16) INST(32,  2112, 7168, 64, 14, 32)
    INST(64,  2112, 7168, 32,  7, 64) INST(128, 2112, 7168, 32,  8, 64)
    INST(256, 2112, 7168, 32,  8, 64)
    // q_b (N=3072, K=1536)
    INST(4,   3072, 1536, 64,  6, 16) INST(8,   3072, 1536, 64,  6, 16)
    INST(16,  3072, 1536, 64,  6, 16) INST(32,  3072, 1536, 32,  6, 32)
    INST(64,  3072, 1536, 32,  4, 64) INST(128, 3072, 1536, 32,  4, 64)
    INST(256, 3072, 1536, 32,  2, 64)
    // router / MoE gate (N=384, K=7168)
    INST(32,  384,  7168, 64,  8, 32) INST(64,  384,  7168, 32,  8, 64)
    INST(128, 384,  7168, 32,  8, 64) INST(256, 384,  7168, 32,  4, 64)
#undef INST
    TORCH_CHECK(false, "splitk_gemm_prezero_tuned: no precompiled instance for (M,N,K,BN,SPLITK,BM)=(",
                M, ",", N, ",", K, ",", BN, ",", SPLITK, ",", BM, ")");
}

// ---------------------------------------------------------------------------
// BENCH-ONLY: runtime (BN, SPLITK) sweep for the router shape (N=384, K=7168) to
// find the (BN,SPLITK) that best fills the 256 CUs. zero_init=false (C pre-zeroed
// by caller). BK is fixed 128 (rot021). SPLITK must divide 56 (=K/128). BN in
// {16,32,64}; BN=16 needs BM=64 (MCH_W integrality). blocks=(384/BN)*SPLITK*ceil(M/BM).
void splitk_gemm_bench(torch::Tensor& C, torch::Tensor& A, torch::Tensor& B, int BN, int SPLITK) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(N == 384 && K == 7168, "bench: router shape only");
    auto* a = reinterpret_cast<const bf16*>(A.data_ptr());
    auto* b = reinterpret_cast<const bf16*>(B.data_ptr());
    auto* c = reinterpret_cast<bf16*>(C.data_ptr());
    using namespace aiter::prezero_gemm;
#define TRY(MM, BMV, BNV, SK) \
    if (M == MM && BN == BNV && SPLITK == SK) { \
        launch<MM, 384, 7168, BNV, 128, SK, BMV>(a, b, c, nullptr, 0, false, stream); return; }
#define TRY_BN_SK(MM, BMV, BNV) \
    TRY(MM, BMV, BNV, 4) TRY(MM, BMV, BNV, 7) TRY(MM, BMV, BNV, 8) \
    TRY(MM, BMV, BNV, 14) TRY(MM, BMV, BNV, 28)
    // M=32 (BM=32): BN in {32,64}   (BN=16 invalid w/ BM=32)
    TRY_BN_SK(32, 32, 32) TRY_BN_SK(32, 32, 64)
    // M=64 (BM=64): BN in {16,32,64}
    TRY_BN_SK(64, 64, 16) TRY_BN_SK(64, 64, 32) TRY_BN_SK(64, 64, 64)
    // M=128 (BM=64, grid.z=2): BN in {16,32,64}
    TRY_BN_SK(128, 64, 16) TRY_BN_SK(128, 64, 32) TRY_BN_SK(128, 64, 64)
#undef TRY_BN_SK
#undef TRY
    TORCH_CHECK(false, "bench: unsupported (M,BN,SPLITK)=(", M, ",", BN, ",", SPLITK, ")");
}

}  // namespace aiter

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splitk_gemm_with_prezero", &aiter::splitk_gemm_with_prezero,
          py::arg("C"), py::arg("A"), py::arg("B"));
    m.def("splitk_gemm_prezero_tuned", &aiter::splitk_gemm_prezero_tuned,
          py::arg("C"), py::arg("A"), py::arg("B"),
          py::arg("BN"), py::arg("SPLITK"), py::arg("BM"));
    m.def("splitk_gemm_bench", &aiter::splitk_gemm_bench,
          py::arg("C"), py::arg("A"), py::arg("B"), py::arg("BN"), py::arg("SPLITK"));
}
