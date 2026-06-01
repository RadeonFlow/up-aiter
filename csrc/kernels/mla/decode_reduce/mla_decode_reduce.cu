// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include "mla.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/amd_detail/amd_hip_bf16.h>

#include "aux/mla_reduce.cuh"

namespace {

// MLA decode shape currently instantiated (num_heads, kv_lora_rank). Adding a
// new shape: extend the dispatch switches below and add a CSV row in
// aiter/configs/tuned_mla_decode_reduce.csv.
constexpr int MLA_DECODE_H = 16;
constexpr int MLA_DECODE_K = 512;

}  // namespace


void mla_decode_reduce(
    torch::Tensor& partial_output,
    torch::Tensor& partial_lse,
    torch::Tensor& reduced,
    torch::Tensor& kv_indptr,
    int64_t T,
    int64_t batch,
    int64_t vec,
    int64_t num_splits)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(partial_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // (H, K) come from the kernel instantiation. `num_splits` is the
    // (concurrency-adaptive) KV-split count — stage1 packs this many partial
    // tiles per token, so the reduce reads N_SPLITS*T tiles. `batch` is the
    // prefetch-depth knob, `vec` is the load-width / work-per-CTA knob (1/2/4
    // fp32 per thread along D_V).
    constexpr int H        = MLA_DECODE_H;
    constexpr int K        = MLA_DECODE_K;
    constexpr int THREADS  = 128;
    // BATCH must divide N_SPLITS; caller may pass a larger batch than there are
    // splits (e.g. CSV-tuned batch=8 with adaptive num_splits=2) → clamp.
    const int64_t eff_batch = (batch < num_splits) ? batch : num_splits;

    auto launch = [&](auto NS_TAG, auto BATCH_TAG, auto VEC_TAG) {
        constexpr int N_SPLITS = decltype(NS_TAG)::value;
        constexpr int BATCH    = decltype(BATCH_TAG)::value;
        constexpr int VEC      = decltype(VEC_TAG)::value;
        constexpr int BD      = THREADS * VEC;
        constexpr int D_TILES = K / BD;
        const int grid        = T * H * D_TILES;
        mla_reduce_ns::mla_reduce_kernel<H, K, N_SPLITS, BATCH, VEC>
            <<<dim3(grid), dim3(THREADS), 0, stream>>>(
            reinterpret_cast<const float*>(partial_output.data_ptr()),
            reinterpret_cast<const float*>(partial_lse.data_ptr()),
            reinterpret_cast<__hip_bfloat16*>(reduced.data_ptr()),
            reinterpret_cast<const int*>(kv_indptr.data_ptr()),
            static_cast<int>(T));
    };

    // Dispatch (num_splits, batch) → only emit batch <= num_splits combos, both
    // powers of two so BATCH always divides N_SPLITS.
    auto dispatch_ns_batch = [&](auto VEC_TAG) {
        auto bad = [&]() {
            TORCH_CHECK(false, "mla_decode_reduce: unsupported (num_splits=",
                        num_splits, ", batch=", eff_batch, ")");
        };
        switch (num_splits) {
            case 1:
                launch(std::integral_constant<int, 1>{},
                       std::integral_constant<int, 1>{}, VEC_TAG); break;
            case 2:
                switch (eff_batch) {
                    case 1: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{}, VEC_TAG); break;
                    case 2: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}, VEC_TAG); break;
                    default: bad();
                } break;
            case 4:
                switch (eff_batch) {
                    case 1: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}, VEC_TAG); break;
                    case 2: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}, VEC_TAG); break;
                    case 4: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}, VEC_TAG); break;
                    default: bad();
                } break;
            case 8:
                switch (eff_batch) {
                    case 1: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{}, VEC_TAG); break;
                    case 2: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{}, VEC_TAG); break;
                    case 4: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{}, VEC_TAG); break;
                    case 8: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{}, VEC_TAG); break;
                    default: bad();
                } break;
            case 16:
                switch (eff_batch) {
                    case 2: launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{}, VEC_TAG); break;
                    case 4: launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 4>{}, VEC_TAG); break;
                    case 8: launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 8>{}, VEC_TAG); break;
                    default: bad();
                } break;
            default:
                TORCH_CHECK(false, "mla_decode_reduce: unsupported num_splits=",
                            num_splits, " (supported: 1, 2, 4, 8, 16)");
        }
    };

    switch (vec) {
        case 1: dispatch_ns_batch(std::integral_constant<int, 1>{}); break;
        case 2: dispatch_ns_batch(std::integral_constant<int, 2>{}); break;
        case 4: dispatch_ns_batch(std::integral_constant<int, 4>{}); break;
        default:
            TORCH_CHECK(false, "mla_decode_reduce: unsupported vec=", vec,
                        " (supported: 1, 2, 4)");
    }
}
