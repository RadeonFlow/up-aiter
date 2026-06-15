// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch host wrappers + pybind for the hand-written MLA-decode bf16 rmsnorm kernels
// (mla_rmsnorm.cuh). Exposes the two ops that replace ATOM's _fuse_rmsnorm_quant on the
// Kimi-K2.5 MLA input stage:
//   mla_add_rmsnorm : op1  input residual-add + rmsnorm (N=7168), optional prezero of the
//                     qkv_a-GEMM output (GZ=2112) for the tiny_pre_zero_splitk_gemm co-design.
//   mla_qk_rmsnorm  : op3  q & k rmsnorm in one launch (QN=1536, KN=512), optional prezero of
//                     the q_b-GEMM output (GZ=3072). q_in/k_in may be column-slices of qkv_lora
//                     (runtime row strides honored).

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>
#include <optional>

namespace py = pybind11;

#include "mla_rmsnorm.cuh"

using aiter::prezero_gemm::bf16;

namespace aiter {

// op1 — input residual-add + rmsnorm; optional prezero of the qkv_a-GEMM output row [m*2112].
void mla_add_rmsnorm(torch::Tensor& out, torch::Tensor& residual_out, torch::Tensor& input,
                     torch::Tensor& residual_in, torch::Tensor& weight, double eps,
                     std::optional<torch::Tensor> gemm_zero) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int m = input.size(0);
    const int n = input.size(-1);
    // GZ = downstream split-K GEMM output row width to prezero (0 = no prezero, pure rmsnorm).
    const int GZ = gemm_zero.has_value() ? (int)gemm_zero->size(-1) : 0;

    auto* I  = reinterpret_cast<const bf16*>(input.data_ptr());
    auto* R  = reinterpret_cast<const bf16*>(residual_in.data_ptr());
    auto* W  = reinterpret_cast<const bf16*>(weight.data_ptr());
    auto* O  = reinterpret_cast<bf16*>(out.data_ptr());
    auto* RO = reinterpret_cast<bf16*>(residual_out.data_ptr());
    bf16* Z  = gemm_zero.has_value() ? reinterpret_cast<bf16*>(gemm_zero->data_ptr()) : nullptr;
    const int in_s = input.stride(0), rin_s = residual_in.stride(0),
              ro_s = residual_out.stride(0), o_s = out.stride(0);
    using namespace aiter::prezero_gemm;
    // (N, GZ) are compile-time (full unroll -> all-b128 loads). Runtime-dispatch over the PRECOMPILED
    // instance set below. To support a new (N,GZ): add an INST() line here AND the matching shape in
    // python's _PREZERO_ADD_SHAPES (aiter/ops/mla_decode.py); add_rmsnorm_prezero gates on that set
    // (pure shape check, no CSV) and falls back to native rmsnorm + torch.zeros otherwise.
#define INST(NV, GZV) \
    if (n == NV && GZ == GZV) { \
        launch_mla_add_rmsnorm<NV, GZV>(O, RO, I, R, W, eps, m, stream, Z, in_s, rin_s, ro_s, o_s); return; }
    INST(7168, 2112) INST(7168, 0)
#undef INST
    TORCH_CHECK(false, "mla_add_rmsnorm: no precompiled (N,GZ)=(", n, ",", GZ,
                "); add an INST() + CSV row, or call via add_rmsnorm_prezero for fallback");
}

// op3 — q & k rmsnorm (one 2D launch); optional prezero of the q_b-GEMM output row [m*3072].
void mla_qk_rmsnorm(torch::Tensor& q_out, torch::Tensor& k_out, torch::Tensor& q_in,
                    torch::Tensor& k_in, torch::Tensor& q_weight, torch::Tensor& k_weight,
                    double q_eps, double k_eps, std::optional<torch::Tensor> gemm_zero,
                    std::optional<torch::Tensor> k_pe_out) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q_in));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int m  = q_in.size(0);
    const int qn = q_in.size(-1), kn = k_in.size(-1);
    const int GZ = gemm_zero.has_value() ? (int)gemm_zero->size(-1) : 0;

    auto* QI = reinterpret_cast<const bf16*>(q_in.data_ptr());
    auto* KI = reinterpret_cast<const bf16*>(k_in.data_ptr());
    auto* QW = reinterpret_cast<const bf16*>(q_weight.data_ptr());
    auto* KW = reinterpret_cast<const bf16*>(k_weight.data_ptr());
    auto* QO = reinterpret_cast<bf16*>(q_out.data_ptr());
    auto* KO = reinterpret_cast<bf16*>(k_out.data_ptr());
    bf16* Z  = gemm_zero.has_value() ? reinterpret_cast<bf16*>(gemm_zero->data_ptr()) : nullptr;
    const int qi_s = q_in.stride(0), ki_s = k_in.stride(0),
              qo_s = q_out.stride(0), ko_s = k_out.stride(0);
    using namespace aiter::prezero_gemm;
    // optional k_pe free-rider: the kernel copies each row's `rope` rope-cols (k_in[row]+KN) into KPE.
    bf16* KPE = nullptr; int rope = 0;
    if (k_pe_out.has_value()) {
        KPE  = reinterpret_cast<bf16*>(k_pe_out->data_ptr());
        rope = (int)k_pe_out->size(-1);
    }
    // (QN, KN, GZ) compile-time; runtime-dispatch over the precompiled set below. New (QN,KN,GZ): add an
    // INST() here AND the shape in python's _PREZERO_QK_SHAPES; qk_rmsnorm_prezero gates on it + falls back.
#define INST(QNV, KNV, GZV) \
    if (qn == QNV && kn == KNV && GZ == GZV) { \
        launch_mla_qk_rmsnorm<QNV, KNV, GZV>(QO, KO, QI, KI, QW, KW, (float)q_eps, (float)k_eps, m, stream, Z, qi_s, ki_s, qo_s, ko_s, KPE, rope); return; }
    INST(1536, 512, 3072) INST(1536, 512, 0)
#undef INST
    TORCH_CHECK(false, "mla_qk_rmsnorm: no precompiled (QN,KN,GZ)=(", qn, ",", kn, ",", GZ,
                "); add an INST() + CSV row, or call via qk_rmsnorm_prezero for fallback");
}

}  // namespace aiter

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_add_rmsnorm", &aiter::mla_add_rmsnorm,
          py::arg("out"), py::arg("residual_out"), py::arg("input"), py::arg("residual_in"),
          py::arg("weight"), py::arg("eps"), py::arg("gemm_zero") = std::nullopt);
    m.def("mla_qk_rmsnorm", &aiter::mla_qk_rmsnorm,
          py::arg("q_out"), py::arg("k_out"), py::arg("q_in"), py::arg("k_in"),
          py::arg("q_weight"), py::arg("k_weight"), py::arg("q_eps"), py::arg("k_eps"),
          py::arg("gemm_zero") = std::nullopt, py::arg("k_pe_out") = std::nullopt);
}
