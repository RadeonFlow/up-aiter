// mla_reduce.cuh — MLA decode reduce kernel template.
//
// LSE-weighted reduce across N_SPLITS partial logits (fp32) → bf16 reduced
// activation. Drop-in replacement for aiter.mla_reduce_v1's reduce on the
// plain-TP decode path; output dtype matches so the downstream V up-proj +
// o_proj tail is unchanged.
//
// Task distribution:
//   Grid = T × H × (K / BD)         — flood the GPU with small CTAs
//   BD   = 128 elements of D_V (kv_lora_rank) per CTA
//   K    = 512 → 4 D_V-slice CTAs per (token, head)
//   At T=16: 16 × 16 × 4 = 1024 CTAs (≫ 256 CU; HBM latency hidden by OCC)
//
// Per CTA: 128 threads (2 waves), OCC≥8 hint, ~8 KB HBM read (16 splits ×
// 128 fp32). Each thread covers 1 D_V element across all 16 splits.
//
// Pipeline:
//   1. Each thread loads its 16 LSE values for (t, h, splits 0..15)
//   2. Compute softmax weights w[16] (uniform across threads — but recomputed
//      per thread; the alternative LDS-broadcast costs more than the compute)
//   3. Each thread loads 16 fp32 partials for (t, h, splits, d), reduces
//      acc = sum w[s] * partial[s]
//   4. Write bf16 to reduced[t, h, d]
//
// Shapes (the instantiated decode config, hardcoded):
//   partial_output [T*16, H=16, K=512] fp32
//   partial_lse    [T*16, H=16]        fp32
//   reduced        [T, H=16, K=512]    bf16
//   kv_indptr      [T+1]               int32  (my_valid computed inline as
//                                              min(ceil((indptr[t+1]-indptr[t])/64), 16))

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define DEVICE_INLINE __device__ __forceinline__

namespace mla_reduce_ns {

using buffer_rsrc_t = __amdgpu_buffer_rsrc_t;
using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using u16x2 = uint16_t __attribute__((ext_vector_type(2)));
using u16x4 = uint16_t __attribute__((ext_vector_type(4)));

DEVICE_INLINE uint16_t bf16_bits(float x) {
    return __builtin_bit_cast(uint16_t, (__hip_bfloat16)x);
}

DEVICE_INLINE buffer_rsrc_t make_buffer_rsrc(const void* base, uint32_t num_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), (short)0, (int)num_bytes, /*flags=*/0x00020000);
}

// Load VEC contiguous fp32 from a buffer rsrc as one VEC*32-bit transaction.
// VEC ∈ {1,2,4} → b32 / b64 / b128. Returned as raw int bits (free bitcast).
template <int VEC>
DEVICE_INLINE void load_vec(int (&bits)[VEC], buffer_rsrc_t rsrc,
                            int voffset, int soffset) {
    if constexpr (VEC == 1) {
        bits[0] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voffset, soffset, 0);
    } else if constexpr (VEC == 2) {
        const i32x2 v = __builtin_bit_cast(
            i32x2, __builtin_amdgcn_raw_buffer_load_b64(rsrc, voffset, soffset, 0));
        bits[0] = v[0]; bits[1] = v[1];
    } else {
        const i32x4 v = __builtin_bit_cast(
            i32x4, __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, soffset, 0));
        bits[0] = v[0]; bits[1] = v[1]; bits[2] = v[2]; bits[3] = v[3];
    }
}

template <int H, int K, int N_SPLITS, int BATCH, int VEC>
__global__ void __launch_bounds__(128, 8)
mla_reduce_kernel(
    const float*          __restrict__ partial_output,  // [T*16, 16, 512] fp32
    const float*          __restrict__ partial_lse,     // [T*16, 16]      fp32
    __hip_bfloat16*       __restrict__ reduced,         // [T, 16, 512]    bf16
    const int*            __restrict__ kv_indptr,       // [T+1]           int32
    int T)
{
    constexpr int BD       = 128 * VEC;
    constexpr int D_TILES  = K / BD;
    static_assert(K % BD == 0, "K must be a multiple of BD=128*VEC");
    static_assert(N_SPLITS % BATCH == 0, "N_SPLITS must be divisible by BATCH");
    static_assert(VEC == 1 || VEC == 2 || VEC == 4, "VEC must be 1, 2, or 4");

    const int block_idx = blockIdx.x;
    // Layout: block_idx = (token_idx * H + head_idx) * D_TILES + d_tile_idx.
    // d_tile_idx innermost so adjacent CTAs share (token, head) → consecutive
    // stripes of the same row, good for cache-line streaming if any CTAs land
    // on the same CU.
    const int d_tile_idx     = block_idx % D_TILES;
    const int token_head_idx = block_idx / D_TILES;
    const int head_idx       = token_head_idx % H;
    const int token_idx      = token_head_idx / H;
    const int thread_idx     = threadIdx.x;
    const int d_base         = d_tile_idx * BD + thread_idx * VEC;  // first of this thread's VEC elems

    // ── my_valid: computed inline from kv_indptr (was a tensor arg before;
    //    folding the 4-op host pipeline (sub/add/floor_div/clamp_max) into
    //    the kernel saves ~20μs per layer of graph-launch overhead).
    //    Stage1 splits each token's KV across N_SPLITS=16 buckets of MGC=64
    //    keys; my_valid = min(ceil(kv_len / 64), 16).
    int my_valid;
    {
        constexpr int MGC = 64;
        const buffer_rsrc_t rsrc = make_buffer_rsrc(
            kv_indptr, (uint32_t)((T + 1) * (int)sizeof(int)));
        const int s_t  = __builtin_amdgcn_raw_buffer_load_b32(
            rsrc,  token_idx      * (int)sizeof(int), 0, 0);
        const int s_t1 = __builtin_amdgcn_raw_buffer_load_b32(
            rsrc, (token_idx + 1) * (int)sizeof(int), 0, 0);
        const int kv_len = s_t1 - s_t;
        int v = (kv_len + (MGC - 1)) / MGC;
        if (v > N_SPLITS) v = N_SPLITS;
        my_valid = v;
    }

    // ── LSE: each thread loads 16 fp32 (same per-(t,h) data across threads;
    // L1D broadcasts to all 64 lanes of a wave per cache line). 16×16 = 256B
    // = 2 cache lines fetched per wave. ─────────────────────────────────
    const buffer_rsrc_t LSE_rsrc = make_buffer_rsrc(
        partial_lse, (uint32_t)(T * N_SPLITS * H * (int)sizeof(float)));
    const int lse_base = (token_idx * N_SPLITS) * H + head_idx;   // element offset
    float lse[N_SPLITS];
    float max_lse = -INFINITY;
    #pragma unroll
    for (int s = 0; s < N_SPLITS; s++) {
        const int off = (lse_base + s * H) * (int)sizeof(float);
        const int bits = __builtin_amdgcn_raw_buffer_load_b32(LSE_rsrc, off, 0, 0);
        float v = __int_as_float(bits);
        lse[s] = v;
        if (s < my_valid) max_lse = fmaxf(max_lse, v);
    }
    // Skip __expf for invalid splits — AMDGPU v_exp_f32 input range is
    // [-126, 127.5]; calling __expf(-1e30 - max_lse) is UB and intermittently
    // returns NaN, which then poisons sum → inv_sum → acc → output bf16 NaN.
    // Symptom: occasional `!!!!!` token cascade in production decode.
    float w[N_SPLITS];
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < N_SPLITS; s++) {
        w[s] = (s < my_valid) ? __expf(lse[s] - max_lse) : 0.0f;
        sum += w[s];
    }
    // Guard against my_valid=0 (kv_seq_len=0): no contributions → output 0.
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    #pragma unroll
    for (int s = 0; s < N_SPLITS; s++) w[s] *= inv_sum;
    const buffer_rsrc_t A_rsrc = make_buffer_rsrc(
        partial_output,
        (uint32_t)((size_t)T * N_SPLITS * H * K * sizeof(float)));
    const int row_byte_base = ((token_idx * N_SPLITS) * H * K + head_idx * K + d_tile_idx * BD)
                              * (int)sizeof(float);
    const int v_voff_base   = thread_idx * VEC * (int)sizeof(float);

    float acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; e++) acc[e] = 0.0f;
    // BATCH: issue this many split loads upfront, then drain + accumulate.
    // Hides HBM latency under arithmetic. Tuned per (H, K, T) — see CSV.
    #pragma unroll
    for (int b0 = 0; b0 < N_SPLITS; b0 += BATCH) {
        int bits[BATCH][VEC];
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const int s = b0 + b;
            // OOB-bump: invalid splits get pushed past rsrc → HW returns 0.
            const int oob = (s < my_valid) ? 0 : 0x40000000;
            const int v_voff = v_voff_base + s * (H * K * (int)sizeof(float)) + oob;
            load_vec<VEC>(bits[b], A_rsrc, v_voff, row_byte_base);
        }
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const int s = b0 + b;
            // Belt-and-suspenders: w[s]=0 already zeros invalid s, but if the
            // OOB-bumped load returned NaN bits (HW behavior depends on rsrc
            // flags), 0 * NaN = NaN would still poison acc. Skip explicitly.
            if (s < my_valid) {
                #pragma unroll
                for (int e = 0; e < VEC; e++)
                    acc[e] += w[s] * __int_as_float(bits[b][e]);
            }
        }
    }

    // ── Store VEC bf16 to reduced[token, head, d_base..+VEC) as one b(16*VEC) write ───
    const size_t out_idx = (size_t)token_idx * H * K + (size_t)head_idx * K + d_base;
    if constexpr (VEC == 1) {
        reduced[out_idx] = (__hip_bfloat16)acc[0];
    } else if constexpr (VEC == 2) {
        const u16x2 o = { bf16_bits(acc[0]), bf16_bits(acc[1]) };
        *reinterpret_cast<u16x2*>(&reduced[out_idx]) = o;
    } else {
        const u16x4 o = { bf16_bits(acc[0]), bf16_bits(acc[1]),
                          bf16_bits(acc[2]), bf16_bits(acc[3]) };
        *reinterpret_cast<u16x4*>(&reduced[out_idx]) = o;
    }
}

}  // namespace mla_reduce_ns
