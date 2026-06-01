# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Micro-benchmark for the standalone MLA-decode reduce kernel
# (aiter.ops.mla_decode_reduce.mla_decode_reduce — csrc/kernels/mla/
# decode_reduce/aux/mla_reduce.cuh), isolating JUST the LSE-weighted
# split-reduce (no stage1, no W^V GEMM), and comparing it head-to-head against
# the STOCK reduce that aiter.mla_decode_fwd uses in plain TP=4 (dp<=4 =>
# persistent path): aiter.mla_reduce_v1 (csrc/kernels/mla/reduce.cu), driven
# through the real persistent planner (aiter.get_mla_metadata_v1).
#
# Why this is the right baseline: in plain TP=4 decode mla_decode_fwd runs the
# PERSISTENT path, whose reduce is mla_reduce_v1 (NOT the non-persistent triton
# stage2). The persistent planner adapts the number of KV-splits per token to
# concurrency (~cu_num/T), so total partial tiles stays ~cu_num regardless of T.
# Our kernel always uses N_SPLITS=16 splits/token, so it reduces T*16 tiles —
# which at high T is many× more HBM traffic than stock's ~cu_num tiles. This
# bench makes that gap explicit (the `tiles` column) alongside wall-clock us.
#
# Mapping to serving (plain TP=4, 1 token/step, no attn-DP): T == decode
# concurrency. So --tokens 32,64,128 == conc 32/64/128. (`batch` is NOT conc —
# it's our kernel's BATCH prefetch-depth template knob; we sweep 2/4/8.)
#
# Shapes hardcoded to the only instantiated config (H=16 MLA TP=4 decode):
#   H=16, K=kv_lora_rank=512, N_SPLITS=16.
#
# Usage:
#   python op_tests/bench_mla_reduce_micro.py
#   python op_tests/bench_mla_reduce_micro.py --tokens 32,64,128 --batches 2,4,8
#   python op_tests/bench_mla_reduce_micro.py --max-split-cap 16   # stock split cap

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.ops.mla_decode_reduce import mla_decode_reduce
from aiter.mla_decode_reduce import _adaptive_num_kv_splits
from aiter.test_common import run_perftest, checkAllclose

# H=16 MLA TP=4 decode shape — must match the csrc instantiation.
H = 16
K = 512  # kv_lora_rank
N_SPLITS = 16
MGC = 64  # stage1 keys-per-split granularity; my_valid = min(ceil(kv_len/MGC), N_SPLITS)


# ── ours: mla_decode_reduce on [T*N_SPLITS, H, K] partials ───────────────────

def build_ours(T, kv_len, device, seed=0, splits=N_SPLITS):
    g = torch.Generator(device=device).manual_seed(seed)
    partial_output = torch.randn(
        (T * splits, H, K), dtype=torch.float32, device=device, generator=g
    )
    partial_lse = torch.randn(
        (T * splits, H), dtype=torch.float32, device=device, generator=g
    )
    reduced = torch.empty((T, H, K), dtype=torch.bfloat16, device=device)
    kv_indptr = torch.arange(
        0, (T + 1) * kv_len, kv_len, dtype=torch.int32, device=device
    )
    return partial_output, partial_lse, reduced, kv_indptr


def ref_ours(partial_output, partial_lse, kv_indptr, T):
    """torch reference for our kernel: per (t,h), softmax over valid splits."""
    po = partial_output.view(T, N_SPLITS, H, K)
    lse = partial_lse.view(T, N_SPLITS, H)
    kv_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.float32)
    my_valid = torch.clamp(torch.ceil(kv_len / MGC).to(torch.int64), max=N_SPLITS)
    out = torch.empty((T, H, K), dtype=torch.bfloat16, device=po.device)
    for t in range(T):
        v = int(my_valid[t].item())
        w = torch.softmax(lse[t, :v], dim=0)
        out[t] = (w[:, :, None] * po[t, :v]).sum(0).to(torch.bfloat16)
    return out


# ── stock: aiter.mla_reduce_v1 with real persistent-planner metadata ─────────

def build_stock(T, kv_len, device, max_split, seed=0):
    """Plan a uniform TP=4 decode (T tokens, page_size=1, kv_len pages each),
    then synthesize random partials sized to the planner's partial buffer. No
    real stage1 needed — mla_reduce_v1 only consumes the partials + metadata."""
    nhead, nhead_kv, page_size = H, 1, 1
    qo_indptr = torch.arange(0, T + 1, dtype=torch.int32, device=device)
    kv_block_nums = torch.full((T,), kv_len, dtype=torch.int32)  # page_size=1
    kv_indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, 0).to(device)
    kv_last_page_lens = torch.ones(T, dtype=torch.int32, device=device)
    max_seqlen_qo = 1
    dtype = dtypes.bf16

    (
        (wmd_s, wmd_t), (wi_s, wi_t), (wis_s, wis_t),
        (ri_s, ri_t), (rfm_s, rfm_t), (rpm_s, rpm_t),
    ) = aiter.get_mla_metadata_info_v1(
        T, max_seqlen_qo, nhead, dtype, dtype, is_sparse=False,
        fast_mode=True, num_kv_splits=max_split, intra_batch_mode=False,
    )
    work_meta_data = torch.empty(wmd_s, dtype=wmd_t, device=device)
    work_indptr = torch.empty(wi_s, dtype=wi_t, device=device)
    work_info_set = torch.empty(wis_s, dtype=wis_t, device=device)
    reduce_indptr = torch.empty(ri_s, dtype=ri_t, device=device)
    reduce_final_map = torch.empty(rfm_s, dtype=rfm_t, device=device)
    reduce_partial_map = torch.empty(rpm_s, dtype=rpm_t, device=device)

    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens, nhead // nhead_kv, nhead_kv,
        False, work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=page_size, kv_granularity=max(page_size, 16),
        max_seqlen_qo=max_seqlen_qo, uni_seqlen_qo=1, fast_mode=True,
        max_split_per_batch=max_split, intra_batch_mode=False,
        dtype_q=dtype, dtype_kv=dtype,
    )

    P = reduce_partial_map.shape[0] * max_seqlen_qo
    g = torch.Generator(device=device).manual_seed(seed)
    partial_output = torch.randn((P, 1, nhead, K), dtype=torch.float32,
                                 device=device, generator=g)
    partial_lse = torch.randn((P, 1, nhead, 1), dtype=torch.float32,
                              device=device, generator=g)
    o = torch.empty((T, nhead, K), dtype=torch.bfloat16, device=device)
    return {
        "args": (partial_output, partial_lse, reduce_indptr, reduce_final_map,
                 reduce_partial_map, max_seqlen_qo, o, None),
        "tiles": P,
    }


def bytes_reduce(num_tiles, T):
    """HBM traffic: read `num_tiles` fp32 partial tiles (+ their LSE), write the
    [T,H,K] bf16 output."""
    return num_tiles * H * K * 4 + num_tiles * H * 4 + T * H * K * 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="32,64,128",
                    help="decode token counts T == conc (default 32,64,128)")
    ap.add_argument("--batches", default="2,4,8",
                    help="our kernel's BATCH prefetch depths (default 2,4,8)")
    ap.add_argument("--vecs", default="1,2,4",
                    help="our kernel's VEC load-width/work-per-CTA knob "
                         "(1/2/4 fp32 per thread → b32/b64/b128; default 1,2,4)")
    ap.add_argument("--kv-len", type=int, default=2048,
                    help="per-token context length (pages, page_size=1); >=1024 "
                         "=> our kernel's 16 splits all valid. Default 2048.")
    ap.add_argument("--max-split-cap", type=int, default=N_SPLITS,
                    help="upper bound on stock splits/token; actual = "
                         "min(ceil(cu_num/T), cap). Default 16 (= our N_SPLITS).")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rotate", type=int, default=4,
                    help="num_rotate_args: rotate over N input copies so partials "
                         "aren't served from L2/MALL cache. Default 4. 0 disables.")
    ap.add_argument("--peak-bw", type=float, default=None,
                    help="peak HBM GB/s for a %%peak column (8000 MI355X, 5300 MI300X)")
    ap.add_argument("--no-check", dest="check", action="store_false",
                    help="skip the torch correctness check (ours)")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = torch.device("cuda")
    tokens = [int(x) for x in a.tokens.split(",") if x]
    batches = [int(x) for x in a.batches.split(",") if x]
    vecs = [int(x) for x in a.vecs.split(",") if x]
    my_valid = min(-(-a.kv_len // MGC), N_SPLITS)
    cu_num = torch.cuda.get_device_properties(device).multi_processor_count

    print(
        f"MLA reduce micro-bench: ours (mla_decode_reduce, fixed {N_SPLITS} "
        f"splits/tok) vs stock (mla_reduce_v1, persistent planner)\n"
        f"  H={H} K={K} kv_len={a.kv_len} cu_num={cu_num} T==decode-conc (plain TP4)"
        f"  iters={a.iters} rotate={a.rotate}"
    )

    if a.check:
        T0 = tokens[0]
        po, lse, red, kvp = build_ours(T0, a.kv_len, device)
        ref = ref_ours(po, lse, kvp, T0)
        for vec in vecs:
            red.zero_()
            mla_decode_reduce(po, lse, red, kvp, T0, batches[0], vec)
            torch.cuda.synchronize()
            checkAllclose(ref.float(), red.float(), rtol=2e-2, atol=2e-2,
                          msg=f"ours correctness T={T0} batch={batches[0]} vec={vec}")

    hdr = f"{'T':>5} {'kernel':>14} {'tiles':>7} {'us':>9} {'GB/s':>8} {'vs_stock':>9}"
    if a.peak_bw:
        hdr += f" {'%peak':>7}"
    print(hdr)
    print("-" * len(hdr))

    def row(T, name, tiles, us, ratio):
        gbps = bytes_reduce(tiles, T) / (us * 1e-6) / 1e9
        s = f"{T:>5} {name:>14} {tiles:>7} {us:>9.2f} {gbps:>8.1f} {ratio:>8.2f}x"
        if a.peak_bw:
            s += f" {100.0 * gbps / a.peak_bw:>6.1f}%"
        return s

    for T in tokens:
        max_split = min(-(-cu_num // T), a.max_split_cap)  # ceil(cu_num/T), capped
        stock = build_stock(T, a.kv_len, device, max_split)
        _, us_stock = run_perftest(
            aiter.mla_reduce_v1, *stock["args"],
            num_iters=a.iters, num_warmup=a.warmup, num_rotate_args=a.rotate,
        )
        print(row(T, f"stock(S<={max_split})", stock["tiles"], us_stock, 1.0))

        po, lse, red, kvp = build_ours(T, a.kv_len, device)
        for vec in vecs:
            for batch in batches:
                _, us = run_perftest(
                    mla_decode_reduce, po, lse, red, kvp, T, batch, vec,
                    num_iters=a.iters, num_warmup=a.warmup,
                    num_rotate_args=a.rotate,
                )
                print(row(T, f"ours v{vec}b{batch}", T * my_valid, us,
                          us_stock / us))
        # ── adaptive split count (matches stock's persistent planner) ────────
        S = _adaptive_num_kv_splits(T, cu_num)
        valid_S = min(-(-a.kv_len // MGC), S)
        po_a, lse_a, red_a, kvp_a = build_ours(T, a.kv_len, device, splits=S)
        eff_b = min(8, S)
        _, us_a = run_perftest(
            mla_decode_reduce, po_a, lse_a, red_a, kvp_a, T, eff_b, 4, S,
            num_iters=a.iters, num_warmup=a.warmup, num_rotate_args=a.rotate,
        )
        print(row(T, f"ours adapt(S={S})", T * valid_S, us_a, us_stock / us_a))
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
