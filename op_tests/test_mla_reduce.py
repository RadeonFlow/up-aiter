# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# The uniform-decode fast path in mla_reduce_v1 is compiled/validated for these
# archs (generic HIP + opus buffer intrinsics on gfx9).
SUPPORTED_GFX = ["gfx942", "gfx950"]

# Fast-path shape constraints (dispatch_mla_reduce_v1): 16 q-heads, 512 v-dim,
# bf16 output. Anything else falls back to the general kernel.
H = 16
DV = 512


def build_uniform_decode_meta(num_tiles, nsplits, device):
    """Synthesize the metadata a uniform decode (max_seqlen_q==1) produces, i.e.
    the exact layout the fast path assumes:
      - reduce_indptr: prefix sums, nsplits partials per tile -> [0, ns, 2*ns, ...]
      - reduce_final_map[t] = [t, t+1]      (output row == tile == q_start)
      - reduce_partial_map  = arange(P)     (pool rows are the identity range)
    These are invariants (A) and (B) documented in csrc/kernels/mla/reduce.cu.
    """
    P = num_tiles * nsplits
    reduce_indptr = torch.arange(
        0, (num_tiles + 1) * nsplits, nsplits, dtype=dtypes.i32, device=device
    )
    rows = torch.arange(num_tiles, dtype=dtypes.i32, device=device)
    reduce_final_map = torch.stack([rows, rows + 1], dim=-1).contiguous()
    reduce_partial_map = torch.arange(P, dtype=dtypes.i32, device=device)
    return P, reduce_indptr, reduce_final_map, reduce_partial_map


def run_torch(partial_output, partial_lse, reduce_indptr, seed_out):
    """Reference log-sum-exp reduce over each tile's partial splits. Reference
    only: fp32 math, cast back — not timed, not in the table. Tiles with < 2
    splits are left at their seed value, mirroring the kernel which skips them
    (stage1 already wrote those outputs)."""
    num_tiles = reduce_indptr.numel() - 1
    out = seed_out.to(dtypes.fp32).clone()
    for t in range(num_tiles):
        base = int(reduce_indptr[t])
        end = int(reduce_indptr[t + 1])
        if end - base < 2:
            continue
        lse = partial_lse[base:end]  # [n, H]
        m = lse.max(dim=0).values  # [H]
        w = torch.exp(lse - m)  # [n, H]
        den = w.sum(dim=0)  # [H]
        acc = (w.unsqueeze(-1) * partial_output[base:end]).sum(dim=0)  # [H, DV]
        out[t] = acc / den.unsqueeze(-1)
    return out.to(seed_out.dtype)


@benchmark()  # (num_tiles, nsplits, dtype) become the table's left-hand columns
def test_mla_reduce(num_tiles, nsplits, dtype):
    device = "cuda"
    P, reduce_indptr, reduce_final_map, reduce_partial_map = build_uniform_decode_meta(
        num_tiles, nsplits, device
    )

    torch.manual_seed(0)
    partial_output = torch.randn(P, H, DV, dtype=dtypes.fp32, device=device)
    partial_lse = torch.randn(P, H, dtype=dtypes.fp32, device=device)
    # Pre-seed final_output: nosplit tiles (nsplits < 2) must be left untouched,
    # so the seed IS the expected output there.
    seed_out = torch.randn(num_tiles, H, DV, dtype=dtype, device=device)
    ref = run_torch(partial_output, partial_lse, reduce_indptr, seed_out)

    return dict(
        partial_output=partial_output,
        partial_lse=partial_lse,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        final_output=final_output,
        final_lse=final_lse,
        max_seqlen_q=qo_len,
        num_out_rows=num_out_rows,
    )


def run_case(splits_per_tile, num_heads, head_dim, out_dtype, qo_len=1):
    p = build_reduce_problem(
        splits_per_tile, num_heads, head_dim, out_dtype, qo_len=qo_len
    )

    # ---- Reference (pure torch) ----
    ref_out = torch.empty_like(p["final_output"])
    ref_lse = torch.empty_like(p["final_lse"])
    torch_mla_reduce_v1(
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        ref_out,
        ref_lse,
    )

    # ---- GPU kernel (timed) ----
    gpu_out = p["final_output"]
    gpu_lse = p["final_lse"]
    # num_kv_splits sizes the LDS scratch (kernel uses max(CU_count, num_kv_splits)),
    # so it must be >= the largest per-tile split count.
    num_kv_splits = max(splits_per_tile)
    _, us = run_perftest(
        aiter.mla_reduce_v1,
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        num_kv_splits,
        gpu_out,
        gpu_lse,
    )

    tag = (
        f"splits={splits_per_tile if len(splits_per_tile) <= 4 else f'{len(splits_per_tile)}x[{splits_per_tile[0]}]'}"
        f" h={num_heads} dv={head_dim} {str(out_dtype).split('.')[-1]}"
    )

    err_out = checkAllclose(
        ref_out.float(),
        gpu_out.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] output {us:>8.2f} us ",
    )
    err_lse = checkAllclose(
        ref_lse,
        gpu_lse,
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] lse    ",
    )
    ok = (err_out == 0) and (err_lse == 0)
    print(f"{'PASS' if ok else 'FAIL'}  {tag}  {us:>8.2f} us")

    # Bytes moved by the reduce: read all partial outputs + partial lses +
    # metadata, write final outputs + final lses. (dominant traffic is the f32
    # partials; metadata is small but counted for completeness)
    total_splits = sum(splits_per_tile)
    num_tiles = len(splits_per_tile)
    read_bytes = (
        total_splits * qo_len * num_heads * head_dim * 4  # partial_output f32
        + total_splits * qo_len * num_heads * 4  # partial_lse f32
        + (num_tiles + 1) * 4  # reduce_indptr int32
        + num_tiles * 2 * 4  # reduce_final_map int32 [T,2]
        + total_splits * 4  # reduce_partial_map int32 [N]
    )
    write_bytes = (
        p["num_out_rows"] * num_heads * head_dim * (torch.finfo(out_dtype).bits // 8)
        + p["num_out_rows"] * num_heads * 4  # final_lse f32
    )
    bytes_total = read_bytes + write_bytes

    ret = {
        "splits": (
            str(splits_per_tile)
            if len(splits_per_tile) <= 4
            else f"{len(splits_per_tile)}x[{splits_per_tile[0]}]"
        ),
        "nhead": num_heads,
        "dv": head_dim,
        "dtype": str(out_dtype).split(".")[-1],
        "qo_len": qo_len,
        "us": round(us, 2),
        "GB/s": round(bytes_total / us / 1e3, 1),
        "pass": ok,
    }
    return ok, ret


def run_case_fast_path(splits_per_tile, num_heads=16, head_dim=512):
    """Exercise the uniform-decode fast path (kn_mla_reduce_v1_fast in
    reduce.cu), not just the general kernel `run_case` above covers.

    The fast path additionally requires max_seqlen_q==1, no LSE output,
    reduce_final_map present, and (h, dv, out_dtype) == (16, 512, bf16) (the
    constexpr gate in dispatch_mla_reduce_v1). With qo_len=1,
    build_reduce_problem already produces exactly the identity metadata the
    fast path assumes: reduce_final_map[t] == [t, t+1] and reduce_partial_map
    == arange(P) — see invariants (A)/(B) documented at their use site in
    reduce.cu (kn_mla_reduce_v1_fast) and asserted host-side under ASM_DEBUG.

    Checks, per split config:
      - fast (final_lse=None) output vs the torch reference
      - fast vs slow (a real final_lse forces fast_eligible=False, routing to
        the general kernel) output parity
      - nosplit tiles (1 split) are left untouched by both paths, mirroring
        the real pipeline where stage1 already wrote that row
    """
    out_dtype = dtypes.bf16
    p = build_reduce_problem(splits_per_tile, num_heads, head_dim, out_dtype, qo_len=1)

    ref_out = torch.empty_like(p["final_output"])
    torch_mla_reduce_v1(
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        ref_out,
        None,
    )

    # nosplit tiles: both kernels skip them (stage1's job in the real
    # pipeline, not the reduce's), so seed final_output and treat the seed as
    # the expected value there instead of the reference's single-split combine.
    seed = torch.randn_like(p["final_output"])
    for t, s in enumerate(splits_per_tile):
        if s < 2:
            q0, q1 = (
                p["reduce_final_map"][t, 0].item(),
                p["reduce_final_map"][t, 1].item(),
            )
            ref_out[q0:q1] = seed[q0:q1]

    num_kv_splits = max(splits_per_tile)

    fast_out = seed.clone()
    _, us_fast = run_perftest(
        aiter.mla_reduce_v1,
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        num_kv_splits,
        fast_out,
        None,
    )

    slow_out = seed.clone()
    slow_lse = torch.empty(
        p["num_out_rows"], num_heads, dtype=dtypes.fp32, device="cuda"
    )
    run_perftest(
        aiter.mla_reduce_v1,
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        num_kv_splits,
        slow_out,
        slow_lse,
    )

    tag = (
        f"splits={splits_per_tile if len(splits_per_tile) <= 4 else f'{len(splits_per_tile)}x[{splits_per_tile[0]}]'}"
        f" h={num_heads} dv={head_dim} bf16 FASTPATH"
    )
    err_ref = checkAllclose(
        ref_out.float(),
        fast_out.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] fast vs ref  ",
    )
    err_parity = checkAllclose(
        slow_out.float(),
        fast_out.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] fast vs slow",
    )
    ok = (err_ref == 0) and (err_parity == 0)
    print(f"{'PASS' if ok else 'FAIL'}  {tag}  {us_fast:>8.2f} us")

    read_bytes = (
        sum(splits_per_tile) * num_heads * head_dim * 4  # partial_output f32
        + sum(splits_per_tile) * num_heads * 4  # partial_lse f32
    )
    write_bytes = (
        p["num_out_rows"] * num_heads * head_dim * (torch.finfo(out_dtype).bits // 8)
    )
    bytes_total = read_bytes + write_bytes

    ret = {
        "splits": (
            str(splits_per_tile)
            if len(splits_per_tile) <= 4
            else f"{len(splits_per_tile)}x[{splits_per_tile[0]}]"
        )
        return out

    candidates = {"fast": fast_call}

    if nsplits >= 2:
        # Requesting an LSE output makes fast_eligible False, forcing the general
        # kernel, which reduces the SAME rows via the reduce_final_map /
        # reduce_partial_map indirections. Its final_output must match the fast
        # path — this is the fast==slow parity check. Skipped for the nosplit
        # config, where the general kernel does not share the fast path's
        # "leave it to stage1" skip semantics.
        final_lse = torch.empty(num_tiles, H, dtype=dtypes.fp32, device=device)

        def slow_call():
            out = seed_out.clone()
            aiter.mla_reduce_v1(
                partial_output,
                partial_lse,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                1,
                nsplits,
                out,
                final_lse,
            )
            return out

        candidates["slow"] = slow_call

    # Memory-bound reduce roofline: read all partials + lse, write the reduced
    # tiles; ~2 flops (mul+add) per accumulated element.
    nbytes = (
        P * H * DV * partial_output.element_size()
        + P * H * partial_lse.element_size()
        + num_tiles * H * DV * seed_out.element_size()
    )
    flops = P * H * DV * 2

    ret = {"gfx": get_gfx(), "P": P}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: mla_reduce_v1 [{num_tiles=} {nsplits=}]",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mla_reduce_v1 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="output dtype. Fast path is bf16-only. Default: bf16.",
    )
    parser.add_argument(
        "-t",
        "--tiles",
        type=int,
        nargs="*",
        default=[1, 8, 64, 128],
        help="number of reduce tiles (== decode tokens). Default: 1 8 64 128.",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        nargs="*",
        # 1 exercises the nosplit skip; 2/4/8/16 sweep the fast-path chunk widths;
        # 3/17/33 straddle those boundaries and the MAX_SPLITS chunk loop.
        default=[1, 2, 3, 4, 8, 16, 17, 33],
        help="kv splits per tile. Default: 1 2 3 4 8 16 17 33.",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for num_tiles, nsplits in itertools.product(args.tiles, args.nsplits):
            df.append(test_mla_reduce(num_tiles, nsplits, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce_v1 summary (%s, markdown):\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
