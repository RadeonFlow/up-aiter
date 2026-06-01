# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# aiter.mla_decode_reduce — CSV-driven MLA decode reduce dispatcher.
#
# Mirrors the structure of aiter.fused_moe: a CSV table
# (tuned_mla_decode_reduce.csv) keyed by (cu_num, num_heads, kv_lora_rank,
# v_head_dim, num_kv_splits) holds the best reduce_kernel name per shape; this
# module parses that name into a BATCH template arg and routes to the per-tuple
# template instantiation in csrc/kernels/mla/decode_reduce/.

from __future__ import annotations

import csv
import functools
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch

from .ops.attention import mla_decode_stage1_asm_fwd
from .ops.mla_decode_reduce import mla_decode_reduce

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Kernel name encoding ────────────────────────────────────────────────────
#
# reduce_kernel:  mla_reduce_BATCH{2,4,8}
#
# This name appears verbatim in tuned_mla_decode_reduce.csv. Adding a new
# template instantiation: extend the C++ switch in mla_decode_reduce.cu *and*
# update the regex below if the encoding grows.

_REDUCE_RE = re.compile(r"^mla_reduce_BATCH(?P<batch>\d+)$")


def _parse_reduce_kname(name: str) -> int:
    m = _REDUCE_RE.match(name or "")
    if not m:
        raise ValueError(f"bad reduce kernel name: {name!r}")
    return int(m.group("batch"))


@dataclass(frozen=True)
class MlaDecodeReduceCfg:
    """Resolved reduce-kernel selection for a (shape, token) tuple."""
    reduce_batch: int   # BATCH template arg for mla_decode_reduce


# ── CSV loader (LRU cached) ─────────────────────────────────────────────────

_CSV_PATH = os.path.join(_THIS_DIR, "configs", "tuned_mla_decode_reduce.csv")


@functools.lru_cache(maxsize=1)
def _load_csv() -> list[dict]:
    if not os.path.exists(_CSV_PATH):
        return []
    path = _CSV_PATH
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Coerce numeric fields used in lookup.
    for r in rows:
        for k in ("cu_num", "token", "num_heads", "kv_lora_rank",
                  "v_head_dim", "num_kv_splits"):
            r[k] = int(r[k])
    return rows


@functools.lru_cache(maxsize=512)
def get_mla_decode_reduce_cfg(
    cu_num: int,
    token: int,
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    num_kv_splits: int,
) -> Optional[MlaDecodeReduceCfg]:
    """Look up the tuned reduce kernel for this (shape, T) tuple.

    Returns None on CSV miss — caller falls back to the legacy path
    (aiter.mla_decode_fwd). Falls back to the `token=0` wildcard (default) row
    when no T-specific row matches.
    """
    rows = _load_csv()
    if not rows:
        return None
    key = (cu_num, num_heads, kv_lora_rank, v_head_dim, num_kv_splits)

    def _match(r):
        return (r["cu_num"], r["num_heads"], r["kv_lora_rank"],
                r["v_head_dim"], r["num_kv_splits"]) == key

    # Prefer exact-T match; fall back to token=0 default row.
    best = None
    for r in rows:
        if not _match(r):
            continue
        if r["token"] == token:
            best = r
            break
        if r["token"] == 0 and best is None:
            best = r
    if best is None:
        return None
    return MlaDecodeReduceCfg(
        reduce_batch=_parse_reduce_kname(best["reduce_kernel"]),
    )


# ── Reduce-only entrypoint (stage1 + LSE-reduce, NO W^V GEMM) ────────────────
#
# Replaces aiter.mla_decode_fwd's built-in LSE-reduce with aiter's stage1 +
# tuned reduce kernel on the MLA decode path for low-head (H=16) models. ATOM
# calls this in `_forward_decode` in place of mla_decode_fwd; on shape mismatch
# / CSV miss the function returns None and the caller falls through to the
# legacy mla_decode_fwd path. No W^V weight is consumed here — the reduce never
# reads the V up-proj weight, so no preshuffle / strong contract is involved.

def _cu_num() -> int:
    # MI300/MI355 family has a fixed CU count per device; we don't need to
    # probe HW here — pass the value from the caller (ATOM knows it). Default
    # to 256 (MI355) so standalone tests Just Work on common config.
    return int(os.environ.get("AITER_CU_NUM", "256"))


# Reduce kernel is instantiated for these split counts (csrc dispatch). Adaptive
# selection snaps to this set so BATCH (a power of two) always divides N_SPLITS.
_SPLIT_CHOICES = (1, 2, 4, 8, 16)


@functools.lru_cache(maxsize=256)
def _adaptive_num_kv_splits(B: int, cu_num: int) -> int:
    """Concurrency-adaptive KV-split count, matching what the stock persistent
    planner does (aiter.mla.get_meta_param's GPU-fill term).

    The reduce is memory-bound: total partial tiles = B * num_splits, so we want
    the FEWEST splits that still fill the GPU. With B decode tokens × H=16 heads
    × (K/BD) D-slices already flooding the CUs at high B, extra splits are pure
    redundant HBM traffic. Pick the split count whose CTA grid best fills the
    last wave (ties → fewer splits = less overhead). Depends only on (B, cu_num),
    so it's host-side and graph-capture safe (no kv-len sync). Short contexts are
    handled downstream: stage1's num_valid = min(num_splits, ceil(kv_len/64)) and
    the reduce's my_valid match, so over-allocating splits for a short seq just
    leaves trailing tiles unread (OOB → 0), never wrong.
    """
    # Heads × D-slices per token already on the grid; one "unit of work" ≈ a CTA.
    # GPU-fill efficiency for i splits = filled_lanes / rounded_up_to_full_waves.
    best_i, best_fill = 1, -1.0
    for i in _SPLIT_CHOICES:
        units = B * i
        waves = -(-units // cu_num)  # ceil
        fill = units / (waves * cu_num)
        if fill > best_fill + 1e-9:  # strictly better → ties keep the smaller i
            best_fill, best_i = fill, i
    return best_i


@functools.lru_cache(maxsize=64)
def _get_num_kv_splits_indptr(B: int, num_kv_splits: int, device_str: str) -> torch.Tensor:
    # Cached by (B, num_kv_splits, device) so the arange doesn't enter the
    # captured CUDA graph per layer.
    return torch.arange(
        0, (B + 1) * num_kv_splits, num_kv_splits,
        dtype=torch.int32, device=device_str,
    )


def mla_reduce_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o_scratch: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    kv_lora_rank: int,
    v_head_dim: int,
) -> Optional[torch.Tensor]:
    """MLA decode stage1 + LSE-reduce only (no W^V GEMM).

    Args:
        q:                 [B, H, qk_head_dim] query
        kv_buffer:         paged KV cache, reshaped to [-1, 1, 1, qk_head_dim]
                           for stage1
        o_scratch:         [B, H, kv_lora_rank] scratch the aiter stage1
                           wrapper expects; not read after the call
        cu_seqlens_q / kv_indptr / kv_indices / kv_last_page_lens / max_seqlen_q:
                           stage1 metadata
        scale:             softmax scale
        q_scale, k_scale:  fp8 dequant scales (one_scale tensor when unused)
        kv_lora_rank, v_head_dim: shape passthrough; v_head_dim is used only
                           for the CSV key (reduce kernel is tuned per full
                           shape tuple).

    Returns:
        [B, H, kv_lora_rank] bf16 — the LSE-reduced attention output, a drop-in
        for the `o` that aiter.mla_decode_fwd writes, ready to feed a legacy V
        up-proj + o_proj tail. Or None on CSV miss, so the caller falls back to
        mla_decode_fwd.
    """
    B, H, qk_head_dim = q.shape
    device = q.device
    cu_num = _cu_num()

    # CSV is keyed on the instantiated split *family* (16); the reduce_batch it
    # carries is tuned for that. The actual runtime split count is adaptive.
    cfg = get_mla_decode_reduce_cfg(
        cu_num=cu_num,
        token=B,
        num_heads=H,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        num_kv_splits=16,
    )
    if cfg is None:
        return None

    # Concurrency-adaptive KV splits: at high decode concurrency the GPU is
    # already flooded, so fewer splits/token = fewer partial tiles = less HBM
    # traffic in stage1's writes AND the reduce's reads (the dominant cost).
    # AITER_MLA_SPLITS pins a value (1/2/4/8/16) for A/B testing; default adaptive.
    _pin = os.environ.get("AITER_MLA_SPLITS")
    num_kv_splits = int(_pin) if _pin else _adaptive_num_kv_splits(B, cu_num)

    num_kv_splits_indptr = _get_num_kv_splits_indptr(B, num_kv_splits, str(device))

    logits = torch.empty(
        (B, num_kv_splits, H, kv_lora_rank),
        dtype=torch.float32, device=device,
    )
    attn_lse_split = torch.empty(
        (B, num_kv_splits, H, 1),
        dtype=torch.float32, device=device,
    )

    mla_decode_stage1_asm_fwd(
        q,
        kv_buffer.view(-1, 1, 1, qk_head_dim),
        cu_seqlens_q,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_kv_splits_indptr,
        None, None, None,            # non-persistent (stage1 doesn't need work_*)
        max_seqlen_q,
        1, 1,                        # page_size, nhead_kv
        scale,
        logits, attn_lse_split, o_scratch, None,
        q_scale, k_scale,
    )

    reduced_scratch = torch.empty(
        (B, H, kv_lora_rank), dtype=torch.bfloat16, device=device,
    )
    mla_decode_reduce(
        logits.view(B * num_kv_splits, H, kv_lora_rank),
        attn_lse_split.view(B * num_kv_splits, H),
        reduced_scratch,
        kv_indptr,
        B,
        cfg.reduce_batch,   # C++ clamps batch to num_kv_splits
        vec=4,              # b128 loads: best achieved BW (see bench_mla_reduce_micro)
        num_splits=num_kv_splits,
    )
    return reduced_scratch


__all__ = [
    "MlaDecodeReduceCfg",
    "get_mla_decode_reduce_cfg",
    "mla_reduce_decode",
]
