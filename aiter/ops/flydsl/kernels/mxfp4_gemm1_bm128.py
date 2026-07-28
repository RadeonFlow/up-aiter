# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from . import dpp_utils
from .mxfp4_gemm_common import (
    kStages,
    kBS_stride_k0_dw,
    _raw,
    _lds_ptr3,
    _lds_base_ptr3,
    _gep3,
    _global_base_ptr1,
    _gep1,
    _global_ptr1,
    _buffer_rsrc,
    _lds_swizzle_mask,
    _fabs_f32,
    _e8m0_roundup,
    _e8m0_from_amax,
    _umax_i32,
    _inline_dpp_quad_amax,
    kmchunks_for,
    lds_acc_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kunroll_for,
    kbs_stride_n0_dw_for,
    kas_per_chunk_dw_for,
    num_n_blocks_for,
    kbs_per_expert_dw_for,
    bq_bytes_for,
    bscale_bytes_for,
)


# Experiment knobs (env-driven so a sweep never needs a source edit; every run
# must still use a cold FlyDSL cache -- see /dev/shm/nocache.py).
_ASM_ALBD = False
_FENCE_VMCNT = 14
_ADSRD = False

# B-scale wide load. The preshuffled B-scale for one n0 unit (32 N rows) is
# CONTIGUOUS along K: K-tile t sits at byte t*256 within the unit. So one
# `buffer_load_dwordx4 ... lds` (64 lanes x 16 B = 1024 B) fetches FOUR K-tiles at
# once, cutting the steady-loop B-scale VMEM count from 2/iter to 2 per 4 iters.
# LDS is only needed to transpose: the gather lands scales in natural order
# (lane g -> bytes g*16..+15), while the MFMA wants lane L to hold scale L of one
# tile -- a stride-256 gather no single VMEM op can do. Same idiom as
# fp4_gemm_4wave's ScaleLoaderLDS.
_BSC_X4 = False

# Thunk-interleaved steady loop: instead of a 19-ds_read block before the first
# mfma, issue only what the first quad needs and weave the rest one per
# _ILV_STRIDE mfma, so each load hides in an mfma execute shadow.
_BDEEP = True
# i-outer mfma order (for k: for i: for J) with every load woven in.
# Requires _BDEEP: the B loads can only move once B is triple-buffered.
_IOUT = True
_ILV = False
_ILV_STRIDE = 2
_BSC_TILES = 4  # K-tiles covered by one dwordx4 gather (1024 B / 256 B)
_BSC_SLOTS = 2  # double buffer over groups of _BSC_TILES
_BSC_WAVE_BYTES = 2 * _BSC_TILES * 256  # 2 mw x 4 tiles x 256 B = 2 KB
_BSC_SLOT_BYTES = 4 * _BSC_WAVE_BYTES  # 4 waves = 8 KB
_BSC_LDS_BYTES = _BSC_SLOTS * _BSC_SLOT_BYTES  # 16 KB


def _build_thunk_plan(kMChunks, kSubBlocks):
    """Compile-time (NON-traced) plan for the interleaved steady loop.

    Must live outside @flyc.kernel: a `for` over a python list inside a traced
    function gets rewritten into scf.for by the DSL tracer.

    Returns (prologue, thunks):
      prologue -- what the FIRST mfma quad of the NEXT iteration needs, so that
        iteration can start issuing mfma immediately after its barrier.
        mfma_cluster(J=0) consumes, in order, a[0][0] a[1][0] a[0][1] a[1][1]
        with a_scale[0] -- M-blocks 0/1, both k-halves, plus A-scale sub 0.
      thunks -- the rest of the NEXT iteration's A, plus this iteration's albd
        refill, woven one per `stride` mfma across all 64 mfma of the current
        iteration.

    Why the loads must be for the NEXT iteration, not this one: within a single
    J the 16 mfma already touch every one of the 8 M-blocks (mfma 4*s+{0,1,2,3}
    reads a[2s][0] a[2s+1][0] a[2s][1] a[2s+1][1]). There is no slack to hoist
    A[i] above its own consumer inside J=0 -- weaving this iteration's A at
    stride 2 would fire a[2][0] after mfma#6 when mfma#4 already needs it.
    fp4_gemm_4wave can interleave its own tile because one quad's 32 mfma reuse
    only 4 A tiles (each across 4 j); we have no such reuse. Prefetching the
    next tile instead gives the loads all 64 mfma to hide in.
    Safe because kAStages=3: iteration OFFSET reads slot OFFSET%3, filled by
    the albd of OFFSET-2 and published by the barrier at the top of OFFSET-1.

    Ordering rules inside `thunks`:
      * `albd` writes the A LDS write_slot. Only the iteration-top s_barrier
        guarantees the other 3 waves are done reading it, so albd may not be
        hoisted above that barrier -- but it is free to sit anywhere after it.
        It targets slot (OFFSET+2)%3, a different buffer from the (OFFSET+1)%3
        the prefetch ds_reads are reading, so the two do not race.
      * `bld` overwrites b[slot_b], which EVERY mfma of this iteration reads
        (WAR), so all 8 B loads stay after the last mfma of their J. Same for
        the B-scale gather.
    """
    pro = [("asc", 0), ("a", 0, 0), ("a", 1, 0), ("a", 0, 1), ("a", 1, 1)]
    th = []
    for sub in range(kSubBlocks):
        th.append(("albd", sub))
    for sub in range(1, kSubBlocks):
        th.append(("asc", sub))
    for i in range(kMChunks):
        for k in range(2):
            if (i, k) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                continue
            th.append(("a", i, k))
    return pro, th


def _mfma_weave_order(kMChunks, kSubBlocks):
    """(i, k) consumption order of one mfma_cluster call, matching the loop in
    mfma_cluster: for sub in kSubBlocks: i0=2s, i1=2s+1, then k=0,0,1,1."""
    out = []
    for sub in range(kSubBlocks):
        i0, i1 = sub * 2, sub * 2 + 1
        out += [(i0, 0), (i1, 0), (i0, 1), (i1, 1)]
    return out


def _iouter_run(act, nxt_slot, nxt_kt, write_slot, K_C, b_slot, a_nxt, asc_nxt,
                f_a, f_asc, f_albd, f_bld):
    """Execute one weave action. Module-level with everything passed explicitly:
    the DSL AST rewriter mangles closures over a traced function's locals (their
    captures silently come back empty)."""
    kind = act[0]
    if kind == "a":
        a_nxt[act[1]][act[2]] = f_a(nxt_slot, act[1], act[2])
    elif kind == "asc":
        asc_nxt[act[1]] = f_asc(nxt_kt, act[1])
    elif kind == "albd":
        f_albd(write_slot, K_C, act[1])
    elif kind == "bld":
        f_bld(b_slot, K_C, act[1], act[2])


def _build_iouter_plan(kMChunks, kSubBlocks):
    """Schedule for the i-outer mfma order: for k(2): for i(8): for J(4).

    See resource_inspect/gen_gemm1_iouter_schedule.py, which emits the same plan
    as a readable listing next to the fp4_gemm_4wave reference.

    The order change is what makes weaving possible. With J outermost (the
    original), J=0's 16 mfma already touch all 8 M-blocks, so every A ds_read
    must land before the first mfma -- that is the ~300-cycle dead head we
    measured. With i outermost, A[i,k] is reused by 4 consecutive mfma, so
    A[i+1,k] only has to arrive 4 mfma later and hides in A[i]'s shadow.
    It also stretches the same accumulator's reuse distance from 8 mfma to 32.

    Returns (carry, weave):
      carry -- issued at the END of the previous iteration: what the first mfma
               needs (A[0,0] A[0,1] Asc[0]).
      weave -- [(mfma_index, action)] for this iteration, action being
               ("a",i,k) / ("asc",sub) / ("albd",sub) / ("bld",j,half).
               Every entry is placed before its first consumer, and no two VMEM
               ops land within 2 mfma of each other (back-to-back buffer_loads
               queue on L1 and their issue latency blows up).
    """
    order = [(k, i, j) for k in range(2) for i in range(kMChunks) for j in range(4)]
    first_a = {}
    for n, (k, i, _j) in enumerate(order):
        first_a.setdefault((i, k), n)

    carry = [("asc", 0), ("a", 0, 0), ("a", 0, 1)]

    # ds_read queue, ordered by deadline; A[0,*] and Asc[0] come in on the carry.
    todo = [(first_a[(i, k)], ("a", i, k)) for k in range(2) for i in range(1, kMChunks)]
    todo += [(first_a[(2 * s, 0)], ("asc", s)) for s in range(1, kSubBlocks)]
    # next iteration's carry, re-read at the end so it is freshest
    todo += [(len(order), a) for a in carry]
    todo.sort(key=lambda x: x[0])

    # VMEM: 4 albd (no deadline this iteration -- the NEXT barrier publishes the
    # slot) + 8 B loads (free to move only because B is triple-buffered).
    vmem = [("albd", s) for s in range(kSubBlocks)]
    vmem += [("bld", j, h) for h in range(2) for j in range(4)]
    step = len(order) // (len(vmem) + 1)
    vmem_at = {1 + step * (n + 1): v for n, v in enumerate(vmem)}

    weave = []
    ti = 0
    for n in range(len(order)):
        if n in vmem_at:
            weave.append((n, vmem_at[n]))
            continue
        if ti < len(todo):
            deadline, act = todo[ti]
            if n < deadline:
                weave.append((n, act))
                ti += 1
    for _, act in todo[ti:]:
        weave.append((len(order) - 1, act))

    # VMEM ops issued after the LAST albd. The steady fence must use exactly
    # this count so it retires all 4 albd (the barrier that follows publishes
    # that A slot to the other 3 waves) without draining anything else.
    vm_seq = [a for _, a in weave if a[0] in ("albd", "bld")]
    post_albd = len(vm_seq) - 1 - max(
        i for i, a in enumerate(vm_seq) if a[0] == "albd"
    )
    return carry, weave, post_albd


def _pipe_alloc(pipe, shape):
    """pipe[1] <- a fresh empty holder (shape = kMChunks for A, else a count).

    Any `pipe[1] = ...` written inside `if const_expr(...)` in a traced kernel
    is dropped by the DSL AST rewriter, which treats names assigned in a
    control-flow body as captured closure variables. Doing the store from a
    module-level function keeps it out of the rewriter's way.
    """
    if isinstance(shape, tuple):
        pipe[1] = [[None, None] for _ in range(shape[0])]
    else:
        pipe[1] = [None] * shape


def _rotate_pipe(pipe):
    """pipe[0] <- pipe[1] (in place). Same rewriter caveat as _pipe_alloc."""
    pipe[0] = pipe[1]


def _weave_run(act, ctx, fns, a_pipe, asc_pipe):
    """Execute one weave action. `act` is a plain tuple from the compile-time
    plan, `ctx` = (nxt_slot, nxt_kt, write_slot, K_C), `fns` = the three issue
    helpers. Module-level so no closure over traced-function locals is needed.
    """
    nxt_slot, nxt_kt, write_slot, K_C = ctx
    f_a, f_asc, f_albd = fns
    if act[0] == "asc":
        asc_pipe[1][act[1]] = f_asc(nxt_kt, act[1])
    elif act[0] == "a":
        a_pipe[1][act[1]][act[2]] = f_a(nxt_slot, act[1], act[2])
    else:
        f_albd(write_slot, K_C, act[1])


def _weave_tick(weave, cursor, stride, ctx, fns, a_pipe, asc_pipe):
    """Issue the next weave action if this mfma is on a `stride` boundary.

    Module-level (not nested in the traced kernel) so the DSL AST rewriter,
    which rebinds names assigned in the enclosing traced function, cannot break
    it. cursor is [next_action, mfma_count], mutated in place.
    """
    if cursor[0] < len(weave) and (cursor[1] % stride) == 0:
        _weave_run(weave[cursor[0]], ctx, fns, a_pipe, asc_pipe)
        cursor[0] += 1
    cursor[1] += 1


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def n_out_for(inter):
    return 2 * inter


def out_as_per_chunk_dw_for(inter):
    return ((inter // 32) // 4 // 2) * 64


def k_g2_half_for(inter):
    return inter // 2


LOG2E = 1.4426950408889634


def _silu_mul(g, u):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    sig = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
    return g * sig * u


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = ir.Type.parse("vector<2xi16>")
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _inline_e8m0(amax_u16_i32):
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32)


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


@flyc.jit
def _gemm1_body(
    allocator,
    lds_off,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    i32_total_m_blocks,
    *,
    BM,
    BN,
    BK,
    KH_TILE,
    kAStages,
    kSubBlocks,
    kMChunks,
    K,
    K_HALF,
    K_TILES_TOTAL,
    kUnroll,
    kAS_per_chunk_dw,
    kBS_stride_n0_dw,
    kBS_per_expert_dw,
    BQ_BYTES,
    BSCALE_BYTES,
    N_OUT,
    NUM_N_BLOCKS,
    OUT_AS_PER_CHUNK_DW,
    K_G2_HALF,
):
    BN_INT = BN // 2
    M_REPS = BM // 16

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(
        T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    )
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_HALF)))
    aq_rsrc = _buffer_rsrc(arg_aq, aq_num_records)
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = _buffer_rsrc(arg_ascale, ascale_num)
    bq_rsrc = _buffer_rsrc(arg_bq, BQ_BYTES)
    bscale_rsrc = _buffer_rsrc(arg_bscale, BSCALE_BYTES)

    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,))
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    s_bsc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE + kSubBlocks * K_TILES_TOTAL * 256,
        T.i8,
        shape=(_BSC_LDS_BYTES,),
    )
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks):
        idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        cached_actual_row.append(
            llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4)))
        )

    # Row indices for the weave-able albd, materialized once with plain python
    # indices. cached_actual_row above is built inside `range_constexpr`, which
    # makes it unusable from a helper called with a python int sub.
    _wrows = [None] * (BM // 32)
    if const_expr(_ILV or _IOUT):
        # range_constexpr (a real python unroll) -- a bare `range` here is
        # rewritten into a dynamic scf.for, which cannot carry a python list.
        for sub in range_constexpr(BM // 32):
            _widx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            _wrows[sub] = fx.Int32(
                llvm.load(T.i32, _global_ptr1(arg_mind, _widx * fx.Int32(4)))
            )

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(4):
        tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j)
        g = tile_il & fx.Int32(1)
        n0 = tile_il >> fx.Int32(1)
        col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    np_gate = n_block_idx * fx.Int32(BN // 64) + wave
    np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * 4 for _ in range(kMChunks)]
    # B is triple-buffered when _BDEEP: with only 2 buffers the load that refills
    # b[slot_b] targets the very buffer this iteration's mfma are still reading,
    # so bld[j] may not be issued before the last mfma that reads B[j] (WAR).
    # That pins all 8 B loads to fixed points in the schedule. With 3 buffers the
    # load writes a slot nobody is reading, so it can go anywhere -- which is
    # what lets the loads be spread evenly through the mfma stream.
    # Cost is 32 VGPR (4 J x 2 halves x i32x4); the kernel uses 284 of 512.
    kBStages = 3 if _BDEEP else kStages
    b = [[[None, None] for _ in range(4)] for _ in range(kBStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            base_i32 = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            if const_expr(_ASM_ALBD):
                # INLINE-ASM g2s (fp4_gemm_4wave idiom): LLVM sees no LDS write
                # here, so its alias analysis stops inserting the extra
                # `s_waitcnt vmcnt(10)` before the following ds_reads. The A slot
                # read this iteration was filled kAStages iters ago and is already
                # covered by our explicit vmcnt fence.
                m0 = rocdl.readfirstlane(T.i32, _raw(base_i32 + off))
                llvm.inline_asm(
                    None,
                    [
                        _raw(m0),
                        _raw(voffset),
                        _raw(aq_rsrc),
                        _raw(fx.Int32(kt * KH_TILE)),
                    ],
                    "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds",
                    "s,v,s,s",
                    has_side_effects=True,
                )
            else:
                rocdl.raw_ptr_buffer_load_lds(
                    aq_rsrc,
                    _lds_ptr3(base_i32, off),
                    fx.Int32(16),
                    voffset,
                    fx.Int32(kt * KH_TILE),
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_load_lds_one(slot, kt, sub):
        """One albd step, for weaving into the mfma stream.

        Uses the row cached ONCE before the loop (_wrows). Re-issuing the
        `arg_mind` global load here instead forces the m0 readfirstlane to wait
        on it, and the compiler inserts an `s_waitcnt vmcnt(0)` before every
        albd -- which serializes the whole pipeline (measured: 105 of them).
        """
        lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
        mask = _lds_swizzle_mask(lds_row + lane_div_8)
        voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + _wrows[sub] * fx.Int32(K_HALF)
        base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_aq.get()))
        off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
        m0 = rocdl.readfirstlane(T.i32, _raw(base_i32 + off))
        llvm.inline_asm(
            None,
            [_raw(m0), _raw(voffset), _raw(aq_rsrc), _raw(fx.Int32(kt * KH_TILE))],
            "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds",
            "s,v,s,s",
            has_side_effects=True,
        )

    def issue_a_ds_read_one(slot, i, k):
        """One A fragment ds_read (M-block i, k-half k) -> i32x4."""
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
        lds_row = lane_mod_16 + fx.Int32(i * 16)
        off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE) + lds_col
        return llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))

    def issue_a_scale_ds_read_one(kt, sub):
        """One A-scale ds_read -> i32."""
        base_ptr = _lds_base_ptr3(s_asc.get())
        lds_dw = (
            fx.Int32(sub * kAS_per_chunk_dw)
            + fx.Int32(kt * 64)
            + lane_div_16 * fx.Int32(16)
            + lane_mod_16
        )
        return llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4)))

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][k] = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                rocdl.raw_ptr_buffer_load_lds(
                    ascale_rsrc,
                    _lds_ptr3(
                        asc_base, lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256)
                    ),
                    fx.Int32(4),
                    v4,
                    s_off,
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_scale_ds_read(kt):
        base_ptr = _lds_base_ptr3(s_asc.get())
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4))))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave * fx.Int32(4) + lane_div_16

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            frag = buffer_ops.buffer_load(
                bq_rsrc,
                (v + fx.Int32(half * 1024)) // fx.Int32(4),
                vec_width=4,
                dtype=T.i32,
                soffset_bytes=b_load_s_base[j],
            )
            b_slot[j][half] = Vec(frag)

    def issue_b_load_one(b_slot, K_C, j, half):
        """One B buffer_load, for weaving individually into the mfma stream.
        Only legal to place freely when B is triple-buffered (_BDEEP): with 2
        buffers this write races the mfma still reading the same slot."""
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        b_slot[j][half] = Vec(
            buffer_ops.buffer_load(
                bq_rsrc,
                (v + fx.Int32(half * 1024)) // fx.Int32(4),
                vec_width=4,
                dtype=T.i32,
                soffset_bytes=b_load_s_base[j],
            )
        )

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            bs_slot[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=s_off,
            )

    # ---- wide (dwordx4) B-scale path -------------------------------------
    # gather: lane g reads the 16 contiguous bytes at unit_base + grp*1024 + g*16
    #   and writes them to LDS at wave_region + g*16. Because the gather is dense
    #   and in order, i32 j of the unit's 1024-byte window lands at LDS byte j*4,
    #   i.e. tile t's scale L is at (t*256 + L*4) -- exactly what the ds_read wants.
    # grp = K_C // _BSC_TILES; the 4 tiles of a group share one gather.
    bsc_lds_base = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_bsc.get()))
    bsc_wave_off = wave * fx.Int32(_BSC_WAVE_BYTES)
    bsc_v = lane * fx.Int32(16)
    bsc_read_v = lane * fx.Int32(4)

    def issue_b_scale_gather(grp):
        slot = grp % _BSC_SLOTS
        for mw in range_constexpr(2):
            lds_off_b = (
                fx.Int32(slot * _BSC_SLOT_BYTES)
                + bsc_wave_off
                + fx.Int32(mw * _BSC_TILES * 256)
            )
            m0 = rocdl.readfirstlane(T.i32, _raw(bsc_lds_base + lds_off_b))
            # byte offset of this group inside the n0 unit = grp*_BSC_TILES*256
            s_off = rocdl.readfirstlane(
                T.i32, _raw(b_scale_s_base[mw] + fx.Int32(grp * _BSC_TILES * 256))
            )
            llvm.inline_asm(
                None,
                [_raw(m0), _raw(bsc_v), _raw(bscale_rsrc), _raw(s_off)],
                "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds",
                "s,v,s,s",
                has_side_effects=True,
            )

    def read_b_scale(K_C):
        grp = K_C // _BSC_TILES
        t = K_C - grp * _BSC_TILES
        slot = grp % _BSC_SLOTS
        out = [None, None]
        base_ptr = _lds_base_ptr3(s_bsc.get())
        for mw in range_constexpr(2):
            off = (
                fx.Int32(slot * _BSC_SLOT_BYTES + mw * _BSC_TILES * 256 + t * 256)
                + bsc_wave_off
                + bsc_read_v
            )
            out[mw] = fx.Int32(llvm.load(T.i32, _gep3(base_ptr, off)))
        return out

    mfma_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)

    def mfma_cluster(
        b_slot, a, a_scale, bs_slot, J, init, weave=None, stride=2, cursor=None,
        wctx=None, wpipes=None,
    ):
        # `weave` is a list of zero-arg thunks (ds_read / buffer_load). One is
        # issued every `stride` mfma so each load gets more than one mfma execute
        # shadow to hide behind. fp4 mfma is ~6-cyc issue / 16-cyc execute, so a
        # 5-cyc ds_read or 3-cyc buffer_load fits free in the gap -- 64 mfma give
        # 64*(16-6) = 640 shadow cycles against only ~137 cycles of loads to
        # issue. Bunching them (as a plain block of 19 ds_reads) wastes that.
        # Mirrors fp4_gemm_4wave's MmaFp4._interleaved_cluster.
        # `cursor` is [n, m] carried across the 4 quads of one iteration so the
        # weave spreads over all 64 mfma instead of restarting each quad.
        # _tick is the module-level helper, called with explicit arguments: the
        # DSL AST rewriter rebinds names assigned in the enclosing traced
        # function, so a nested def that reads mfma_cluster's locals breaks.
        # No nested def for the tick, and no closures in the weave: the DSL AST
        # rewriter breaks both (UnboundLocalError, or silently-lost default-arg
        # bindings). The weave is a list of plain tuples executed inline below.
        _wv = weave if weave is not None else []
        _cur = cursor if cursor is not None else [0, 0]
        _wctx = wctx if wctx is not None else (0, 0, 0, 0)
        _wfns = (issue_a_ds_read_one, issue_a_scale_ds_read_one, issue_a_load_lds_one)
        _wap = wpipes[0] if wpipes is not None else [None, []]
        _wsp = wpipes[1] if wpipes is not None else [None, []]

        mni = J % 2
        in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        if const_expr(kMChunks == 1):
            sa = a_scale[0]
            if const_expr(init):
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
            else:
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, accm[0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
            accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[0][1], bJ1, accm[0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
        else:
            for sub in range_constexpr(kSubBlocks):
                i0 = sub * 2 + 0
                i1 = sub * 2 + 1
                sa = a_scale[sub]
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                    )
                    _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb]
                    )
                    _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][1], bJ1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb]
                )
                _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][1], bJ1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb]
                )
                _weave_tick(_wv, _cur, stride, _wctx, _wfns, _wap, _wsp)

    def mfma_iouter(b_slot, b_wr, a, a_scale, bs_slot, init, weave, ctx,
                    a_nxt, asc_nxt):
        """Emit all 64 mfma in i-outer order (k, i, J) with `weave` woven in.

        Original order is J outermost, which gives A zero reuse inside a quad and
        forces all 19 ds_reads to complete before the first mfma. Here A[i,k] is
        shared by 4 consecutive mfma (one per J), so the next A can be read in
        their shadow, and the same accumulator is revisited every 32 mfma instead
        of every 8.
        """
        nxt_slot, nxt_kt, write_slot, K_C = ctx
        at = {}
        for n, act in weave:
            at.setdefault(n, []).append(act)
        n = 0
        for k in range_constexpr(2):
            for i in range_constexpr(kMChunks):
                a_ik = a[i][k]
                sa = a_scale[i // 2]
                # opsel selects the byte lane of the packed e8m0 scale:
                # A side (i%2) + 2k, B side (J//2) + 2k -- same encoding the
                # J-outer mfma_cluster uses.
                osa = (i % 2) + 2 * k
                for J in range_constexpr(4):
                    # matches mfma_cluster (interleave=False): the B-scale slot
                    # is indexed by J%2 while the opsel byte-lane uses J//2.
                    osb = (J // 2) + 2 * k
                    # Zero only on the very first mfma of an accumulator, i.e.
                    # k==0 of the first iteration. k==1 always accumulates onto
                    # k==0's result -- zeroing it too drops half the K-tile.
                    src = zero4 if const_expr(init and k == 0) else accm[i][J]
                    accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty,
                        [a_ik, b_slot[J][k], src, 4, 4, osa, sa, osb, bs_slot[J % 2]],
                    )
                    for act in at.get(n, ()):
                        _iouter_run(
                            act, nxt_slot, nxt_kt, write_slot, K_C, b_wr,
                            a_nxt, asc_nxt,
                            issue_a_ds_read_one, issue_a_scale_ds_read_one,
                            issue_a_load_lds_one, issue_b_load_one,
                        )
                    n += 1

    _relax_prologue = True
    # ADSRD: rotate the A / A-scale ds_reads one iteration EARLIER, so they issue
    # at the END of the previous iteration (inside its mfma shadow) instead of in
    # the ~300-cycle bare window between s_barrier and the first mfma (ATT: that
    # head is 19% of the steady iteration with zero mfma covering it).
    # Safe because kAStages=3: iter OFFSET reads slot OFFSET%3, which was filled by
    # the albd of iter OFFSET-2 and published by the barrier at the top of OFFSET-1.
    # So iter OFFSET-1 may already read it, and the concurrent albd of OFFSET-1
    # writes slot (OFFSET+1)%3 -- a different buffer.
    _adsrd = _ADSRD
    _bsc_x4 = _BSC_X4
    _ilv = _ILV
    _iout = _IOUT and _BDEEP
    _IOUT_CARRY, _IOUT_WEAVE, _IOUT_POST_ALBD = _build_iouter_plan(
        kMChunks, kSubBlocks
    )
    _PRO, _TH = _build_thunk_plan(kMChunks, kSubBlocks)
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
        if const_expr(not _relax_prologue):
            for j in range_constexpr(4):
                issue_b_load_j(b[K_C], K_C, j)
        if const_expr(not _relax_prologue):
            issue_b_scale_load(b_scale_v[K_C], K_C)
    if const_expr(_relax_prologue):
        rocdl.sched_barrier(0)
        if const_expr(_bsc_x4):
            # Group 0 first so it is the OLDEST VMEM in the prologue and the
            # steady fence has certainly retired it before iteration 0 reads it.
            issue_b_scale_gather(0)
        for K_C in range_constexpr(kStages):
            for j in range_constexpr(4):
                issue_b_load_j(b[K_C], K_C, j)
            if const_expr(not _bsc_x4):
                issue_b_scale_load(b_scale_v[K_C], K_C)

    # ADSRD prologue: iteration 0's A/A-scale ds_reads. Needs the first barrier to
    # publish slot 0, so it sits right after a full barrier here.
    a_pipe = [None, None]
    asc_pipe = [None, None]
    if const_expr(_adsrd or _ilv or _iout):
        gpu.barrier()
        asc_pipe[0] = issue_a_scale_ds_read(0)
        a_pipe[0] = issue_a_ds_read(0)

    for OFFSET in range_constexpr(kUnroll): #  28 主循环.
        K_C = kStages + OFFSET # 2 + i
        read_slot = OFFSET % kAStages # 
        write_slot = K_C % kAStages
        slot_b = OFFSET % kBStages
        # B-scale keeps its own 2-deep register buffer (independent of the B
        # data buffers), so it is indexed with kStages, not kBStages.
        slot_bsc = OFFSET % kStages
        # Tile K_C = OFFSET+kStages is the one being prefetched. With kBStages=3
        # its slot differs from the slot being read (OFFSET%3), so the refill has
        # no WAR against this iteration's mfma and can be issued anywhere.
        write_b = K_C % kBStages
        # Steady fence: A read from read_slot was written by iter OFFSET-2
        # (already landed), so barrier need NOT wait on this iter's 14 in-flight
        # VMEM. Compiler can't see the double-buffer -> drains to vmcnt(10);
        # relax to vmcnt(14) (don't gate VMEM). s_barrier keeps 4-wave sync.
        if True:
            # ADSRD needs the PREVIOUS iteration's 4 albd drained *before* the
            # barrier, so the barrier publishes slot (OFFSET+1)%3 to all 4 waves
            # and this iteration's tail ds_read of that slot is safe. In flight at
            # this point = iter OFFSET-1's 14 VMEM issued albd(4) -> bld(8) ->
            # bsc(2), so vmcnt(10) retires exactly the 4 oldest = the albd.
            # The fence must retire the previous iteration's albd, because the
            # barrier right after it is what publishes that A slot to the other
            # 3 waves. The count is "how many VMEM ops the weave issues AFTER
            # the last albd": _adsrd/_ilv keep the original albd -> bld -> bsc
            # order (4 albd first, then 10), while _iout's plan puts the 4 albd
            # at mfma 5/9/13/17 followed by exactly 8 B loads.
            if const_expr(_iout):
                _fv = _IOUT_POST_ALBD
            elif const_expr(_adsrd or _ilv):
                _fv = 10
            else:
                _fv = _FENCE_VMCNT
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt vmcnt({_fv})", "", has_side_effects=True
            )
            rocdl.s_barrier()
        else:
            gpu.barrier()
        _weave = None
        _wctx = (0, 0, 0, 0)
        _wcur = [0, 0]
        if const_expr(_ilv or _iout):
            # A / A-scale for THIS iteration were prefetched by the previous one
            # (or by the prologue), so the mfma stream starts right after the
            # barrier with no ds_read block in front of it.
            a_cur = a_pipe[0]
            asc_cur = asc_pipe[0]
            nxt_slot = (OFFSET + 1) % kAStages
            # Clamp: the final iteration would prefetch tile K_TILES_TOTAL. The
            # tail loop re-reads what it needs, so the clamped extra read is
            # harmless (idempotent, result never consumed).
            nxt_kt = min(K_C - kStages + 1, K_TILES_TOTAL - 1)
            # The weave is a PLAN (list of tuples), not a list of closures: the
            # DSL AST rewriter mangles lambdas that capture this function's
            # locals -- their default-arg bindings are silently lost, so the
            # thunks write into a stale list and the carry comes out empty.
            # mfma_cluster executes each action inline instead.
            _pipe_alloc(a_pipe, (kMChunks,))
            _pipe_alloc(asc_pipe, kSubBlocks)
            # The next iteration's first-quad operands go LAST in the weave: they
            # are the newest ds_reads, so putting them at the end of this
            # iteration's mfma stream still leaves them a full barrier away from
            # their use, while the earlier thunks get the deepest shadow.
            _weave = _TH + _PRO
            _wctx = (nxt_slot, nxt_kt, write_slot, K_C)
        elif const_expr(_adsrd):
            # issued at the tail of the previous iteration, inside its mfma shadow
            asc_cur = asc_pipe[0]
            a_cur = a_pipe[0]
        else:
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
            a_cur = issue_a_ds_read(read_slot)
        if const_expr(_bsc_x4):
            # tile OFFSET's scales, gathered ~4 iterations ago (>=48 VMEM ops), so
            # the vmcnt fence above has long retired that dwordx4. Each wave owns
            # its own LDS region here, so no cross-wave barrier is needed.
            bs_cur = read_b_scale(OFFSET)
        else:
            bs_cur = b_scale_v[slot_bsc]
        if const_expr(not _ilv and not _iout):
            issue_a_load_lds(write_slot, K_C) # d2s
        if const_expr(_iout):
            # i-outer: all 64 mfma emitted together, every load woven in.
            _pipe_alloc(a_pipe, (kMChunks,))
            _pipe_alloc(asc_pipe, kSubBlocks)
            mfma_iouter(
                b[slot_b], b[write_b], a_cur, asc_cur, bs_cur,
                (OFFSET == 0), _IOUT_WEAVE,
                ((OFFSET + 1) % kAStages,
                 min(K_C - kStages + 1, K_TILES_TOTAL - 1),
                 write_slot, K_C),
                a_pipe[1], asc_pipe[1],
            )
        else:
          for J in range_constexpr(4):
            # One shared weave cursor across all 4 quads: 24 thunks at stride 2
            # need 48 of the 64 mfma, so they spread over the whole iteration
            # rather than bunching into J=0.
            mfma_cluster(
                b[slot_b], a_cur, asc_cur, bs_cur, J, init=(OFFSET == 0),
                weave=_weave, stride=_ILV_STRIDE, cursor=_wcur,
                wctx=_wctx, wpipes=(a_pipe, asc_pipe),
            )
            if const_expr(not _ilv):
                rocdl.sched_barrier(0)
            issue_b_load_j(b[write_b], K_C, J) # 2 * B128
            if const_expr(not _ilv):
                rocdl.sched_barrier(0)
            if const_expr(_adsrd and J == 1):
                # Next iteration's A/A-scale, issued mid-mfma so the LDS latency
                # overlaps the remaining 32 mfma. Slot (OFFSET+1)%kAStages was
                # filled by iter OFFSET-1's albd and published by this iteration's
                # top barrier; the albd running now targets a different slot.
                nxt = (OFFSET + 1) % kAStages
                asc_pipe[0] = issue_a_scale_ds_read(K_C - kStages + 1)
                a_pipe[0] = issue_a_ds_read(nxt)
                rocdl.sched_barrier(0)
        if const_expr(_bsc_x4):
            # One gather every _BSC_TILES iterations instead of 2 loads every
            # iteration. Issue it on the FIRST iteration of the current group, so
            # the next group's gather has a full _BSC_TILES iterations (~56 VMEM
            # ops) to land before its first read. Issuing it on the LAST iteration
            # leaves only one iteration of lead, and the steady vmcnt(14) fence
            # then retires nothing -- the read races the gather (verified: nan).
            # The slot written is the other one of the 2, so the group being read
            # right now is untouched.
            if const_expr((OFFSET % _BSC_TILES) == 0):
                nxt_grp = OFFSET // _BSC_TILES + 1
                if const_expr(nxt_grp * _BSC_TILES < K_TILES_TOTAL):
                    issue_b_scale_gather(nxt_grp)
        else:
            issue_b_scale_load(b_scale_v[slot_bsc], K_C) # 2 * B32?
        if const_expr(_ilv):
            # Drain any thunk the 64 mfma did not reach, then rotate the carry:
            # what this iteration prefetched becomes the next one's operands.
            # The bounds are plain python ints (the weave plan is compile-time),
            # so slice the list directly -- a `while`, or an `if` inside the
            # loop, would be rewritten into scf and break on the list index.
            for _t in _weave[_wcur[0]:]:
                _weave_run(
                    _t, _wctx,
                    (issue_a_ds_read_one, issue_a_scale_ds_read_one,
                     issue_a_load_lds_one),
                    a_pipe, asc_pipe,
                )
            _wcur[0] = len(_weave)
            _rotate_pipe(a_pipe)
            _rotate_pipe(asc_pipe)
        if const_expr(_iout):
            # what this iteration prefetched becomes the next one's operands
            _rotate_pipe(a_pipe)
            _rotate_pipe(asc_pipe)

    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        asc_cur = issue_a_scale_ds_read(kt)
        a_cur = issue_a_ds_read(kt % kAStages)
        bs_t = read_b_scale(kt) if const_expr(_bsc_x4) else b_scale_v[kt % kStages]
        for J in range_constexpr(4):
            mfma_cluster(b[kt % kBStages], a_cur, asc_cur, bs_t, J, init=False)

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    s_bsc._view_cache = None
    lds_acc._view_cache = None

    wave_n = wave
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_acc_base, idx * fx.Int32(4)))

    gpu.barrier()

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_base = _global_base_ptr1(arg_aqout)
    scales_per_mr = [None] * M_REPS

    for mr in range_constexpr(M_REPS):
        row_local = fx.Int32(mr * 16) + m_lane

        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_off = (row_local * fx.Int32(BN) + gate_col) * fx.Int32(4)
            up_off = (row_local * fx.Int32(BN) + up_col) * fx.Int32(4)
            gate_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, gate_off)))
            up_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, up_off)))
        result = _silu_mul_batch(gate_vs, up_vs)

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        lm_i = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
        local_max = fx.Float32(_raw(lm_i).bitcast(T.f32))

        e8m0, qscale = _e8m0_from_amax(local_max)
        scales_per_mr[mr] = e8m0

        packed_i32 = _raw(fx.Int32(0))
        qscale_raw = _raw(qscale)
        for w in range_constexpr(4):
            packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                packed_i32,
                _raw(result[2 * w]),
                _raw(result[2 * w + 1]),
                qscale_raw,
                w,
            )
        packed = fx.Int32(packed_i32)

        byte_pos = (
            n_block_idx * fx.Int32(BN_INT // 2)
            + wave_grp * fx.Int32(16)
            + kk * fx.Int32(4)
        )
        out_row = m_row + row_local
        store_off = out_row * fx.Int32(K_G2_HALF) + byte_pos
        llvm.StoreOp(
            _raw(packed),
            _gep1(aqout_base, store_off),
            alignment=4,
            nontemporal=True,
        )

    ascaleout_base = _global_base_ptr1(arg_ascaleout)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        if True:
            for sub in range_constexpr(kSubBlocks):
                chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                dword_off = (
                    chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                    + ku * fx.Int32(64)
                    + wave_grp * fx.Int32(16)
                    + m_lane
                )
                pair_i32 = scales_per_mr[sub * 2 + 0] | (
                    scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                )
                pair_i16 = arith.TruncIOp(T.i16, _raw(pair_i32)).result
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                llvm.StoreOp(
                    pair_i16,
                    _gep1(ascaleout_base, addr),
                    alignment=2,
                )


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL):
    kAStages = 3
    kSubBlocks = BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(
        s_aq_bytes + s_asc_bytes + (_BSC_LDS_BYTES if _BSC_X4 else 0), lds_acc_bytes
    )
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_port(
    BM=128,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    BK=256,
    xcd_swizzle=0,
):
    assert BM == 128, (
        f"mxfp4_gemm1_bm128 is the BM=128 cached path only; got BM={BM}. "
        "Other variants live in mxfp4_gemm1.py."
    )

    assert BN == 256 and BK == 256, f"only BN==BK==256 supported, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _kUnroll = kunroll_for(_K, BK)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _kBS_per_expert_dw = kbs_per_expert_dw_for(_N_OUT, _K)
    _BQ_BYTES = bq_bytes_for(_NE, _N_OUT, _K)
    _BSCALE_BYTES = bscale_bytes_for(_NE, _N_OUT, _K)
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)
    _OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(_INTER)
    _K_G2_HALF = k_g2_half_for(_INTER)

    kAStages, kSubBlocks, kMChunks, lds_bytes = _bm_constants(
        BM, BN, KH_TILE, _K_TILES_TOTAL
    )

    variant_tag = "cached"
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "sep"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}_{gu_tag}"
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm1port_smem_{name_suffix}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            _ng = fx.Int32(_SW * _NUM_N_BLOCKS)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_NUM_N_BLOCKS) + n_block

        if fx.Int32(bx_i32) < bound:
            if const_expr(_SW > 0):
                _tile = _xcd(bx_i32)
            else:
                _tile = bx_i32
            _gemm1_body(
                allocator,
                lds_off,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                _tile,
                lane,
                wave,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                KH_TILE=KH_TILE,
                kAStages=kAStages,
                kSubBlocks=kSubBlocks,
                kMChunks=kMChunks,
                K=_K,
                K_HALF=_K_HALF,
                K_TILES_TOTAL=_K_TILES_TOTAL,
                kUnroll=_kUnroll,
                kAS_per_chunk_dw=_kAS_per_chunk_dw,
                kBS_stride_n0_dw=_kBS_stride_n0_dw,
                kBS_per_expert_dw=_kBS_per_expert_dw,
                BQ_BYTES=_BQ_BYTES,
                BSCALE_BYTES=_BSCALE_BYTES,
                N_OUT=_N_OUT,
                NUM_N_BLOCKS=_NUM_N_BLOCKS,
                OUT_AS_PER_CHUNK_DW=_OUT_AS_PER_CHUNK_DW,
                K_G2_HALF=_K_G2_HALF,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_grid)
        gemm1_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
