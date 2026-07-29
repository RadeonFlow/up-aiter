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


# B-scale wide load. The preshuffled B-scale for one n0 unit (32 N rows) is
# contiguous along K -- K-tile t sits at byte t*256 within the unit -- so one
# `buffer_load_dwordx4 ... lds` (64 lanes x 16 B = 1024 B) fetches FOUR K-tiles
# at once, cutting the steady-loop B-scale VMEM count from 2/iter to 2 per 4
# iters. LDS is needed only to transpose: the gather lands scales in natural
# order (lane g -> bytes g*16..+15) while the mfma wants lane L to hold scale L
# of one tile, a stride-256 pattern no single VMEM op can express. Same idiom
# as fp4_gemm_4wave's ScaleLoaderLDS.
#
# It was 6.5% SLOWER while the gather and its ds_read were still placed the way
# the old J-outer schedule wanted them -- the gather exposed after all 64 mfma,
# the read sitting between the barrier and the first mfma. With both woven into
# the mfma stream it is now +0.8% (3481 vs 3452 TFLOP/s, three interleaved A/B
# pairs, deltas 28.7 / 29.0 / 29.5 -- small but very repeatable).
# Geometry of the B-scale LDS transpose region.
_BSC_TILES = 4  # K-tiles covered by one dwordx4 gather (1024 B / 256 B)
_BSC_SLOTS = 2  # double buffer over groups of _BSC_TILES
_BSC_WAVE_BYTES = 2 * _BSC_TILES * 256  # 2 mw x 4 tiles x 256 B = 2 KB
_BSC_SLOT_BYTES = 4 * _BSC_WAVE_BYTES  # 4 waves = 8 KB
_BSC_LDS_BYTES = _BSC_SLOTS * _BSC_SLOT_BYTES  # 16 KB


def _lds_scopes(n_a_slots):
    """One #llvm.alias_scope per disjoint LDS region, in a shared domain.

    Regions: one per A slot (n_a_slots of them), then the A-scale area and the
    B-scale gather area, which are separate allocations entirely.

    si-insert-waitcnts asks, for every LDS access, whether it may alias any
    outstanding LDS DMA (SIInsertWaitcnts.cpp:2542). With alias info it checks
    them one by one and waits only on the overlapping ones; without it, it
    waits on all of them at once, which for this loop is a full vmcnt(0)
    immediately ahead of each ds_read. The A tile is triple buffered and
    iteration N reads slot (N+1)%3 while its DMAs fill (N+2)%3, so they never
    overlap -- but the addresses are ptrtoint arithmetic off one 128 KB
    addrspace(3) global, which the backend cannot see through.
    """
    from flydsl._mlir import ir

    dom = '#llvm.alias_scope_domain<id = "gemm1.lds", description = "gemm1 LDS">'
    names = [f"A.{i}" for i in range(n_a_slots)] + ["Asc", "Bsc"]
    return [
        ir.Attribute.parse(f'#llvm.alias_scope<id = "gemm1.{n}", domain = {dom}>')
        for n in names
    ]


def _tag_alias(op, scopes, slot):
    """Mark `op` as touching only LDS region `slot`, and no other."""
    from flydsl._mlir import ir

    op = getattr(op, "owner", op)
    others = [sc for i, sc in enumerate(scopes) if i != slot]
    op.attributes["alias_scopes"] = ir.ArrayAttr.get([scopes[slot]])
    op.attributes["noalias_scopes"] = ir.ArrayAttr.get(others)


def _a_read_order(kMChunks, kSubBlocks):
    """(kind, *idx) for the A / A-scale ds_reads of one iteration, in deadline
    order.

    The mfma run k-outer, i, then J, so A[i,k] is first consumed at mfma
    k*32 + i*4, and A-scale[s] at the first i it covers, i = 2s. Sorting by
    that interleaves the two streams -- asc[1] lands between A[2,0] and A[3,0],
    not after all sixteen A fragments. A[0,*] and asc[0] are excluded: they are
    the next iteration's opening operands, and the caller issues them last so
    they spend the least time sitting in registers. Folding them in here --
    where they would sort to the front on deadline 0 -- measures 0.2% slower
    on all three interleaved A/B pairs.
    """
    out = [(i * 4 + k * 32, ("a", i, k))
           for k in range(2) for i in range(1, kMChunks)]
    out += [(2 * s * 4, ("asc", s)) for s in range(1, kSubBlocks)]
    out.sort(key=lambda x: x[0])
    return [a for _, a in out]


def _a_read_thunks(order, a_nxt, asc_nxt, slot, kt, f_a, f_asc):
    """Turn _a_read_order's list into thunks that store into the two holders."""
    out = []
    for act in order:
        if act[0] == "a":
            out += _store_thunks(f_a, a_nxt,
                                 ((act[1], act[2]), (slot, act[1], act[2])))
        else:
            out += _store_thunks(f_asc, asc_nxt, (act[1], (kt, act[1])))
    return out


def _thunks(fn, *arglists):
    """Bind fn to each argument tuple, as a list of zero-arg thunks.

    Module level on purpose: a lambda defined inside a traced kernel loses its
    captures to the DSL AST rewriter (they silently come back empty), so the
    binding has to happen out here. Same reason fp4_gemm_4wave keeps its
    _g2s_thunks / _s2r_thunks at module scope.
    """
    return [(lambda f=fn, a=args: f(*a)) for args in arglists]


def _store_thunks(fn, holder, *specs):
    """Like _thunks, but each thunk stores fn's result into the holder.

    A spec is (dst, args): dst is an index or an (i, j) pair into holder, and
    args is the tuple passed to fn.
    """
    out = []
    for dst, args in specs:
        def _one(f=fn, h=holder, d=dst, a=args):
            if isinstance(d, tuple):
                h[d[0]][d[1]] = f(*a)
            else:
                h[d] = f(*a)
        out.append(_one)
    return out


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
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4 # 28672 = (7168//32)*16*4  224*128.
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = _buffer_rsrc(arg_ascale, ascale_num)
    bq_rsrc = _buffer_rsrc(arg_bq, BQ_BYTES)
    bscale_rsrc = _buffer_rsrc(arg_bscale, BSCALE_BYTES)

    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,)) # lds_off = 0, kAStages = 3, BM = 128, KH_TILE = 128
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,), # 4 * 28 * 256 = 28KB
    )
    s_bsc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE + kSubBlocks * K_TILES_TOTAL * 256,
        T.i8,
        shape=(_BSC_LDS_BYTES,), # 16KB. 
    )
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,)) # 128 * 256 * 4 = 128KB 所以这里是随便共用的.

    _LDS_SCOPES = _lds_scopes(kAStages)
    _SC_ASC = kAStages       # index of the A-scale region's scope
    _SC_BSC = kAStages + 1   # index of the B-scale region's scope

    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks): # 4
        idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        cached_actual_row.append(
            llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4))) # 存会load的x id.
        )

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(4):
        tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j) # 全局tile号 0..63
        g = tile_il & fx.Int32(1) # gate or up
        n0 = tile_il >> fx.Int32(1) # ith 16.
        col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16) # 定位到col
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF) # col 0, col 16.
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v)) # 每个wave 4个.

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    np_gate = n_block_idx * fx.Int32(BN // 64) + wave # n_bid * 4 + wave
    np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base = []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)

    accm = [[None] * 4 for _ in range(kMChunks)]
    # B is triple-buffered. With only 2 buffers the load that refills b[slot_b]
    # targets the very buffer this iteration's mfma are still reading, so bld[j]
    # could not be issued before the last mfma that reads B[j] (WAR) -- which
    # pinned all 8 B loads to fixed points. With 3 buffers the load writes a
    # slot nobody is reading and can go anywhere, which is what lets them be
    # spread evenly through the mfma stream.
    kBStages = 3
    b = [[[None, None] for _ in range(4)] for _ in range(kBStages)]

    def issue_a_load_lds_one(slot, kt, sub):
        """One A global->LDS DMA step: 32 M rows of tile `kt` into `slot`.

        Reads the row out of cached_actual_row, loaded once before the loop.
        Re-issuing the `arg_mind` load here instead makes the address depend on
        it, and the compiler then emits `s_waitcnt vmcnt(0)` before every one
        of these -- 105 of them, serializing the whole pipeline.
        """
        lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
        mask = _lds_swizzle_mask(lds_row + lane_div_8)
        voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + fx.Int32(
            cached_actual_row[sub]
        ) * fx.Int32(K_HALF)
        base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_aq.get()))
        off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
        _dma = rocdl.raw_ptr_buffer_load_lds(
            aq_rsrc,
            _lds_ptr3(base_i32, off),
            fx.Int32(16),
            voffset,
            fx.Int32(kt * KH_TILE),
            fx.Int32(0),
            fx.Int32(0),
        )
        _tag_alias(_dma, _LDS_SCOPES, slot)

    def issue_a_load_lds(slot, kt):
        """All kSubBlocks steps of one tile, for the prologue."""
        for sub in range_constexpr(kSubBlocks):
            issue_a_load_lds_one(slot, kt, sub)

    def issue_a_ds_read_one(slot, i, k):
        """One A fragment ds_read (M-block i, k-half k) -> i32x4."""
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
        lds_row = lane_mod_16 + fx.Int32(i * 16)
        off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE) + lds_col
        val = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
        _tag_alias(val.owner, _LDS_SCOPES, slot)
        return val

    def issue_a_scale_ds_read_one(kt, sub):
        """One A-scale ds_read -> i32."""
        base_ptr = _lds_base_ptr3(s_asc.get())
        lds_dw = (
            fx.Int32(sub * kAS_per_chunk_dw)
            + fx.Int32(kt * 64)
            + lane_div_16 * fx.Int32(16)
            + lane_mod_16
        )
        val = llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4)))
        _tag_alias(val.owner, _LDS_SCOPES, _SC_ASC)
        return val

    def issue_a_ds_read(slot):
        """All A fragments of one slot, for the prologue and the drain."""
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            for i in range_constexpr(kMChunks):
                a[i][k] = issue_a_ds_read_one(slot, i, k)
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
            _d = rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            _tag_alias(_d, _LDS_SCOPES, _SC_ASC)
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                _d = rocdl.raw_ptr_buffer_load_lds(
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
                _tag_alias(_d, _LDS_SCOPES, _SC_ASC)

    def issue_a_scale_ds_read(kt):
        """All kSubBlocks A-scale reads of tile kt."""
        return [issue_a_scale_ds_read_one(kt, sub) for sub in range(kSubBlocks)]

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
        Only legal to place freely because B is triple-buffered: with 2
        buffers this write would race the mfma still reading the same slot."""
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
            # byte offset of this group inside the n0 unit = grp*_BSC_TILES*256
            s_off = rocdl.readfirstlane(
                T.i32, _raw(b_scale_s_base[mw] + fx.Int32(grp * _BSC_TILES * 256))
            )
            _d = rocdl.raw_ptr_buffer_load_lds(
                bscale_rsrc,
                _lds_ptr3(bsc_lds_base, lds_off_b),
                fx.Int32(16),
                bsc_v,
                s_off,
                fx.Int32(0),
                fx.Int32(0),
            )
            _tag_alias(_d, _LDS_SCOPES, _SC_BSC)

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
            _bv = llvm.load(T.i32, _gep3(base_ptr, off))
            _tag_alias(_bv.owner, _LDS_SCOPES, _SC_BSC)
            out[mw] = fx.Int32(_bv)
        return out

    mfma_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):

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
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                    )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb]
                    )
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][1], bJ1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb]
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][1], bJ1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb]
                )

    def mfma_iouter(b_slot, a, a_scale, bs_slot, init, interleave, stride):
        """Emit all 64 mfma in i-outer order (k, i, J), issuing one thunk from
        `interleave` every `stride` mfma.

        i-outer is what makes the weave possible: A[i,k] is shared by the 4
        consecutive mfma of one (k,i), so the ds_read of the next fragment only
        has to land 4 mfma later. With J outermost, J=0's 16 mfma already touch
        all 8 M-blocks and every A ds_read has to complete before the first
        mfma. It also stretches one accumulator's reuse from 8 mfma to 32.
        """
        nth = 0
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
                    # matches mfma_cluster: the B-scale slot
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
                    if nth < len(interleave) and (n % stride) == 0:
                        interleave[nth]()
                        nth += 1
                    n += 1
        # Drain whatever the 64 mfma did not reach. A `while` here would be
        # rewritten into scf.while by the DSL tracer; the bounds are plain
        # python ints, so slice instead.
        for _t in interleave[nth:]:
            _t()


    # The A / A-scale ds_reads of one iteration, in the order their consumers
    # need them.
    _A_READ_ORDER = _a_read_order(kMChunks, kSubBlocks)

    # One thunk every _IOUT_STRIDE mfma. 32 thunks over 64 mfma at stride 2
    # spread across the first ~25 and leave the rest a clean mfma run, and no
    # two VMEM ops land closer than 4 mfma apart -- back-to-back buffer_loads
    # queue on L1 and their issue latency blows up.
    _IOUT_STRIDE = 2

    # Steady fence. An iteration issues 4 albd then 8 B loads, and vmcnt(N)
    # retires oldest-first, so waiting for <= 8 in flight retires exactly the
    # 4 albd. That is what the barrier needs, because right after it the other
    # 3 waves start ds_read-ing that A tile. It is also the loosest legal
    # value: only 12 VMEM are ever in flight, so vmcnt(9) or looser leaves an
    # albd unlanded -- swept, and cosine degrades monotonically from vmcnt(9)
    # on (0.0133, 0.0239, 0.0561, 0.1043 for 9..12).
    #
    # Note this is NOT the same budget as the compiler's own vmcnt(12)/(11)/(10)
    # guarding the B operands. Those cover a bld issued two iterations before
    # its use, so they get a full iteration of slack. The albd sit at the FRONT
    # of the 12, so their slack is only what follows them. Same counter, two
    # different distances. See resource_inspect/gemm1_fence_explained.txt.
    _IOUT_POST_ALBD = 2 * 4  # the 8 B loads that follow the last albd

    # ---- prologue ---------------------------------------------------------
    # A-scale for the whole K range is resident in LDS, so it loads once.
    issue_a_scale_load()

    # A first, then B, rather than interleaving them per tile. Same reason the
    # steady schedule puts the albd ahead of the B loads: vmcnt retires
    # oldest-first, so grouping the albd at the front lets iteration 0's fence
    # be the same vmcnt(8) as every other iteration. Interleaving would leave
    # only 5 VMEM behind the last albd and force a tighter first fence.
    # sched_barrier stops the compiler from mixing the two groups back up.
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
    rocdl.sched_barrier(0)
    # Group 0 first so it is the oldest VMEM in the prologue and the steady
    # fence has certainly retired it before iteration 0 reads it.
    issue_b_scale_gather(0)
    for K_C in range_constexpr(kStages):
        for j in range_constexpr(4):
            issue_b_load_j(b[K_C], K_C, j)

    # Iteration 0's A / A-scale fragments. Every later iteration gets these
    # from the previous one's weave, so the steady loop never has a ds_read
    # ahead of its first mfma. Needs a full barrier first to publish slot 0.
    a_pipe = [None, None]
    asc_pipe = [None, None]
    bsc_pipe = [None, None]
    gpu.barrier()
    asc_pipe[0] = issue_a_scale_ds_read(0)
    a_pipe[0] = issue_a_ds_read(0)
    bsc_pipe[0] = read_b_scale(0)

    for OFFSET in range_constexpr(kUnroll): #  28 主循环.
        K_C = kStages + OFFSET # 2 + i
        read_slot = OFFSET % kAStages # 
        write_slot = K_C % kAStages
        slot_b = OFFSET % kBStages
        # Tile K_C = OFFSET+kStages is the one being prefetched. With kBStages=3
        # its slot differs from the slot being read (OFFSET%3), so the refill has
        # no WAR against this iteration's mfma and can be issued anywhere.
        write_b = K_C % kBStages
        # Steady fence: A read from read_slot was written by iter OFFSET-2
        # The fence must retire the previous iteration's albd: the barrier right
        # after it is what publishes that A slot to the other 3 waves. vmcnt
        # counts ops and retires oldest-first, and the schedule issues the 4
        # albd ahead of all 8 B loads, so waiting for <= 8 in flight retires
        # exactly the albd. A bare s_barrier would not do -- it synchronises
        # program position and does not drain VMEM at all.
        llvm.InlineAsmOp(
            None, [], f"s_waitcnt vmcnt({_IOUT_POST_ALBD})", "", has_side_effects=True
        )
        rocdl.s_barrier()
        # A / A-scale for this iteration were prefetched by the previous one
        # (or by the prologue), so the mfma stream starts right after the
        # barrier with no ds_read block in front of it.
        a_cur = a_pipe[0]
        asc_cur = asc_pipe[0]
        # Read out of LDS by the previous iteration's weave, like the A
        # fragments -- so no ds_read sits between the barrier and the first
        # mfma. Each wave owns its own region of s_bsc, so there is no
        # cross-wave ordering to respect here.
        bs_cur = bsc_pipe[0]
        a_nxt = [[None, None] for _ in range(kMChunks)]
        asc_nxt = [None] * kSubBlocks
        a_pipe[1], asc_pipe[1] = a_nxt, asc_nxt
        nxt_slot = (OFFSET + 1) % kAStages
        # Clamp: the last iteration would prefetch tile K_TILES_TOTAL. The
        # drain re-reads what it needs, so the extra read is idempotent.
        nxt_kt = min(K_C - kStages + 1, K_TILES_TOTAL - 1)

        # The schedule for this iteration's 64 mfma, in issue order. One
        # thunk goes out every _IOUT_STRIDE mfma.
        #
        #   A1  the next iteration's A fragments and A-scale (LDS -> reg)
        #   A2  this iteration's A refill        (global -> LDS, async DMA)
        #   B2  the B fragments two iterations out    (global -> VGPR)
        #
        # Order matters twice over. The A1 reads come first because they are
        # the ones with a deadline (the next iteration's first mfma), and
        # the 4 albd sit ahead of all 8 B loads because the steady fence has
        # to cover them -- vmcnt counts ops, oldest first, so the fence
        # value is exactly the number of VMEM ops issued after the last
        # albd. Putting the B loads first would drive it to vmcnt(0).
        il = (
            _a_read_thunks(_A_READ_ORDER, a_nxt, asc_nxt, nxt_slot, nxt_kt,
                           issue_a_ds_read_one, issue_a_scale_ds_read_one)
            + _thunks(issue_a_load_lds_one,
                      *[(write_slot, K_C, s) for s in range(kSubBlocks)])
            + _store_thunks(
                issue_a_scale_ds_read_one, asc_nxt, (0, (nxt_kt, 0)),
            )
            + _store_thunks(
                issue_a_ds_read_one, a_nxt,
                ((0, 0), (nxt_slot, 0, 0)), ((0, 1), (nxt_slot, 0, 1)),
            )
            + _store_thunks(read_b_scale, bsc_pipe, (1, (OFFSET + 1,)))
            + _thunks(issue_b_load_one,
                      *[(b[write_b], K_C, j, h)
                        for h in range(2) for j in range(4)])
        )
        # One gather covers _BSC_TILES tiles, so it only runs on the first
        # iteration of a group -- which gives the next group a full group of
        # VMEM to land behind. Issuing it on the LAST iteration would leave one
        # iteration of lead and the fence would retire nothing (nan). It goes
        # after the B loads so it does not move the last albd, which is what
        # sets the fence; being the newest op in flight costs it nothing, since
        # nobody reads it for another four iterations.
        _grp = OFFSET // _BSC_TILES + 1
        if const_expr(OFFSET % _BSC_TILES == 0
                      and _grp * _BSC_TILES < K_TILES_TOTAL):
            il = il + _thunks(issue_b_scale_gather, (_grp,))
        mfma_iouter(
            b[slot_b], a_cur, asc_cur, bs_cur, (OFFSET == 0),
            il, _IOUT_STRIDE,
        )
        # what this iteration prefetched becomes the next one's operands
        a_pipe[0], asc_pipe[0] = a_pipe[1], asc_pipe[1]
        bsc_pipe[0] = bsc_pipe[1]

    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        asc_cur = issue_a_scale_ds_read(kt)
        a_cur = issue_a_ds_read(kt % kAStages)
        bs_t = read_b_scale(kt)
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
        s_aq_bytes + s_asc_bytes + _BSC_LDS_BYTES, lds_acc_bytes
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
