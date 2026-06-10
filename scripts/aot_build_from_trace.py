# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AOT-precompile the aiter JIT modules recorded by a build trace.

Workflow
--------
1. Record what your workload actually compiles by running it once with the
   trace enabled (the trace hook lives at the top of build_module()):

       export AITER_TRACE_BUILD=$HOME/aiter/aiter/jit/build/build_trace.jsonl
       # ... run your real serving / benchmark workload ...

   Every JIT compile appends one fully-resolved build_module() invocation to
   that JSONL file (one JSON object per line). Parametrized modules (mha /
   moe_ck2stages / ...) resolve their shapes into a distinct md_name *before*
   reaching build_module, so each variant is captured precisely.

2. Replay the trace ahead of time, in parallel:

       python scripts/aot_build_from_trace.py \
           --trace $HOME/aiter/aiter/jit/build/build_trace.jsonl --jobs 4

Each record is replayed by calling build_module(**record) verbatim -- no
gen_func / shape resolution is re-run -- so the result is identical to what
the lazy JIT path would have produced. Modules already protect themselves
with a per-md_name lock, so parallel replay across processes is safe.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Keys we expect in each trace record == build_module() kwargs.
_BUILD_KEYS = {
    "md_name",
    "srcs",
    "flags_extra_cc",
    "flags_extra_hip",
    "blob_gen_cmd",
    "extra_include",
    "extra_ldflags",
    "verbose",
    "is_python_module",
    "is_standalone",
    "torch_exclude",
    "third_party",
    "hipify",
}


# Must match core.AITER_ROOT_PLACEHOLDER: the token the trace writer uses in
# place of the absolute repo root so the trace is portable.
AITER_ROOT_PLACEHOLDER = "${AITER_ROOT}"


def _aiter_root() -> str:
    """Repo root of *this* checkout (scripts/ lives directly under it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def _restore_paths(obj, root: str):
    """Inverse of core._portable_paths: substitute the placeholder back with
    the local repo root so records replay against this checkout."""
    if isinstance(obj, str):
        return obj.replace(AITER_ROOT_PLACEHOLDER, root)
    if isinstance(obj, list):
        return [_restore_paths(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: _restore_paths(v, root) for k, v in obj.items()}
    return obj


def _default_trace_path() -> str:
    env = os.environ.get("AITER_TRACE_BUILD")
    if env:
        return env
    return os.path.join(_aiter_root(), "aiter", "jit", "build", "build_trace.jsonl")


def load_records(trace_path: str):
    """Read JSONL trace, dedup by md_name (last wins). Returns list of dicts."""
    if not os.path.exists(trace_path):
        sys.exit(f"[aot] trace file not found: {trace_path}")
    root = _aiter_root()
    by_name = {}
    n_lines = 0
    with open(trace_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[aot] skip malformed line {lineno}: {e}", file=sys.stderr)
                continue
            md = rec.get("md_name")
            if not md:
                print(f"[aot] skip line {lineno}: no md_name", file=sys.stderr)
                continue
            # keep only the known build_module kwargs; ignore extra fields,
            # and restore the local repo root for portable (${AITER_ROOT}) paths
            rec = _restore_paths(rec, root)
            by_name[md] = {k: v for k, v in rec.items() if k in _BUILD_KEYS}
    print(
        f"[aot] read {n_lines} trace record(s) -> {len(by_name)} unique module(s)"
    )
    return list(by_name.values())


def _build_one(record: dict, max_jobs_per_module: int):
    """Worker entry: replay a single build_module() call. Runs in a subprocess."""
    # Never re-record while replaying.
    os.environ.pop("AITER_TRACE_BUILD", None)
    # Bound ninja parallelism per module so N parallel modules don't
    # oversubscribe the box. check_and_set_ninja_worker() treats MAX_JOBS as
    # an upper bound (it only ever lowers it), so this is a hard cap.
    os.environ["MAX_JOBS"] = str(max(1, max_jobs_per_module))

    from aiter.jit.core import build_module  # imported in the worker

    md_name = record["md_name"]
    t0 = time.perf_counter()
    try:
        build_module(**record)
        return (md_name, True, None, time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - report, don't abort the whole run
        import traceback

        return (md_name, False, traceback.format_exc(), time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default=_default_trace_path(),
                    help="path to the JSONL build trace (default: "
                         "$AITER_TRACE_BUILD or aiter/jit/build/build_trace.jsonl)")
    ap.add_argument("-j", "--jobs", type=int, default=32,
                    help="number of modules to compile in parallel (default: 4)")
    ap.add_argument("--max-jobs-per-module", type=int, default=8,
                    help="ninja workers per module; 0 => max(1, cpu_count // jobs)")
    ap.add_argument("--filter", default=None,
                    help="only build modules whose md_name matches this regex")
    ap.add_argument("--list", action="store_true",
                    help="list the modules that would be built, then exit")
    args = ap.parse_args()

    records = load_records(args.trace)

    if args.filter:
        import re

        pat = re.compile(args.filter)
        records = [r for r in records if pat.search(r["md_name"])]
        print(f"[aot] {len(records)} module(s) match filter {args.filter!r}")

    if not records:
        sys.exit("[aot] nothing to build")

    if args.list:
        for r in sorted(records, key=lambda r: r["md_name"]):
            print(r["md_name"])
        return

    per_module = args.max_jobs_per_module
    if per_module <= 0:
        per_module = max(1, (os.cpu_count() or 1) // max(1, args.jobs))

    print(
        f"[aot] building {len(records)} module(s): "
        f"jobs={args.jobs}, max_jobs_per_module={per_module}"
    )
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(_build_one, r, per_module): r["md_name"] for r in records
        }
        done = 0
        for fut in as_completed(futs):
            md_name, ok, err, elapsed = fut.result()
            done += 1
            status = "\033[32mOK\033[0m" if ok else "\033[31mFAIL\033[0m"
            print(f"[aot] ({done}/{len(records)}) {status} {md_name} "
                  f"({elapsed:.1f}s)")
            if not ok:
                print(err, file=sys.stderr)
            results.append((md_name, ok, elapsed))

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]
    print(f"\n[aot] done in {time.perf_counter() - t0:.1f}s: "
          f"{len(ok)} ok, {len(bad)} failed")
    if bad:
        print("[aot] failed modules: " + ", ".join(m for m, _, _ in bad),
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
