"""Read per-process field dumps written by the reference instrumentation.

The dump format is written by patches/0002-per-process-field-dump-hook.patch
(components/eamxx/src/share/atm_process/ref_dump.hpp). Each process call
produces <stem>_pre.{bin,json} and <stem>_post.{bin,json}.

Only numpy is required, so this also runs in the project environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_dump(stem: str | Path, names=None) -> tuple[dict, dict[str, np.ndarray]]:
    """Return (metadata, {name: array}) for one dump file stem (without .bin/.json).

    names: only these fields (e.g. to skip HOMME's element fields, which the column
    limit EAMXX_REF_DUMP_NCOLS truncates in the file).
    """
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text())
    raw = np.fromfile(stem.with_suffix(".bin"), dtype="<f8")
    fields = {}
    for f in meta["fields"]:
        if names is not None and f["name"] not in names:
            continue
        arr = raw[f["offset"]: f["offset"] + f["count"]].reshape(f["shape"])
        key = f["name"] if f["name"] not in fields else f"{f['name']}@{f['grid']}"
        fields[key] = arr
    return meta, fields


def read_call(proc_dir: str | Path, step: int, subcycle: int = 0, rank: int = 0):
    """Return (meta_pre, pre, meta_post, post) for one process call."""
    base = Path(proc_dir) / f"s{step}_c{subcycle}_r{rank}"
    mpre, pre = read_dump(f"{base}_pre")
    mpost, post = read_dump(f"{base}_post")
    return mpre, pre, mpost, post


if __name__ == "__main__":
    import sys

    mpre, pre, mpost, post = read_call(sys.argv[1], int(sys.argv[2]))
    print(f"{mpre['proc']} step {mpre['step']} dt={mpre['dt']} {mpre['time']} -> {mpost['time']}")
    for f in mpre["fields"]:
        a, b = pre[f["name"]], post[f["name"]]
        changed = not np.array_equal(a, b, equal_nan=True)
        print(f"  {f['name']:32s} {str(f['dims']):28s} {f['units']:14s} "
              f"{'CHANGED' if changed else '       '} max|pre|={np.nanmax(np.abs(a)) if a.size else 0:.4g}")
