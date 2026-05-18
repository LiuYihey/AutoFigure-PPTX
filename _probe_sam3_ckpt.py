"""Standalone probe for the SAM3 checkpoint loading fix.

Reproduces the new order used in model_builder.build_sam3_image_model:
  1) torch.load (small RSS)
  2) allocate a ~3GB blob to simulate the empty model graph
  3) free everything

Crucial: tests that torch.load itself doesn't AV regardless of weights_only,
and reports RSS so we can see peak memory.
"""

from __future__ import annotations

import gc
import hashlib
import io as _io
import os
import sys
import time
import zipfile

import torch

try:
    import psutil  # type: ignore

    _PROC = psutil.Process()
except Exception:
    _PROC = None


def rss_gib() -> str:
    if _PROC is None:
        return "n/a"
    return f"{_PROC.memory_info().rss / (1024**3):.2f}GiB"


def log(msg: str) -> None:
    print(f"[probe RSS={rss_gib()}] {msg}", flush=True)


def inspect(path: str) -> None:
    size = os.path.getsize(path)
    log(f"size = {size} bytes ({size / (1024**3):.3f} GiB)")

    h = hashlib.sha256()
    read_n = 0
    head_limit = 64 * 1024 * 1024
    with open(path, "rb") as f:
        while read_n < head_limit:
            chunk = f.read(min(1024 * 1024, head_limit - read_n))
            if not chunk:
                break
            h.update(chunk)
            read_n += len(chunk)
    log(f"sha256(first {read_n // (1024 * 1024)} MiB) = {h.hexdigest()}")

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                log(f"!! testzip() corrupt member: {bad!r}")
            else:
                members = zf.namelist()
                log(f"zip ok, {len(members)} members, first={members[0]!r}")
    except zipfile.BadZipFile as e:
        log(f"!! BadZipFile: {e}")


def safe_load(path: str, weights_only: bool):
    last_err = None
    for label, opener in (
        ("path", lambda: path),
        ("file-object", lambda: open(path, "rb")),
        ("BytesIO(full read)", lambda: _io.BytesIO(open(path, "rb").read())),
    ):
        t = time.perf_counter()
        try:
            log(f"torch.load via {label} (weights_only={weights_only}) ...")
            arg = opener()
            try:
                obj = torch.load(arg, map_location="cpu", weights_only=weights_only)
            finally:
                if hasattr(arg, "close"):
                    try:
                        arg.close()
                    except Exception:
                        pass
            log(f"torch.load via {label} OK in {time.perf_counter() - t:.1f}s")
            return obj
        except Exception as e:
            last_err = e
            log(f"torch.load via {label} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"all strategies failed; last={last_err}")


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()

    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\___Desktop___\AutoFigure-Edit\models\sam3\sam3.pt"
    wo_arg = sys.argv[2].lower() if len(sys.argv) > 2 else "false"
    weights_only = wo_arg in ("1", "true", "yes")

    log(f"torch={torch.__version__}  python={sys.version.split()[0]}  win32={sys.platform=='win32'}")
    log(f"checkpoint = {path}")

    inspect(path)

    # --- Step 1: new order -- load BEFORE building model ---
    log("=== NEW ORDER: load checkpoint first ===")
    ckpt = safe_load(path, weights_only=weights_only)
    log(f"loaded type={type(ckpt).__name__}")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    n = len(ckpt) if hasattr(ckpt, "__len__") else -1
    log(f"ckpt has {n} top-level tensors")

    # measure total bytes
    total = 0
    sample_dtypes = set()
    for k, v in list(ckpt.items())[:5000]:
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
            sample_dtypes.add(str(v.dtype))
    log(f"approx tensor bytes (sampled) = {total / (1024**3):.2f} GiB  dtypes={sample_dtypes}")

    # --- Step 2: simulate building the empty model graph AFTER loading ---
    log("=== simulating model build AFTER ckpt load (alloc 3 GiB of empty fp32) ===")
    dummy = torch.empty(int(3.0 * 1024 * 1024 * 1024 // 4), dtype=torch.float32)
    log(f"dummy shape={tuple(dummy.shape)} dtype={dummy.dtype}")
    log("freeing dummy + ckpt")
    del dummy
    ckpt.clear()
    del ckpt
    gc.collect()
    log("done; if you reached this line there is no native crash.")
