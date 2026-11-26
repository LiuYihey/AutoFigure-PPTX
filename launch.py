"""
AutoFigure-Edit Launcher
========================
One-click launcher: starts the web server and opens the browser automatically.

Usage:
    python launch.py

On first run, generates a simple .ico icon file (autofigure_icon.ico)
that can be used when creating a desktop shortcut.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ICON_PATH = BASE_DIR / "autofigure_icon.ico"


def generate_icon() -> None:
    """Generate the .ico file using aggdraw (true anti-aliased rendering).

    Pillow's ImageDraw is aliased (no anti-aliasing) — pixel-drawn icons look
    blurry at every scale. aggdraw uses the AGG library for true anti-aliased
    vector drawing. We render a 1024px master and downscale to each ICO size
    with Lanczos, giving crisp edges at every resolution.
    """
    if ICON_PATH.is_file():
        return  # Already generated, no need to redraw

    sizes = [16, 24, 32, 48, 64, 96, 128, 256]
    master_size = 1024

    try:
        import aggdraw  # type: ignore
        from PIL import Image  # type: ignore

        s = master_size / 256.0
        w = master_size
        img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
        draw = aggdraw.Draw(img)

        # --- Background: near-black rounded square ---
        r = int(56 * s)
        bg_path = (
            f"M {r},0 L {w-r},0 Q {w},0 {w},{r}"
            f" L {w},{w-r} Q {w},{w} {w-r},{w}"
            f" L {r},{w} Q 0,{w} 0,{w-r}"
            f" L 0,{r} Q 0,0 {r},0 Z"
        )
        draw.symbol(
            (0, 0, w, w),
            aggdraw.Symbol(bg_path),
            aggdraw.Brush((14, 17, 22), 255),
        )

        # --- Hairline inner frame (white, low opacity) ---
        inset = int(20 * s)
        fr = int(36 * s)
        x1, y1 = inset, inset
        x2, y2 = w - inset, w - inset
        sw = max(1, int(2 * s))
        frame_path = (
            f"M {x1+fr},{y1} L {x2-fr},{y1} Q {x2},{y1} {x2},{y1+fr}"
            f" L {x2},{y2-fr} Q {x2},{y2} {x2-fr},{y2}"
            f" L {x1+fr},{y2} Q {x1},{y2} {x1},{y2-fr}"
            f" L {x1},{y1+fr} Q {x1},{y1} {x1+fr},{y1} Z"
        )
        draw.symbol(
            (0, 0, w, w),
            aggdraw.Symbol(frame_path),
            aggdraw.Pen((255, 255, 255), sw, 36),
        )

        # --- Minimal node graph (white line + nodes) ---
        pts = [(64, 168), (104, 112), (150, 140), (196, 76)]
        scaled = [int(v * s) for pt in pts for v in pt]
        draw.line(scaled, aggdraw.Pen((236, 241, 246), int(9 * s), 255))

        # --- White nodes (first 3) ---
        nr = int(10 * s)
        brush_white = aggdraw.Brush((236, 241, 246), 255)
        for px, py in [(int(p[0] * s), int(p[1] * s)) for p in pts[:3]]:
            draw.ellipse((px - nr, py - nr, px + nr, py + nr), brush_white)

        # --- Accent node: mint ring with dark center ---
        hx, hy = int(pts[3][0] * s), int(pts[3][1] * s)
        hr = int(15 * s)
        draw.ellipse(
            (hx - hr, hy - hr, hx + hr, hy + hr),
            aggdraw.Brush((94, 234, 212), 255),
        )
        ir = int(6 * s)
        draw.ellipse((hx - ir, hy - ir, hx + ir, hy + ir), aggdraw.Brush((14, 17, 22), 255))

        draw.flush()

        # --- Downscale master to each ICO size with Lanczos ---
        images = [img.resize((sz, sz), Image.LANCZOS) for sz in sizes]
        # CRITICAL: use the largest (256x256) as the primary image. Pillow's ICO
        # writer keys off the primary image's size — if it's 16x16, only that one
        # size gets embedded. Using 256x256 ensures all sizes in `sizes` are saved.
        images[-1].save(
            str(ICON_PATH),
            format="ICO",
            sizes=[(sz, sz) for sz in sizes],
            append_images=images[:-1],
        )
        print(f"[Launcher] Icon generated: {ICON_PATH}")
    except ImportError:
        print("[Launcher] aggdraw/PIL not available, skipping icon generation")
    except Exception as exc:
        print(f"[Launcher] Icon generation failed: {exc}")


def find_python() -> str:
    """Find the Python executable that has server dependencies (fastapi, uvicorn)."""
    # 1. Check AUTOFIGURE_PYTHON env var
    explicit = os.environ.get("AUTOFIGURE_PYTHON")
    if explicit and Path(explicit).is_file():
        return explicit

    # 2. Look for conda "figure" env
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        envs_dir = Path(conda_prefix).parent
        candidate = envs_dir / "figure" / "python.exe"
        if candidate.is_file():
            return str(candidate)

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).parent.parent
        candidate = conda_root / "envs" / "figure" / "python.exe"
        if candidate.is_file():
            return str(candidate)

    # 3. Common conda install locations on Windows
    for base in [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
    ]:
        candidate = base / "envs" / "figure" / "python.exe"
        if candidate.is_file():
            return str(candidate)

    # 4. Dynamic lookup via conda env list
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "figure":
                candidate = Path(parts[1]) / "python.exe"
                if candidate.is_file():
                    return str(candidate)
    except Exception:
        pass

    # 4. Fallback to current Python
    return sys.executable


def main() -> None:
    generate_icon()

    python_exe = find_python()
    print(f"[Launcher] Using Python: {python_exe}")

    # Start server.py as a subprocess
    env = os.environ.copy()
    # Force unbuffered stdout so server.py's "Local access" prints appear promptly
    env["PYTHONUNBUFFERED"] = "1"
    # Don't write .pyc files — prevents stale bytecode causing code/version
    # mismatches after we edit sam3/model_builder.py (seen as "before building
    # model" log from old cached bytecode while source said "after").
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Bind to localhost only — single-user desktop usage, no need to expose the
    # server to the LAN. server.py reads this to set uvicorn's host and prints
    # "Local access: http://127.0.0.1:PORT", so 0.0.0.0 never appears.
    env.setdefault("AUTOFIGURE_HOST", "127.0.0.1")
    # Use mmap=True for loading the 3GB+ SAM3 checkpoint — avoids the native
    # access violation in torch.serialization.load_tensor that occurs with
    # mmap=False (full-read into RAM) on Windows.
    env.setdefault("SAM3_CKPT_LOAD_MMAP", "1")

    proc = subprocess.Popen(
        [python_exe, str(BASE_DIR / "server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
        env=env,
    )

    server_url: str | None = None

    def capture_url():
        nonlocal server_url
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line)
            if server_url is not None:
                continue
            # Match "Local access: http://127.0.0.1:XXXX"
            match = re.search(r"https?://\S+", line)
            if not match:
                continue
            url = match.group(0)
            # Prefer the explicit "Local access" line
            if "Local access" in line:
                server_url = url
            # Fallback: uvicorn's "Uvicorn running on http://HOST:PORT"
            elif "Uvicorn running on" in line or "running on" in line:
                server_url = url

    capture_thread = threading.Thread(target=capture_url, daemon=True)
    capture_thread.start()

    # Wait up to 30 seconds for the server to print its URL
    for _ in range(300):
        if server_url:
            break
        if proc.poll() is not None:
            print("[Launcher] Server process exited before URL was printed.")
            sys.exit(1)
        time.sleep(0.1)

    if server_url:
        print(f"[Launcher] Opening browser: {server_url}")
        # Small delay to ensure the server is ready to accept connections
        time.sleep(0.5)
        webbrowser.open(server_url)
    else:
        print("[Launcher] Could not detect server URL. Please check the output above.")

    # Wait for the server process to finish
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down...")
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
