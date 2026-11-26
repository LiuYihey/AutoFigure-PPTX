from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
import mimetypes
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def _find_best_python() -> str:
    """Find the best Python executable: prefer figure conda env, fallback to sys.executable."""
    explicit = os.environ.get("AUTOFIGURE_PYTHON")
    if explicit and Path(explicit).is_file():
        return explicit

    # Try to find figure conda env Python
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).parent.parent
        for env_name in ["figure"]:
            candidates = [
                conda_root / "envs" / env_name / "python.exe",
                conda_root / "envs" / env_name / "bin" / "python",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

    # Also check CONDA_PREFIX siblings
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        envs_dir = Path(conda_prefix).parent
        for env_name in ["figure"]:
            for suffix in ["python.exe", "bin/python"]:
                candidate = envs_dir / env_name / suffix
                if candidate.is_file():
                    return str(candidate)

    return sys.executable


PYTHON_EXECUTABLE = _find_best_python()

DEFAULT_SAM_PROMPT = "icon, illustration, person, robot, machine, device, animal, Signal, MRI"
DEFAULT_PLACEHOLDER_MODE = "label"
DEFAULT_MERGE_THRESHOLD = 0.3
DEFAULT_MAX_BOX_AREA_RATIO = 0.1
DEFAULT_SAM_MIN_SCORE = 0.2

SVG_EDIT_CANDIDATES = [
    ("vendor/svg-edit/editor/index.html", WEB_DIR / "vendor" / "svg-edit" / "editor" / "index.html"),
    ("vendor/svg-edit/editor.html", WEB_DIR / "vendor" / "svg-edit" / "editor.html"),
    ("vendor/svg-edit/index.html", WEB_DIR / "vendor" / "svg-edit" / "index.html"),
]


def _resolve_svg_edit_path() -> tuple[bool, str | None]:
    for rel, path in SVG_EDIT_CANDIDATES:
        if path.is_file():
            return True, f"/{rel}"
    return False, None


@dataclass
class Job:
    job_id: str
    output_dir: Path
    process: subprocess.Popen
    queue: queue.Queue
    log_path: Path
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    seen: set[str] = field(default_factory=set)
    done: bool = False

    def push(self, event: str, data: dict) -> None:
        self.queue.put({"event": event, "data": data})

    def write_log(self, stream: str, line: str) -> None:
        with self.log_lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stream}] {line}\n")


class RunRequest(BaseModel):
    method_text: str = Field(..., min_length=1)
    image_model: Optional[str] = None
    svg_model: Optional[str] = None
    fix_svg_model: Optional[str] = None
    image_provider: str = "gemini"
    image_api_key: str = ""
    image_base_url: str = ""
    svg_provider: str = "gemini"
    svg_api_key: str = ""
    svg_base_url: str = ""
    fix_svg_provider: str = "gemini"
    fix_svg_api_key: str = ""
    fix_svg_base_url: str = ""
    sam_prompt: Optional[str] = None
    sam_backend: Optional[str] = None
    sam_api_key: Optional[str] = None
    sam_max_masks: Optional[int] = None
    sam_min_score: Optional[float] = None
    placeholder_mode: Optional[str] = None
    merge_threshold: Optional[float] = None
    max_box_area_ratio: Optional[float] = None
    optimize_iterations: Optional[int] = None
    reference_image_path: Optional[str] = None
    input_image_path: Optional[str] = None
    resume_dir: Optional[str] = None
    stop_after: Optional[int] = None


app = FastAPI()

JOBS: dict[str, Job] = {}


@app.get("/api/config")
def get_config() -> JSONResponse:
    available, rel_path = _resolve_svg_edit_path()
    return JSONResponse({"svgEditAvailable": available, "svgEditPath": rel_path})


@app.post("/api/run")
def run_job(req: RunRequest) -> JSONResponse:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(BASE_DIR / "autofigure2.py"),
        "--method_text",
        req.method_text,
        "--output_dir",
        str(output_dir),
    ]

    if req.image_model:
        cmd += ["--image_model", req.image_model]
    if req.svg_model:
        cmd += ["--svg_model", req.svg_model]
    if req.image_provider:
        cmd += ["--image_provider", req.image_provider]
    if req.image_api_key:
        cmd += ["--image_api_key", req.image_api_key]
    if req.image_base_url:
        cmd += ["--image_base_url", req.image_base_url]
    if req.svg_provider:
        cmd += ["--svg_provider", req.svg_provider]
    if req.svg_api_key:
        cmd += ["--svg_api_key", req.svg_api_key]
    if req.svg_base_url:
        cmd += ["--svg_base_url", req.svg_base_url]
    fix_svg_provider = req.fix_svg_provider or req.svg_provider or ""
    fix_svg_api_key = req.fix_svg_api_key or req.svg_api_key or ""
    fix_svg_base_url = req.fix_svg_base_url or req.svg_base_url or ""
    fix_svg_model = req.fix_svg_model or req.svg_model or ""
    if fix_svg_provider:
        cmd += ["--fix_svg_provider", fix_svg_provider]
    if fix_svg_api_key:
        cmd += ["--fix_svg_api_key", fix_svg_api_key]
    if fix_svg_base_url:
        cmd += ["--fix_svg_base_url", fix_svg_base_url]
    if fix_svg_model:
        cmd += ["--fix_svg_model", fix_svg_model]

    sam_prompt = req.sam_prompt or DEFAULT_SAM_PROMPT
    placeholder_mode = req.placeholder_mode or DEFAULT_PLACEHOLDER_MODE
    merge_threshold = (
        req.merge_threshold if req.merge_threshold is not None else DEFAULT_MERGE_THRESHOLD
    )
    max_box_area_ratio = (
        req.max_box_area_ratio if req.max_box_area_ratio is not None else DEFAULT_MAX_BOX_AREA_RATIO
    )
    sam_min_score = (
        req.sam_min_score if req.sam_min_score is not None else DEFAULT_SAM_MIN_SCORE
    )

    cmd += ["--sam_prompt", sam_prompt]
    cmd += ["--placeholder_mode", placeholder_mode]
    cmd += ["--merge_threshold", str(merge_threshold)]
    cmd += ["--max_box_area_ratio", str(max_box_area_ratio)]
    cmd += ["--min_score", str(sam_min_score)]
    if req.sam_backend:
        cmd += ["--sam_backend", req.sam_backend]
    if req.sam_api_key:
        cmd += ["--sam_api_key", req.sam_api_key]
    if req.sam_max_masks is not None:
        cmd += ["--sam_max_masks", str(req.sam_max_masks)]
    if req.optimize_iterations is not None:
        cmd += ["--optimize_iterations", str(req.optimize_iterations)]

    reference_path = req.reference_image_path
    if reference_path:
        reference_path = (
            str((BASE_DIR / reference_path).resolve())
            if not Path(reference_path).is_absolute()
            else reference_path
        )
        cmd += ["--reference_image_path", reference_path]

    input_image_path = req.input_image_path
    if input_image_path:
        input_image_path = (
            str((BASE_DIR / input_image_path).resolve())
            if not Path(input_image_path).is_absolute()
            else input_image_path
        )
        cmd += ["--input_image", input_image_path]

    if req.resume_dir:
        resume_path = req.resume_dir
        if not Path(resume_path).is_absolute():
            resume_path = str((OUTPUTS_DIR / resume_path).resolve())
        cmd += ["--resume_dir", resume_path]

    if req.stop_after is not None:
        cmd += ["--stop_after", str(req.stop_after)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # 子进程若段错误，向 stderr 打印 Python/C 栈（便于排查 SAM3 / torch 原生崩溃）
    env.setdefault("PYTHONFAULTHANDLER", "1")

    log_path = output_dir / "run.log"
    log_path.write_text(
        f"[meta] python={PYTHON_EXECUTABLE}\n[meta] cmd={' '.join(cmd)}\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(BASE_DIR),
    )

    job = Job(
        job_id=job_id,
        output_dir=output_dir,
        process=process,
        queue=queue.Queue(),
        log_path=log_path,
    )
    JOBS[job_id] = job

    monitor_thread = threading.Thread(target=_monitor_job, args=(job,), daemon=True)
    monitor_thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/upload")
async def upload_reference(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        ext = ".png"

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    name = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / name
    out_path.write_bytes(data)

    rel_path = out_path.relative_to(BASE_DIR).as_posix()
    return JSONResponse(
        {"path": rel_path, "url": f"/api/uploads/{name}", "name": file.filename}
    )


@app.get("/api/events/{job_id}")
def stream_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        while True:
            try:
                item = job.queue.get(timeout=1.0)
            except queue.Empty:
                if job.done:
                    break
                continue
            if item.get("event") == "close":
                break
            yield _format_sse(item["event"], item["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/artifacts/{job_id}/{path:path}")
def get_artifact(job_id: str, path: str) -> Response:
    job = JOBS.get(job_id)
    if job:
        output_dir = job.output_dir
    else:
        # 允许刷新/重启后仅通过磁盘上的 outputs/<job_id>/ 访问（画布页会拉取 final.svg）
        output_dir = OUTPUTS_DIR / job_id
        if not output_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")

    candidate = (output_dir / path).resolve()
    if not str(candidate).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # 为了避免运行中被持续追加的文件 (如 run.log) 
    # 被 FastAPI/Starlette 的 FileResponse 预计算出较小的 Content-Length
    # 导致读取时传输的数据大于 Content-Length 而触发 h11._util.LocalProtocolError
    # 这里直接将较小文件（<15MB）读入内存作为普通 Response 返回
    size_limit = 15 * 1024 * 1024
    if candidate.stat().st_size < size_limit:
        mime_type, _ = mimetypes.guess_type(candidate)
        return Response(content=candidate.read_bytes(), media_type=mime_type)
    
    return FileResponse(candidate)


class SaveSvgRequest(BaseModel):
    svg: str


@app.post("/api/save-svg/{job_id}")
async def save_svg(job_id: str, req: SaveSvgRequest) -> JSONResponse:
    job = JOBS.get(job_id)
    if job:
        output_dir = job.output_dir
    else:
        # Job may no longer be in memory (server restart / completed) — resolve from disk
        output_dir = OUTPUTS_DIR / job_id
        if not output_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")
    svg_path = output_dir / "final.svg"
    svg_path.write_text(req.svg, encoding="utf-8")
    return JSONResponse({"ok": True, "path": str(svg_path)})


class ExportPptxRequest(BaseModel):
    svg: str


# ── Native (DrawingML) PPTX export via svg2ooxml ──────────────────────────────
# svg2ooxml requires Python >= 3.13 (PEP 695 type statements). The main figure
# conda env is Python 3.10, so we spawn a dedicated Python 3.13 interpreter
# that has svg2ooxml installed and pipe the SVG string in via stdin.
_PY313_CANDIDATES = [
    r"C:\Python313\python.exe",
    r"C:\Python3.13\python.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python313\python.exe"),
]


def _find_python313() -> str | None:
    """Return the first existing Python 3.13 interpreter path, or None."""
    # Allow override via env var
    explicit = os.environ.get("AUTOFIGURE_PY313")
    if explicit and Path(explicit).is_file():
        return explicit
    for cand in _PY313_CANDIDATES:
        if Path(cand).is_file():
            return cand
    # Last resort: check PATH for a python that reports version 3.13
    for exe_name in ("python3.13", "python"):
        found = shutil.which(exe_name)
        if not found:
            continue
        try:
            out = subprocess.check_output(
                [found, "-c", "import sys; print(sys.version_info[:2])"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
            if out.startswith("(3, 13)"):
                return found
        except Exception:
            continue
    return None


@app.post("/api/export-pptx/{job_id}")
async def export_pptx(job_id: str, req: ExportPptxRequest) -> Response:
    """Export SVG to PPTX as native DrawingML shapes & text via svg2ooxml.

    Every ``<rect>``/``<text>``/``<path>`` in the SVG becomes a first-class
    ``<p:sp>`` shape with ``<a:prstGeom>`` and ``<p:txBody>``.  Text remains
    directly editable (font, size, content) and shape geometry / colors /
    positions are preserved at EMU precision — no "Convert to Shapes" step
    needed in PowerPoint.

    Implementation: spawn ``svg2pptx_bridge.py`` under Python 3.13, feed the
    SVG via stdin, write the PPTX to a temp file, then stream it back.
    """
    if not req.svg or not req.svg.strip():
        raise HTTPException(status_code=400, detail="SVG content is required")

    job = JOBS.get(job_id)
    if job:
        output_dir = job.output_dir
    else:
        output_dir = OUTPUTS_DIR / job_id
        if not output_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")

    py313 = _find_python313()
    if not py313:
        raise HTTPException(
            status_code=500,
            detail=(
                "Python 3.13 not found. svg2ooxml requires Python >= 3.13. "
                "Install Python 3.13 and svg2ooxml (pip install svg2ooxml), "
                "or set AUTOFIGURE_PY313 env var to the python.exe path."
            ),
        )

    bridge = BASE_DIR / "svg2pptx_bridge.py"
    if not bridge.is_file():
        raise HTTPException(status_code=500, detail=f"Bridge script missing: {bridge}")

    # Use a temp output path inside the job dir to avoid clashing with the
    # embedded-SVG export. Cleaned up after reading.
    out_path = output_dir / f"native_{uuid.uuid4().hex}.pptx"

    try:
        # 子进程 stdin/stdout 必须显式用 UTF-8：Windows 默认 GBK，
        # SVG 中的特殊符号（如 ⋮ U+22EE）会导致 stdin.write 抛 UnicodeEncodeError。
        bridge_env = os.environ.copy()
        bridge_env["PYTHONUTF8"] = "1"
        proc = subprocess.run(
            [py313, str(bridge), str(out_path)],
            input=req.svg,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
            cwd=str(BASE_DIR),
            env=bridge_env,
        )
    except subprocess.TimeoutExpired:
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        raise HTTPException(status_code=504, detail="svg2ooxml conversion timed out (>120s)")

    if proc.returncode != 0:
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise HTTPException(
            status_code=500,
            detail=f"svg2ooxml conversion failed (Python 3.13): {detail}",
        )

    if not out_path.is_file():
        raise HTTPException(status_code=500, detail="svg2ooxml produced no output file")

    try:
        pptx_bytes = out_path.read_bytes()
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass

    filename = f"figure_{job_id}_native.pptx"
    return Response(
        content=pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/uploads/{filename}")
def get_upload(filename: str) -> FileResponse:
    candidate = (UPLOADS_DIR / filename).resolve()
    if not str(candidate).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _monitor_job(job: Job) -> None:
    job.push("status", {"state": "started"})

    stdout_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    idle_cycles = 0
    while True:
        _scan_artifacts(job)

        if job.process.poll() is not None:
            idle_cycles += 1
        else:
            idle_cycles = 0

        if idle_cycles >= 4:
            break
        time.sleep(0.5)

    _scan_artifacts(job)
    job.push("status", {"state": "finished", "code": job.process.returncode})
    job.push(
        "artifact",
        {
            "kind": "log",
            "name": job.log_path.name,
            "path": job.log_path.relative_to(job.output_dir).as_posix(),
            "url": f"/api/artifacts/{job.job_id}/{job.log_path.name}",
        },
    )
    job.done = True
    job.push("close", {})


def _pipe_output(job: Job, pipe, stream_name: str) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        text = line.rstrip()
        if text:
            job.write_log(stream_name, text)
            job.push("log", {"stream": stream_name, "line": text})
    pipe.close()


def _scan_artifacts(job: Job) -> None:
    output_dir = job.output_dir
    candidates = [
        output_dir / "figure.png",
        output_dir / "samed.png",
        output_dir / "template.svg",
        output_dir / "final.svg",
    ]

    icons_dir = output_dir / "icons"
    if icons_dir.is_dir():
        candidates.extend(icons_dir.glob("icon_*.png"))

    for path in candidates:
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        if rel_path in job.seen:
            continue
        job.seen.add(rel_path)

        kind = _classify_artifact(rel_path)
        job.push(
            "artifact",
            {
                "kind": kind,
                "name": path.name,
                "path": rel_path,
                "url": f"/api/artifacts/{job.job_id}/{rel_path}",
            },
        )


def _classify_artifact(rel_path: str) -> str:
    if rel_path == "figure.png":
        return "figure"
    if rel_path == "samed.png":
        return "samed"
    if rel_path.endswith("_nobg.png"):
        return "icon_nobg"
    if rel_path.startswith("icons/") and rel_path.endswith(".png"):
        return "icon_raw"
    if rel_path == "template.svg":
        return "template_svg"
    if rel_path == "final.svg":
        return "final_svg"
    return "artifact"


@app.get("/local-config.js")
def local_config_js() -> Response:
    """Serve optional local-only UI prefill.

    ``web/local-config.js`` is git-ignored so personal API keys / endpoints
    stay on this machine; the public repo ships an empty shell.
    """
    candidate = WEB_DIR / "local-config.js"
    if candidate.is_file():
        return FileResponse(candidate, media_type="application/javascript")
    return Response("// no local overrides\n", media_type="application/javascript")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    def find_available_port(start_port: int, max_attempts: int = 100) -> int:
        for port in range(start_port, start_port + max_attempts):
            # Quick check: if we can connect to 127.0.0.1:port, treat it as in-use.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                try:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        print(f"Port {port} already serving on 127.0.0.1, trying next...")
                        continue
                except OSError:
                    pass

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # On Windows, SO_EXCLUSIVEADDRUSE prevents successful bind when another process
                # already owns the port (important when another app listens on 127.0.0.1).
                if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    print(f"Port {port} is in use, trying next...")
                    continue
        raise IOError(f"No available ports found in range ({start_port} - {start_port + max_attempts})")

    def start_server_with_fallback() -> int:
        # Allow overriding the initial port from environment for flexibility in Docker/PM2/etc.
        initial_port = int(os.environ.get("AUTOFIGURE_PORT", 8000))
        # Host can be overridden via env var. Default 0.0.0.0 keeps backward
        # compatibility for Docker/LAN deployments; launch.py sets 127.0.0.1 for
        # single-user desktop usage so the server only listens on localhost.
        host = os.environ.get("AUTOFIGURE_HOST", "0.0.0.0")
        # The address shown to the user in the browser: 127.0.0.1 for localhost,
        # otherwise the configured host (e.g. for LAN deployments).
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host

        try:
            actual_port = find_available_port(initial_port)
        except Exception as e:
            print(f"Startup failed while searching port: {e}")
            raise

        for attempt in range(3):
            try:
                print("--- Starting Server ---")
                print(f"Local access: http://{display_host}:{actual_port}")
                print("-----------------------")

                uvicorn.run(
                    "server:app",
                    host=host,
                    port=actual_port,
                    reload=False,
                    access_log=False,
                )
                return actual_port
            except OSError as exc:  # Catch bind errors even if a race occurs after our probe
                print(f"Failed to bind port {actual_port} ({exc}), retrying with next port...")
                actual_port += 1
        raise SystemExit("Failed to start server after multiple port attempts")

    try:
        start_server_with_fallback()
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)
