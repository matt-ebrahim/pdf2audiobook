"""FastAPI web application for pdf2audiobook.

Upload a PDF, watch chapters stream through clean->chunk->synth in real-time,
and start listening as soon as each chapter's audio is ready.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.responses import StreamingResponse

app = FastAPI(title="pdf2audiobook")

UPLOAD_DIR = Path("uploads")

# ── Global Configuration ────────────────────────────────────

_config: Any = None  # Loaded at startup


def _load_global_config() -> Any:
    """Load config from config.toml if present, otherwise defaults."""
    from pdf2audiobook.config import load_config

    for candidate in (Path("config.toml"), Path.home() / ".pdf2audiobook" / "config.toml"):
        if candidate.exists():
            return load_config(candidate)
    return load_config()


def _get_config() -> Any:
    global _config
    if _config is None:
        _config = _load_global_config()
    return _config


@dataclass
class ChapterInfo:
    index: int
    title: str = ""
    status: str = "pending"
    error: str | None = None


@dataclass
class Job:
    id: str
    pdf_path: Path
    original_name: str = ""
    status: str = "pending"
    chapters: list[ChapterInfo] = field(default_factory=list)
    output_dir: Path | None = None
    error: str | None = None
    _queues: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)


# In-memory job store
_jobs: dict[str, Job] = {}


def _push_event(job: Job, event: str, data: dict[str, Any]) -> None:
    """Push an SSE event to all connected clients. Thread-safe."""
    payload = {"event": event, **data}
    if job._loop is not None:
        for q in job._queues:
            asyncio.run_coroutine_threadsafe(q.put(payload), job._loop)


# ── Static & Upload ──────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    static_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(static_path.read_text(encoding="utf-8"))


@app.get("/settings")
async def get_settings():
    """Return current settings (API keys masked)."""
    cfg = _get_config()

    def mask(key: str) -> str:
        if not key:
            return ""
        if len(key) <= 8:
            return "***"
        return key[:4] + "..." + key[-4:]

    return JSONResponse({
        "llm_model": cfg.clean.llm_model,
        "llm_backend": cfg.clean.llm_backend,
        "tts_engine": cfg.synth.tts_engine,
        "voice": cfg.synth.voice,
        "api_keys": {
            "openai": mask(cfg.api_keys.openai),
            "anthropic": mask(cfg.api_keys.anthropic),
            "elevenlabs": mask(cfg.api_keys.elevenlabs),
        },
    })


@app.post("/settings")
async def update_settings(body: dict):
    """Update runtime settings (API keys, LLM model, TTS engine)."""
    cfg = _get_config()

    if "llm_model" in body and body["llm_model"]:
        cfg.clean.llm_model = body["llm_model"]
    if "llm_backend" in body:
        cfg.clean.llm_backend = body["llm_backend"]
    if "tts_engine" in body and body["tts_engine"]:
        cfg.synth.tts_engine = body["tts_engine"]
    if "voice" in body and body["voice"]:
        cfg.synth.voice = body["voice"]

    # Only update API keys if a new non-masked value is provided
    keys = body.get("api_keys", {})
    if keys.get("openai") and "..." not in keys["openai"]:
        cfg.api_keys.openai = keys["openai"]
    if keys.get("anthropic") and "..." not in keys["anthropic"]:
        cfg.api_keys.anthropic = keys["anthropic"]
    if keys.get("elevenlabs") and "..." not in keys["elevenlabs"]:
        cfg.api_keys.elevenlabs = keys["elevenlabs"]

    # Also push API keys into environment so litellm picks them up
    import os
    if cfg.api_keys.openai:
        os.environ["OPENAI_API_KEY"] = cfg.api_keys.openai
    if cfg.api_keys.anthropic:
        os.environ["ANTHROPIC_API_KEY"] = cfg.api_keys.anthropic
    if cfg.api_keys.elevenlabs:
        os.environ["ELEVENLABS_API_KEY"] = cfg.api_keys.elevenlabs

    return JSONResponse({"status": "ok"})


@app.post("/upload")
async def upload(file: UploadFile):
    job_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"

    content = await file.read()
    pdf_path.write_bytes(content)

    original_name = file.filename or "upload"
    job = Job(id=job_id, pdf_path=pdf_path, original_name=original_name)
    job._loop = asyncio.get_event_loop()
    _jobs[job_id] = job

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job, job)

    return JSONResponse({"job_id": job_id})


# ── SSE Progress Stream ─────────────────────────────────────


@app.get("/jobs/{job_id}/progress")
async def progress_stream(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    queue: asyncio.Queue = asyncio.Queue()
    job._queues.append(queue)

    async def event_generator():
        try:
            snapshot = {
                "event": "snapshot",
                "status": job.status,
                "chapters": [
                    {"index": c.index, "title": c.title, "status": c.status}
                    for c in job.chapters
                ],
            }
            yield f"data: {_json_dumps(snapshot)}\n\n"

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {_json_dumps(payload)}\n\n"
                    if payload.get("event") in ("pipeline_complete", "job_error"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {_json_dumps({'event': 'keepalive'})}\n\n"
        finally:
            if queue in job._queues:
                job._queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chapter Data ─────────────────────────────────────────────


@app.get("/jobs/{job_id}/chapters")
async def get_chapters(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse([
        {"index": c.index, "title": c.title, "status": c.status, "error": c.error}
        for c in job.chapters
    ])


@app.get("/jobs/{job_id}/text/{chapter_index}")
async def get_text(job_id: str, chapter_index: int):
    """Serve cleaned chapter text split into paragraphs for the reader."""
    job = _jobs.get(job_id)
    if not job or not job.output_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)

    text_path = job.output_dir / "cleaned" / f"chapter_{chapter_index:03d}.txt"
    if not text_path.exists():
        return JSONResponse({"error": "Text not ready"}, status_code=404)

    text = text_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    title = ""
    if chapter_index < len(job.chapters):
        title = job.chapters[chapter_index].title

    return JSONResponse({"title": title, "paragraphs": paragraphs})


# ── Audio Streaming & Downloads ──────────────────────────────


@app.get("/jobs/{job_id}/audio/{chapter_index}")
async def get_audio(job_id: str, chapter_index: int):
    job = _jobs.get(job_id)
    if not job or not job.output_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)

    mp3_path = job.output_dir / "audio" / f"chapter_{chapter_index:03d}.mp3"
    if not mp3_path.exists():
        return JSONResponse({"error": "Audio not ready"}, status_code=404)

    return FileResponse(mp3_path, media_type="audio/mpeg")


@app.get("/jobs/{job_id}/download/{chapter_index}")
async def download_chapter(job_id: str, chapter_index: int):
    """Download a single chapter MP3 with proper filename."""
    job = _jobs.get(job_id)
    if not job or not job.output_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)

    mp3_path = job.output_dir / "audio" / f"chapter_{chapter_index:03d}.mp3"
    if not mp3_path.exists():
        return JSONResponse({"error": "Audio not ready"}, status_code=404)

    title = f"chapter_{chapter_index:03d}"
    if chapter_index < len(job.chapters):
        raw = job.chapters[chapter_index].title
        title = "".join(c for c in raw if c.isalnum() or c in " _-").strip() or title

    return FileResponse(
        mp3_path,
        media_type="audio/mpeg",
        filename=f"{chapter_index:02d} - {title}.mp3",
    )


@app.get("/jobs/{job_id}/download-all")
async def download_all(job_id: str):
    """Download all ready chapter MP3s as a zip archive."""
    job = _jobs.get(job_id)
    if not job or not job.output_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)

    ready = [c for c in job.chapters if c.status == "ready"]
    if not ready:
        return JSONResponse({"error": "No chapters ready"}, status_code=400)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ch in ready:
            mp3_path = job.output_dir / "audio" / f"chapter_{ch.index:03d}.mp3"
            if mp3_path.exists():
                safe = "".join(c for c in ch.title if c.isalnum() or c in " _-").strip()
                arcname = f"{ch.index:02d} - {safe or f'Chapter {ch.index + 1}'}.mp3"
                zf.write(mp3_path, arcname)
    buf.seek(0)

    stem = Path(job.original_name).stem
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_audiobook.zip"'},
    )


@app.get("/jobs/{job_id}/pdf")
async def get_pdf(job_id: str):
    """Serve the original uploaded PDF for the embedded viewer."""
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(job.pdf_path, media_type="application/pdf")


# ── Background Pipeline Runner ───────────────────────────────


def _run_job(job: Job) -> None:
    """Run the full pipeline in a background thread."""
    from pdf2audiobook.checkpoint import Checkpoint
    from pdf2audiobook.progress import Progress

    try:
        job.status = "running"
        _push_event(job, "job_status", {"status": "running"})

        stem = Path(job.original_name).stem.replace(" ", "_")
        output_dir = Path("output") / f"web_{stem}_{job.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = output_dir

        config = _get_config()

        def on_event(event: str, data: dict[str, Any]) -> None:
            if event == "chapter_stage":
                ch_idx = data["index"]
                stage = data["stage"]
                if ch_idx < len(job.chapters):
                    job.chapters[ch_idx].status = stage
                _push_event(job, "chapter_stage", data)
            elif event == "chapter_text_ready":
                _push_event(job, "chapter_text_ready", data)
            elif event == "chapter_audio_ready":
                ch_idx = data["index"]
                if ch_idx < len(job.chapters):
                    job.chapters[ch_idx].status = "ready"
                _push_event(job, "chapter_audio_ready", data)
            elif event == "chapter_error":
                ch_idx = data["index"]
                if ch_idx < len(job.chapters):
                    job.chapters[ch_idx].status = "error"
                    job.chapters[ch_idx].error = data.get("error")
                _push_event(job, "chapter_error", data)
            elif event == "pipeline_complete":
                _push_event(job, "pipeline_complete", data)

        progress = Progress(on_event=on_event)

        checkpoint_path = output_dir / "checkpoint.json"
        checkpoint = Checkpoint.load_or_create(checkpoint_path, job.pdf_path)

        from pdf2audiobook.parse import parse_pdf

        chapters = parse_pdf(job.pdf_path, output_dir, config.parse, checkpoint, progress)

        # Generate executive summary and inject as first chapter
        from pdf2audiobook.summary import generate_and_inject_summary

        chapters = generate_and_inject_summary(
            chapters, output_dir, config.clean, config.api_keys, checkpoint, progress
        )

        job.chapters = [
            ChapterInfo(index=ch.index, title=ch.title) for ch in chapters
        ]
        _push_event(job, "chapters_parsed", {
            "chapters": [
                {"index": c.index, "title": c.title, "status": c.status}
                for c in job.chapters
            ]
        })

        if config.synth.tts_engine in ("openai", "elevenlabs"):
            if config.chunk.max_chars == 400:
                config.chunk.max_chars = 4096

        from pdf2audiobook.streaming import StreamingPipeline

        pipeline = StreamingPipeline(
            chapters=chapters,
            output_dir=output_dir,
            config=config,
            checkpoint=checkpoint,
            progress=progress,
            on_event=on_event,
        )
        pipeline.run()

        job.status = "complete"
        _push_event(job, "pipeline_complete", {"total": len(chapters)})

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        _push_event(job, "job_error", {"error": str(e)})


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def main() -> None:
    """Entry point for pdf2audiobook-web command."""
    import uvicorn
    print("Starting pdf2audiobook web server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
