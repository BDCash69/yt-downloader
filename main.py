import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="YT Downloader")

STATIC_DIR = Path(__file__).parent / "static"

QUALITY_FORMATS: dict[str, str] = {
    "best":  "bestvideo+bestaudio/best",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "audio": "bestaudio/best",
}

# ── Job registry ──────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    jid = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[jid] = {
            "status": "pending",    # pending|downloading|zipping|complete|error
            "type": "video",        # video|playlist
            "total": 1,
            "current": 0,
            "current_title": "",
            "progress": "0",
            "speed": "",
            "eta": "",
            "filename": "",
            "filepath": "",
            "tmpdir": "",
            "error": "",
            "created_at": time.time(),
        }
    return jid


def _require_job(jid: str) -> dict:
    job = _jobs.get(jid)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ── Pydantic models ───────────────────────────────────────────────────────────

class InfoRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    quality: str = "best"

class PlaylistEntry(BaseModel):
    url: str
    title: str = ""

class PlaylistDownloadRequest(BaseModel):
    url: str
    quality: str = "best"
    entries: list[PlaylistEntry] | None = None
    playlist_title: str = "playlist"


# ── Background workers ────────────────────────────────────────────────────────

def _progress_hook(jid: str):
    def hook(d: dict) -> None:
        if d["status"] == "downloading":
            raw = d.get("_percent_str", "0").strip().rstrip("%")
            try:
                float(raw)
            except ValueError:
                raw = "0"
            _jobs[jid]["progress"] = raw
            _jobs[jid]["speed"] = d.get("_speed_str", "").strip()
            _jobs[jid]["eta"] = d.get("_eta_str", "").strip()
        elif d["status"] == "finished":
            _jobs[jid]["progress"] = "99"
    return hook


def _ydl_opts(jid: str, out_template: str, quality: str) -> dict:
    is_audio = quality == "audio"
    fmt = QUALITY_FORMATS.get(quality, QUALITY_FORMATS["best"])
    opts: dict = {
        "outtmpl": out_template,
        "progress_hooks": [_progress_hook(jid)],
        "quiet": True,
        "no_warnings": True,
    }
    if is_audio:
        opts["format"] = "bestaudio/best"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    else:
        opts["format"] = fmt
        opts["merge_output_format"] = "mp4"
    return opts


def _find_file(directory: str) -> str | None:
    files = [f for f in os.listdir(directory) if not f.endswith(".part")]
    return files[0] if files else None


def _worker_video(jid: str, url: str, quality: str) -> None:
    job = _jobs[jid]
    tmpdir = tempfile.mkdtemp()
    job["tmpdir"] = tmpdir
    job["status"] = "downloading"

    try:
        opts = _ydl_opts(jid, os.path.join(tmpdir, "%(title)s.%(ext)s"), quality)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            job["current_title"] = info.get("title", "") if info else ""

        filename = _find_file(tmpdir)
        if not filename:
            raise RuntimeError("No output file found after download")

        job["filename"] = filename
        job["filepath"] = os.path.join(tmpdir, filename)
        job["progress"] = "100"
        job["status"] = "complete"

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        shutil.rmtree(tmpdir, ignore_errors=True)


def _worker_playlist(jid: str, url: str, quality: str, entries: list[dict] | None = None, playlist_title: str = "playlist") -> None:
    job = _jobs[jid]
    tmpdir = tempfile.mkdtemp()
    job["tmpdir"] = tmpdir
    job["status"] = "downloading"

    try:
        if entries is None:
            with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "no_warnings": True}) as ydl:
                playlist_info = ydl.extract_info(url, download=False)
            raw = list(playlist_info.get("entries") or [])
            playlist_title = playlist_info.get("title", "playlist")
            entries = []
            for e in raw:
                vid_id = e.get("id", "")
                entries.append({
                    "url": e.get("url") or (f"https://www.youtube.com/watch?v={vid_id}" if vid_id else ""),
                    "title": e.get("title") or f"Video {len(entries) + 1}",
                })

        job["total"] = len(entries)

        videos_dir = os.path.join(tmpdir, "videos")
        os.makedirs(videos_dir)

        for i, entry in enumerate(entries):
            job["current"] = i + 1
            job["current_title"] = entry.get("title", f"Video {i + 1}")
            job["progress"] = "0"
            job["speed"] = ""
            job["eta"] = ""

            video_url = entry.get("url", "")
            if not video_url:
                continue

            try:
                opts = _ydl_opts(
                    jid,
                    os.path.join(videos_dir, f"{i + 1:03d} - %(title)s.%(ext)s"),
                    quality,
                )
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([video_url])
            except Exception as exc:
                print(f"[playlist] skipping video {i + 1}: {exc}")

        # Build zip
        job["status"] = "zipping"
        job["progress"] = "0"
        job["speed"] = ""
        job["eta"] = ""

        safe_title = "".join(c for c in (playlist_title or "playlist") if c.isalnum() or c in " -_").strip()[:60]
        zip_filename = f"{safe_title or 'playlist'}.zip"
        zip_path = os.path.join(tmpdir, zip_filename)

        video_files = sorted(f for f in os.listdir(videos_dir) if not f.endswith(".part"))
        total = len(video_files)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for idx, vf in enumerate(video_files):
                zf.write(os.path.join(videos_dir, vf), vf)
                job["progress"] = str(int((idx + 1) / total * 100)) if total else "100"

        job["filename"] = zip_filename
        job["filepath"] = zip_path
        job["progress"] = "100"
        job["status"] = "complete"

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def status() -> dict:
    return {"ffmpeg": shutil.which("ffmpeg") is not None}


@app.post("/api/info")
async def get_info(req: InfoRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL required")

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    if not info:
        raise HTTPException(400, "Could not extract info from URL")

    if info.get("_type") == "playlist":
        entries = []
        for e in info.get("entries") or []:
            vid_id = e.get("id", "")
            entries.append({
                "id": vid_id,
                "title": e.get("title") or "Unknown",
                "duration": e.get("duration"),
                "thumbnail": e.get("thumbnail") or (
                    f"https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg" if vid_id else None
                ),
                "url": e.get("url") or (
                    f"https://www.youtube.com/watch?v={vid_id}" if vid_id else url
                ),
            })
        return {
            "type": "playlist",
            "title": info.get("title") or "Playlist",
            "uploader": info.get("uploader") or "",
            "count": len(entries),
            "url": url,
            "entries": entries,
        }

    return {
        "type": "video",
        "title": info.get("title") or "Video",
        "uploader": info.get("uploader") or "",
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "url": url,
    }


@app.post("/api/download")
async def start_download(req: DownloadRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL required")
    jid = _new_job()
    threading.Thread(target=_worker_video, args=(jid, url, req.quality), daemon=True).start()
    return {"job_id": jid}


@app.post("/api/download-playlist")
async def start_playlist_download(req: PlaylistDownloadRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL required")
    jid = _new_job()
    _jobs[jid]["type"] = "playlist"
    entries = [e.model_dump() for e in req.entries] if req.entries else None
    threading.Thread(
        target=_worker_playlist,
        args=(jid, url, req.quality, entries, req.playlist_title),
        daemon=True,
    ).start()
    return {"job_id": jid}


_SSE_KEYS = ("status", "type", "total", "current", "current_title", "progress", "speed", "eta", "filename", "error")

@app.get("/api/download/{jid}/progress")
async def job_progress(jid: str) -> StreamingResponse:
    _require_job(jid)

    async def stream():
        while True:
            job = _jobs.get(jid)
            if job is None:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
                break
            yield f"data: {json.dumps({k: job[k] for k in _SSE_KEYS})}\n\n"
            if job["status"] in ("complete", "error"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{jid}/file")
async def serve_file(jid: str) -> StreamingResponse:
    job = _require_job(jid)
    if job["status"] != "complete":
        raise HTTPException(400, "Download not yet complete")

    filepath = job["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found on server")

    filename = job["filename"]
    tmpdir = job["tmpdir"]
    file_size = os.path.getsize(filepath)

    ext = os.path.splitext(filename)[1].lower()
    mime = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".zip": "application/zip",
    }.get(ext, "application/octet-stream")

    def iterator():
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _jobs.pop(jid, None)

    return StreamingResponse(
        iterator(),
        media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Content-Length": str(file_size),
        },
    )


# ── Housekeeping ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("\n  WARNING: ffmpeg not found — merging video+audio streams will fail.")
        print("  Install ffmpeg and add it to PATH: https://ffmpeg.org/download.html\n")
    asyncio.create_task(_cleanup_loop())


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        cutoff = time.time() - 3600
        for jid, job in list(_jobs.items()):
            if job["created_at"] < cutoff:
                shutil.rmtree(job.get("tmpdir", ""), ignore_errors=True)
                _jobs.pop(jid, None)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    import socket
    import uvicorn

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "127.0.0.1"

    print(f"\n  YT Downloader")
    print(f"  Local:   http://localhost:8000")
    print(f"  Network: http://{local_ip}:8000\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
