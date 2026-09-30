"""Web UI local cho review-voice — dán kịch bản tag → nghe → sửa từng câu → tải về CapCut.

  ./rv web            → http://127.0.0.1:8765

Engine VieNeu chạy 1 luồng (ONNX/CPU): mọi job xếp hàng qua 1 lock, UI poll /api/job.
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import render as R  # noqa: E402
import tiktok_adb as tta  # noqa: E402
import voicebuild as vb  # noqa: E402

app = FastAPI()
ENGINE_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
PREVIEW = "_preview"


def _slug(name: str) -> str:
    if name == PREVIEW:
        return PREVIEW
    s = re.sub(r"[^\w-]+", "_", (name or "").strip(), flags=re.UNICODE).strip("_")
    if not s:
        raise HTTPException(400, "Tên dự án trống.")
    return s[:60]


def _project_dir(name: str) -> Path:
    d = vb.OUT / _slug(name)
    if not (d / "manifest.json").exists():
        raise HTTPException(404, f"Chưa có dự án '{name}'.")
    return d


def _run_job(fn, **info) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"state": "queued", "done": 0, "total": 0, "error": None, **info}

    def work():
        with ENGINE_LOCK:
            JOBS[jid]["state"] = "running"
            try:
                fn(JOBS[jid])
                JOBS[jid]["state"] = "done"
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                JOBS[jid].update(state="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return jid


@app.on_event("startup")
def _warm():
    # nạp model sẵn (~8s) để lần render đầu không phải chờ
    def load():
        with ENGINE_LOCK:
            vb._engine()
    threading.Thread(target=load, daemon=True).start()


# ─── pages / meta ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    cloned = sorted(p.stem for p in vb.VOICES.glob("*.json"))
    presets = []
    if vb._tts is not None:
        presets = [n for n, v in vb._tts._preset_voices.items() if v.get("gender") == "male"]
        presets += [n for n, v in vb._tts._preset_voices.items() if v.get("gender") != "male"]
    projects = sorted(
        (p.parent.name for p in vb.OUT.glob("*/manifest.json") if p.parent.name != PREVIEW),
        key=lambda n: -(vb.OUT / n / "manifest.json").stat().st_mtime)
    return {
        "ready": vb._tts is not None,
        "voices": cloned, "presets": presets,
        "tags": [t for t in R.PROFILES if t != "neutral"],
        "native_tags": ["cười", "thở dài", "hắng giọng"],
        "projects": projects,
    }


# ─── render ───────────────────────────────────────────────────────────────────

class RenderReq(BaseModel):
    script: str
    voice: str
    name: str
    speed: float = 1.0
    seed: int = 42
    takes: int = 3


@app.post("/api/render")
def api_render(req: RenderReq):
    name = _slug(req.name)
    total = len(R.parse(req.script))
    if not total:
        raise HTTPException(400, "Kịch bản trống.")

    def fn(job):
        job["total"] = total
        R.render_text(req.script, req.voice, name, max(0.7, min(1.4, req.speed)), req.seed,
                      on_line=lambda i, n, ln: job.update(done=i, total=n),
                      takes=max(1, min(6, req.takes)))

    return {"job": _run_job(fn, name=name)}


class LineReq(BaseModel):
    resynth: bool = True
    seed: int | None = None
    tag: str | None = None
    tts: str | None = None
    subtitle: str | None = None


@app.post("/api/project/{name}/line/{n}")
def api_line(name: str, n: int, req: LineReq):
    _project_dir(name)

    def fn(job):
        job["total"] = 1
        R.redo_line(_slug(name), n, seed=req.seed, tag=req.tag, tts=req.tts,
                    resynth=req.resynth, subtitle=req.subtitle)
        job["done"] = 1

    return {"job": _run_job(fn, name=_slug(name))}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "job không tồn tại")
    return JOBS[jid]


# ─── project files ────────────────────────────────────────────────────────────

@app.get("/api/project/{name}")
def api_project(name: str):
    d = _project_dir(name)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    script = d / "script.txt"
    man["script_text"] = script.read_text(encoding="utf-8") if script.exists() else ""
    man["name"] = d.name
    man["mtime"] = int((d / "full.wav").stat().st_mtime * 1000)
    return man


@app.get("/files/{name}/{path:path}")
def files(name: str, path: str):
    d = _project_dir(name)
    f = (d / path).resolve()
    if d.resolve() not in f.parents or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-store"},
                        filename=f"{d.name}_{f.name}" if f.parent == d else f.name)


@app.get("/api/project/{name}/zip")
def api_zip(name: str):
    d = _project_dir(name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("full.wav", "full.mp3", "full.srt"):
            z.write(d / f, f"{d.name}/{f}")
        for f in sorted((d / "lines").glob("*.wav")):
            z.write(f, f"{d.name}/lines/{f.name}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{d.name}_capcut.zip"'})


# ─── lexicon ──────────────────────────────────────────────────────────────────

@app.get("/api/lexicon")
def get_lex():
    return {"text": (ROOT / "lexicon.json").read_text(encoding="utf-8")}


class LexReq(BaseModel):
    text: str


@app.put("/api/lexicon")
def put_lex(req: LexReq):
    try:
        data = json.loads(req.text)
        assert isinstance(data, dict) and all(isinstance(v, str) for v in data.values())
    except Exception:  # noqa: BLE001
        raise HTTPException(400, 'JSON sai. Dạng đúng: {"FOMO": "phô mô", ...}')
    (ROOT / "lexicon.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return {"ok": True, "count": len([k for k in data if not k.startswith("_")])}


# ─── tiktok: quản lý video tự đăng ────────────────────────────────────────────
# Danh sách lưu ở work/tiktok_post/videos.json, file ở work/tiktok_post/videos/.
# Luồng hẹn giờ quét mỗi 30s: video "scheduled" tới giờ → _publish_tiktok().

TT_DIR = ROOT / "work" / "tiktok_post"
TT_FILES = TT_DIR / "videos"
TT_DB = TT_DIR / "videos.json"
TT_LOCK = threading.Lock()
TT_WAKE = threading.Event()
TT_POST_LOCK = threading.Lock()              # đang đăng ↔ đang dọn file trên máy: không chạy chồng
TT_EXT = {".mp4", ".mov", ".m4v", ".webm"}
TT_PRIVACY = {"public", "friends", "private"}
TT_LOCKED = {"posting", "posted"}          # không sửa / xoá được nữa


def _tt_load() -> list[dict]:
    return json.loads(TT_DB.read_text(encoding="utf-8")) if TT_DB.exists() else []


def _tt_save(items: list[dict]):
    TT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TT_DB.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(TT_DB)


def _tt_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M")


def _tt_patch(vid: str, **patch):
    with TT_LOCK:
        items = _tt_load()
        for x in items:
            if x["id"] == vid:
                x.update(patch)
        _tt_save(items)


def _publish_tiktok(item: dict, path: Path) -> str:
    """Đăng qua điện thoại cắm USB (tiktok_adb.py). Trả về ghi chú, lỗi thì raise."""
    return tta.post(path, item.get("caption", ""), item.get("privacy", "public"), item.get("device") or None,
                    log=lambda s: _tt_patch(item["id"], step=s), tag=item["id"],
                    hashtags=item.get("hashtags", ""), mentions=item.get("mentions", ""))


def _tt_scheduler():
    while True:
        TT_WAKE.wait(30)
        TT_WAKE.clear()
        now = _tt_now()
        with TT_LOCK:
            items = _tt_load()
            due = [it for it in items if it["status"] == "scheduled" and (it.get("schedule_at") or "~") <= now]
            for it in due:
                it["status"] = "posting"
            if due:
                _tt_save(items)
        for it in due:
            try:
                with TT_POST_LOCK:
                    note = _publish_tiktok(it, TT_FILES / it["file"])
                _tt_patch(it["id"], status="posted", posted_at=_tt_now(), note=note, error=None, step=None)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                _tt_patch(it["id"], status="failed", error=str(e), step=None)


@app.on_event("startup")
def _tt_start():
    with TT_LOCK:                              # tắt server giữa lúc đăng → đánh dấu lỗi để đăng lại
        items = _tt_load()
        stuck = [it for it in items if it["status"] == "posting"]
        for it in stuck:
            it.update(status="failed", error="Server tắt giữa lúc đăng — xem TikTok đã có bài chưa rồi mới Thử lại")
        if stuck:
            _tt_save(items)
    threading.Thread(target=_tt_scheduler, daemon=True).start()


class TTFields(BaseModel):
    title: str = ""
    caption: str = ""
    hashtags: str = ""
    mentions: str = ""                         # "@a @b" — nhắc đến
    privacy: str = "public"
    device: str = ""                           # serial adb; trống = máy đầu tiên đang cắm
    schedule_at: str | None = None             # "YYYY-MM-DDTHH:MM" theo giờ máy chạy server


class TTNew(TTFields):
    file: str
    filename: str


class TTEdit(TTFields):
    id: str


class TTCreateReq(BaseModel):
    items: list[TTNew]


class TTEditReq(BaseModel):
    items: list[TTEdit]


class TTIdsReq(BaseModel):
    ids: list[str]


def _tt_fields(f: TTFields) -> dict:
    at = (f.schedule_at or "").strip()[:16] or None
    if at and not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d", at):
        raise HTTPException(400, f"Giờ đăng sai định dạng: {at}")
    return {
        "title": f.title.strip()[:150], "caption": f.caption.strip()[:2200],
        # "##a##b, c" → "#a #b #c" (hashtag dính liền thì TikTok không nhận)
        "hashtags": " ".join(dict.fromkeys("#" + t for t in re.split(r"[\s,#]+", f.hashtags) if t)),
        "mentions": " ".join(dict.fromkeys("@" + t for t in re.split(r"[\s,@]+", f.mentions) if t)),
        "privacy": f.privacy if f.privacy in TT_PRIVACY else "public",
        "device": re.sub(r"[^\w.:-]", "", f.device)[:64],
        "schedule_at": at, "status": "scheduled" if at else "draft", "error": None,
    }


@app.get("/api/tiktok/videos")
def tt_list():
    return {"items": _tt_load(), "now": _tt_now()}


TT_SETTINGS = TT_DIR / "settings.json"
# theo kênh @nana.chan_07: #xuhuong ở 55/80 bài, 2 bài liên tiếp cách nhau ~5 giờ
TT_DEFAULTS = {"caption": "", "hashtags": "#xuhuong", "privacy": "public", "every": 5, "unit": 60}


class TTSettings(BaseModel):
    caption: str = ""
    hashtags: str = ""
    privacy: str = "public"
    every: int = 5
    unit: int = 60                             # phút: 1 / 60 / 1440


@app.get("/api/tiktok/settings")
def tt_settings():
    s = json.loads(TT_SETTINGS.read_text(encoding="utf-8")) if TT_SETTINGS.exists() else {}
    return {**TT_DEFAULTS, **s}


@app.put("/api/tiktok/settings")
def tt_settings_put(req: TTSettings):
    f = _tt_fields(TTFields(caption=req.caption, hashtags=req.hashtags, privacy=req.privacy))
    s = {"caption": f["caption"], "hashtags": f["hashtags"], "privacy": f["privacy"],
         "every": max(1, min(req.every, 999)), "unit": req.unit if req.unit in (1, 60, 1440) else 60}
    TT_DIR.mkdir(parents=True, exist_ok=True)
    TT_SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


@app.get("/api/tiktok/devices")
def tt_devices():
    return {"devices": [d for d in tta.devices() if d["state"] == "device"]}


@app.get("/api/tiktok/phone-files")
def tt_phone_files():
    res = []
    for d in tta.devices():
        if d["state"] == "device":
            files = tta.Phone(d["serial"], log=lambda s: None).pushed_videos()
            res.append({**d, "files": files, "total": sum(f["size"] for f in files)})
    return {"devices": res, "dir": tta.REMOTE_DIR}


class TTPhoneReq(BaseModel):
    serial: str


@app.post("/api/tiktok/phone-files/clear")
def tt_phone_clear(req: TTPhoneReq):
    busy = HTTPException(409, "Đang có video đăng dở — TikTok còn đọc file trên máy, xoá sau khi đăng xong.")
    if any(it["status"] == "posting" for it in _tt_load()) or not TT_POST_LOCK.acquire(blocking=False):
        raise busy
    try:
        return {"deleted": tta.Phone(req.serial, log=lambda s: None).clear_pushed()}
    except tta.PostError as e:
        raise HTTPException(400, str(e)) from None
    finally:
        TT_POST_LOCK.release()


@app.put("/api/tiktok/upload")
async def tt_upload(request: Request, filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in TT_EXT:
        raise HTTPException(400, f"{filename}: chỉ nhận {', '.join(sorted(TT_EXT))}")
    TT_FILES.mkdir(parents=True, exist_ok=True)
    dest = TT_FILES / f"{uuid.uuid4().hex[:12]}{ext}"
    size = 0
    with dest.open("wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    return {"file": dest.name, "size": size}


@app.post("/api/tiktok/videos")
def tt_create(req: TTCreateReq):
    new = []
    for it in req.items:
        if not re.fullmatch(r"[0-9a-f]{12}\.\w+", it.file) or not (TT_FILES / it.file).exists():
            raise HTTPException(400, f"Chưa tải lên xong: {it.filename}")
        new.append({"id": uuid.uuid4().hex[:10], "file": it.file, "filename": it.filename,
                    "size": (TT_FILES / it.file).stat().st_size, "created_at": _tt_now(), **_tt_fields(it)})
    with TT_LOCK:
        _tt_save(new + _tt_load())
    TT_WAKE.set()
    return {"items": new}


@app.patch("/api/tiktok/videos")
def tt_edit(req: TTEditReq):
    edits = {e.id: _tt_fields(e) for e in req.items}
    with TT_LOCK:
        items = _tt_load()
        for it in items:
            if it["id"] in edits and it["status"] not in TT_LOCKED:
                it.update(edits[it["id"]])
        _tt_save(items)
    TT_WAKE.set()
    return {"ok": True}


@app.post("/api/tiktok/delete")
def tt_delete(req: TTIdsReq):
    ids = set(req.ids)
    with TT_LOCK:
        items = _tt_load()
        gone = [it for it in items if it["id"] in ids and it["status"] != "posting"]
        _tt_save([it for it in items if it not in gone])
    for it in gone:
        (TT_FILES / it["file"]).unlink(missing_ok=True)
    return {"deleted": len(gone)}


@app.get("/tiktok/files/{name}")
def tt_file(name: str):
    f = TT_FILES / name
    if not re.fullmatch(r"[0-9a-f]{12}\.\w+", name) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


def _lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))     # không gửi gì, chỉ để hệ điều hành chọn card mạng
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    # ./rv web [port] [--local]   --local = chỉ máy này (không mở cho mạng LAN)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(args[0]) if args else 8765
    host = "127.0.0.1" if "--local" in sys.argv else "0.0.0.0"
    print(f"\n  review-voice")
    print(f"    máy này     → http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"    cùng mạng   → http://{_lan_ip()}:{port}   (điện thoại / máy khác cùng Wi-Fi)")
    print()
    uvicorn.run(app, host=host, port=port, log_level="warning")
