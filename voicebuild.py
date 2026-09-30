"""BUILD GIỌNG từ file mẫu — clone local bằng VieNeu-TTS v3 Turbo.

VieNeu v3 chỉ lấy TỐI ĐA 8 GIÂY đầu của audio mẫu để trích speaker embedding
+ reference codes. Nên "build giọng" = chọn đúng 8s đẹp nhất trong mẫu, không
phải đưa cả file dài. Quy trình:

  prep      mẫu dài → chuẩn hoá → cắt các ứng viên 5-8s tại chỗ ngắt nghỉ
  audition  mỗi ứng viên đọc cùng bộ câu review → nghe so sánh
  pick      chốt 1 ứng viên → lưu voices/<name>.json (dùng lại mãi, không cần mẫu)
  say       đọc thử 1 câu bất kỳ bằng giọng đã chốt

Chạy qua ./rv (dùng .venv riêng của dự án: vieneu + torch):
  ./rv prep samples/adam.mp3 --name adam
  ./rv audition --name adam
  ./rv pick --name adam --cand 3
  ./rv say --name adam "Và rồi, điều không ai ngờ tới đã xảy ra!"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
VOICES = ROOT / "voices"
OUT = ROOT / "out"

SR = 44100
REF_MAX = 8.0          # VieNeu v3 cắt ref ở 8s — dài hơn là phí
REF_MIN = 4.5          # ngắn quá thì embedding thiếu chất giọng
STYLES = ("tu_nhien", "doc_truyen", "tin_tuc")
PRECISION = "fp32"      # "int8" nhanh hơn ~25% — chỉ bật khi đã đo chất lượng không giảm

# Bộ câu audition: đủ các kiểu nhấn của giọng review (hỏi, kể, cao trào, thở dài).
TEST_LINES = [
    "Bạn có tin không? Chỉ vì một chiếc hộp nhỏ, cả thị trấn này đã rơi vào hỗn loạn.",
    "Hắn mỉm cười, [thở dài] nhưng hắn không hề biết, đây chính là sai lầm lớn nhất cuộc đời mình.",
    "Và rồi, điều không ai ngờ tới đã xảy ra!",
]


# ─── audio helpers ────────────────────────────────────────────────────────────

def _ff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                          capture_output=True, text=True)


def _dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def _normalize(src: Path, dst: Path) -> None:
    """mono 44.1k, cắt rung thấp, chuẩn loudness -18 LUFS (mức ref ổn cho encoder)."""
    r = _ff("-i", str(src), "-ac", "1", "-ar", str(SR),
            "-af", "highpass=f=70,loudnorm=I=-18:TP=-2:LRA=11", str(dst))
    if r.returncode:
        sys.exit(f"ffmpeg lỗi khi chuẩn hoá mẫu:\n{r.stderr[-800:]}")


def _silences(path: Path, noise_db: int = -35, min_sil: float = 0.22) -> list[tuple[float, float]]:
    r = _ff("-i", str(path), "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends + [None] * (len(starts) - len(ends))))


def _speech_segments(path: Path) -> list[tuple[float, float]]:
    """Đảo khoảng lặng → các đoạn có tiếng."""
    total = _dur(path)
    segs, cur = [], 0.0
    for s, e in _silences(path):
        if s - cur > 0.15:
            segs.append((cur, s))
        cur = e if e is not None else total
    if total - cur > 0.15:
        segs.append((cur, total))
    return segs


def _candidates(segs: list[tuple[float, float]], k: int) -> list[tuple[float, float, float]]:
    """Cửa sổ 4.5-8s bắt đầu/kết thúc ở ranh giới nghỉ.

    Điểm = độ dài gần 8s (nhiều chất giọng) + tỉ lệ có tiếng cao (ít lặng chết),
    nhưng vẫn cho phép 1-2 chỗ ngắt tự nhiên — nhịp ngắt cũng là 1 phần "chất" giọng.
    Trả về k cửa sổ không chồng lấn, điểm cao nhất trước.
    """
    wins = []
    for i in range(len(segs)):
        start = max(0.0, segs[i][0] - 0.08)
        for j in range(i, len(segs)):
            end = segs[j][1] + 0.12
            length = end - start
            if length > REF_MAX:
                break
            if length < REF_MIN:
                continue
            voiced = sum(b - a for a, b in segs[i:j + 1])
            ratio = voiced / length
            score = 0.6 * (length / REF_MAX) + 0.4 * ratio
            wins.append((score, start, end))
    wins.sort(reverse=True)
    picked: list[tuple[float, float, float]] = []
    for score, a, b in wins:
        if all(b <= pa or a >= pb for _, pa, pb in picked):
            picked.append((score, a, b))
        if len(picked) >= k:
            break
    return sorted(picked, key=lambda x: x[1])


# ─── VieNeu ───────────────────────────────────────────────────────────────────

_tts = None


def _engine():
    global _tts
    if _tts is None:
        print("… nạp VieNeu v3 Turbo (fp32 — chất lượng tối đa)", flush=True)
        t = time.time()
        from vieneu import Vieneu
        # Ép ONNX/CPU: có torch thì "auto" sẽ nhảy sang MPS, mà MPS trên M1 hay rè.
        # int8 nhanh hơn ~25% nhưng phải đọc từ thư mục THẬT (onnxruntime mới chặn symlink HF cache).
        prec = os.environ.get("RV_PRECISION", PRECISION)
        int8_dir = ROOT / "models" / "vieneu_int8"
        if prec == "int8" and int8_dir.exists():
            _tts = Vieneu(backend="onnx", device="cpu", precision="int8", onnx_dir=str(int8_dir))
        else:
            _tts = Vieneu(backend="onnx", device="cpu", precision="fp32")
        print(f"  xong {time.time() - t:.1f}s", flush=True)
    return _tts


def _seed(n: int) -> None:
    random.seed(n)
    try:
        import numpy as np
        np.random.seed(n)
    except ImportError:
        pass


def _load_voice(name: str) -> dict:
    import numpy as np
    p = VOICES / f"{name}.json"
    if not p.exists():
        sys.exit(f"Chưa có giọng '{name}'. Chạy audition + pick trước.")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["speaker_emb"] = np.asarray(d["speaker_emb"], dtype=np.float32)
    d["codes"] = None if d.get("codes") is None else np.asarray(d["codes"], dtype=np.int64)
    for e in (d.get("emotions") or {}).values():
        e["speaker_emb"] = np.asarray(e["speaker_emb"], dtype=np.float32)
        e["codes"] = None if e.get("codes") is None else np.asarray(e["codes"], dtype=np.int64)
    return d


def _enroll(ref: Path, denoise: bool):
    return _engine().encode_reference(ref, denoise=denoise)


# ─── commands ─────────────────────────────────────────────────────────────────

def cmd_prep(a) -> None:
    src = Path(a.sample).expanduser().resolve()
    if not src.exists():
        sys.exit(f"Không thấy file mẫu: {src}")
    d = WORK / a.name
    (d / "cands").mkdir(parents=True, exist_ok=True)
    for old in (d / "cands").glob("*.wav"):
        old.unlink()
    full = d / "full.wav"
    _normalize(src, full)
    segs = _speech_segments(full)
    cands = _candidates(segs, a.k)
    if not cands:
        sys.exit("Không tìm được đoạn 4.5-8s liền mạch — mẫu quá ngắn hoặc quá nhiều lặng.")
    meta = []
    for i, (score, s, e) in enumerate(cands, 1):
        out = d / "cands" / f"cand_{i:02d}.wav"
        _ff("-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(full),
            "-af", "afade=t=in:d=0.03,areverse,afade=t=in:d=0.05,areverse", str(out))
        meta.append({"cand": i, "start": round(s, 2), "end": round(e, 2),
                     "len": round(e - s, 2), "score": round(score, 3), "file": str(out)})
        print(f"  cand_{i:02d}  {s:7.2f}s → {e:7.2f}s  ({e - s:.1f}s)  điểm {score:.2f}")
    (d / "cands.json").write_text(json.dumps({"source": str(src), "cands": meta},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {len(meta)} ứng viên trong {d / 'cands'}\n  → tiếp: ./rv audition --name {a.name}")


def cmd_audition(a) -> None:
    d = WORK / a.name
    meta = json.loads((d / "cands.json").read_text(encoding="utf-8"))["cands"]
    lines = TEST_LINES if not a.text else [a.text]
    text = " ".join(lines)
    outd = d / "audition"
    outd.mkdir(exist_ok=True)
    tts = _engine()
    for m in meta:
        if a.cand and m["cand"] not in a.cand:
            continue
        emb, codes = _enroll(Path(m["file"]), denoise=not a.no_denoise)
        voice = {"speaker_emb": emb, "codes": codes}
        for style in (a.styles or ["doc_truyen"]):
            _seed(a.seed)
            t = time.time()
            wav = tts.infer(text, voice=voice, style=style, temperature=a.temp)
            out = outd / f"cand_{m['cand']:02d}_{style}.wav"
            tts.save(wav, out)
            print(f"  cand_{m['cand']:02d} [{style}]  {len(wav) / tts.sample_rate:.1f}s audio"
                  f" / {time.time() - t:.1f}s sinh  → {out.name}", flush=True)
    print(f"\n✓ Nghe trong {outd}\n  → chốt: ./rv pick --name {a.name} --cand <số>")


def cmd_pick(a) -> None:
    import numpy as np
    d = WORK / a.name
    meta = json.loads((d / "cands.json").read_text(encoding="utf-8"))
    m = next((c for c in meta["cands"] if c["cand"] == a.cand), None)
    if not m:
        sys.exit(f"Không có cand {a.cand}")
    ref_keep = VOICES / f"{a.name}_ref.wav"
    ref_keep.write_bytes(Path(m["file"]).read_bytes())
    emb, codes = _enroll(ref_keep, denoise=not a.no_denoise)
    VOICES.mkdir(exist_ok=True)
    data = {
        "name": a.name, "style": a.style, "temperature": a.temp,
        "source": meta["source"], "window": [m["start"], m["end"]],
        "denoise": not a.no_denoise, "created": time.strftime("%Y-%m-%d %H:%M"),
        "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
        "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
    }
    (VOICES / f"{a.name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Đã chốt giọng '{a.name}' → voices/{a.name}.json (+ {ref_keep.name})")


def cmd_say(a) -> None:
    v = _load_voice(a.name)
    tts = _engine()
    _seed(a.seed)
    style = a.style or v.get("style", "doc_truyen")
    temp = a.temp if a.temp is not None else v.get("temperature", 0.8)
    wav = tts.infer(a.text, voice=v, style=style, temperature=temp)
    OUT.mkdir(exist_ok=True)
    out = Path(a.out) if a.out else OUT / f"say_{a.name}_{int(time.time())}.wav"
    tts.save(wav, out)
    print(f"✓ {out}  ({len(wav) / tts.sample_rate:.1f}s, style={style}, temp={temp}, seed={a.seed})")


def cmd_emo(a) -> None:
    """Thêm 1 đoạn mẫu CẢM XÚC cho giọng đã chốt.

    Đo thật 2026-09-27: VieNeu bắt cả cao độ/năng lượng của đoạn mẫu (ref Adam hào hứng
    163Hz → câu mới 159Hz; ref trêu đùa 124Hz → 122Hz) mà vẫn giữ chất giọng (sim 0.83-0.89).
    Nên mỗi tag cảm xúc dùng ref riêng = cách nhấn nhá mạnh nhất mà không méo giọng.
    --ranges "0-3.95,29.95-34.05" ghép nhiều khúc (tổng ≤ 8s).
    """
    import numpy as np
    p = VOICES / f"{a.name}.json"
    if not p.exists():
        sys.exit(f"Chưa có giọng '{a.name}'.")
    src = Path(a.audio).expanduser().resolve()
    tmp = VOICES / f"_tmp_{a.tag}.wav"
    _normalize(src, tmp)
    out = VOICES / f"{a.name}_emo_{a.tag}.wav"
    if a.ranges:
        rs = [tuple(float(x) for x in r.split("-")) for r in a.ranges.split(",")]
        parts = "".join(f"[0]atrim={s0}:{s1},asetpts=N/SR/TB[p{i}];" for i, (s0, s1) in enumerate(rs))
        chain = parts + "".join(f"[p{i}]" for i in range(len(rs))) + f"concat=n={len(rs)}:v=0:a=1"
        r = _ff("-i", str(tmp), "-filter_complex", chain, str(out))
    else:
        r = _ff("-i", str(tmp), "-t", str(REF_MAX), str(out))
    tmp.unlink(missing_ok=True)
    if r.returncode:
        sys.exit(f"ffmpeg lỗi: {r.stderr[-500:]}")
    dur = _dur(out)
    if dur > REF_MAX + 0.3:
        print(f"  ! đoạn dài {dur:.1f}s — VieNeu chỉ dùng {REF_MAX:.0f}s đầu")
    emb, codes = _enroll(out, denoise=a.denoise)
    d = json.loads(p.read_text(encoding="utf-8"))
    d.setdefault("emotions", {})[a.tag] = {
        "ref": out.name, "dur": round(dur, 2),
        "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
        "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
    }
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"✓ '{a.name}' + cảm xúc [{a.tag}] ({dur:.1f}s) · hiện có: {', '.join(d['emotions'])}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build giọng clone cho review phim")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prep", help="chuẩn hoá mẫu + cắt ứng viên 8s")
    s.add_argument("sample")
    s.add_argument("--name", required=True)
    s.add_argument("-k", type=int, default=6, help="số ứng viên")
    s.set_defaults(fn=cmd_prep)

    s = sub.add_parser("audition", help="mỗi ứng viên đọc bộ câu test")
    s.add_argument("--name", required=True)
    s.add_argument("--cand", type=int, nargs="*", help="chỉ nghe các ứng viên này")
    s.add_argument("--styles", nargs="*", choices=STYLES)
    s.add_argument("--text", help="thay bộ câu test bằng câu này")
    s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no-denoise", action="store_true", help="mẫu đã sạch (vd. xuất từ TTS)")
    s.set_defaults(fn=cmd_audition)

    s = sub.add_parser("pick", help="chốt ứng viên thành giọng")
    s.add_argument("--name", required=True)
    s.add_argument("--cand", type=int, required=True)
    s.add_argument("--style", default="doc_truyen", choices=STYLES)
    s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--no-denoise", action="store_true")
    s.set_defaults(fn=cmd_pick)

    s = sub.add_parser("say", help="đọc thử bằng giọng đã chốt")
    s.add_argument("--name", required=True)
    s.add_argument("text")
    s.add_argument("--style", choices=STYLES)
    s.add_argument("--temp", type=float)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_say)

    s = sub.add_parser("emo", help="thêm đoạn mẫu cảm xúc cho giọng")
    s.add_argument("--name", required=True)
    s.add_argument("--tag", required=True, help="excited / playful / confident / whispering / ...")
    s.add_argument("audio")
    s.add_argument("--ranges", help='khúc cần lấy, giây: "0-3.95,29.95-34.05"')
    s.add_argument("--denoise", action="store_true", help="mẫu có tạp âm (mặc định: không lọc)")
    s.set_defaults(fn=cmd_emo)

    a = p.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
