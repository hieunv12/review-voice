"""RENDER kịch bản có tag kiểu ElevenLabs v3 → audio + SRT cho CapCut.

VieNeu không hiểu [excited]/[whispering]... nên mỗi tag được "dịch" thành:
  - tham số sinh:  style VieNeu + temperature
  - hậu kỳ:        tốc độ, cao độ (rubberband), âm lượng, EQ
Tag có hiệu lực tới khi gặp tag khác (giống ElevenLabs). Tag gốc của VieNeu
([cười] [thở dài] [hắng giọng]) được giữ nguyên để model tự tạo âm.

  ./rv render script.txt --voice adam --name fully
  ./rv redo --name fully --line 4 --seed 7     # đọc lại riêng 1 câu, ghép lại

Đầu ra out/<name>/:  full.wav  full.mp3  full.srt  lines/NN.wav  manifest.json
(stems/ = bản thô chưa master, để redo ghép lại — không kéo vào CapCut)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import voicebuild as vb

ROOT = Path(__file__).resolve().parent
SR = 48000
TARGET_LUFS = -16.0     # chuẩn nền tảng short video
# VieNeu nói nhanh hơn Adam EL ~10% (đo phần có tiếng: 23.0s vs 25.5s cùng bài) → hãm chung.
BASE_TEMPO = 0.90
# NHỊP ĐẦU RA. Đo 19 video trên kênh @nana.chan_07: giọng đăng lên đã được TĂNG TỐC + CẮT LẶNG
# (4.59 âm tiết/s, 0.4 chỗ nghỉ/10s) so với file EL gốc (3.45 âm tiết/s, 3.7 chỗ nghỉ/10s).
#   "kenh" = khớp giọng đã dựng trên kênh (mặc định) · "el" = khớp file ElevenLabs gốc
PACES = {
    "kenh": {"speed": 1.10, "gap_max": 0.18, "inner_max": 0.15},
    "el":   {"speed": 1.00, "gap_max": None, "inner_max": None},
}
TEMP_BOOST = float(__import__("os").environ.get("RV_TEMP_BOOST", "0"))   # thử nghiệm A/B

# tag → (style, temp, tempo, pitch_semitone, gain_dB, eq_filter)
# pitch để 0: đo thật 2026-09-27, dời cao độ làm tụt độ giống Adam 0.86 → 0.79
# (speaker-embedding cosine). Cảm xúc chỉ dùng tốc độ/âm lượng/EQ.
PROFILES: dict[str, tuple] = {
    "neutral":    ("doc_truyen", 0.80, 1.00,   0.0,  0.0, ""),
    "excited":    ("tu_nhien",   0.95, 1.02,  0.0, +1.5, "equalizer=f=3000:t=q:w=1.5:g=2"),
    "playful":    ("tu_nhien",   0.90, 1.00,  0.0, +0.5, ""),
    "amazed":     ("doc_truyen", 0.90, 0.98,  0.0, +0.5, ""),
    "confident":  ("tin_tuc",    0.75, 1.02,  0.0, +1.0, "equalizer=f=180:t=q:w=1.2:g=2"),
    "curious":    ("tin_tuc",    0.85, 0.97,  0.0,  0.0, ""),
    "serious":    ("tin_tuc",    0.70, 0.95,  0.0,  0.0, "equalizer=f=180:t=q:w=1.2:g=2"),
    "sad":        ("doc_truyen", 0.75, 0.90,  0.0, -2.0, ""),
    "dramatic":   ("tu_nhien",   0.85, 0.93,  0.0, +0.5, "equalizer=f=150:t=q:w=1:g=2.5"),
    "whispering": ("doc_truyen", 0.70, 0.97,  0.0, -4.0,          # EL gần như không thì thầm
                   "highpass=f=150,equalizer=f=5000:t=q:w=1.5:g=2"),  # thật: chỉ nhỏ & mỏng nhẹ
    "fast":       ("tu_nhien",   0.85, 1.15,   0.0,  0.0, ""),
}
ALIASES = {"whisper": "whispering", "whispers": "whispering", "happy": "excited",
           "cheerful": "playful", "surprised": "amazed", "shocked": "amazed",
           "calm": "neutral", "narration": "neutral"}
# Tag ÂM THANH (không phải cảm xúc) → token gốc của VieNeu, giữ đúng vị trí trong câu.
SOUND_TAGS = {
    "cười": "[cười]", "laugh": "[cười]", "laughs": "[cười]", "laughing": "[cười]",
    "laughs softly": "[cười]", "giggle": "[cười]", "giggles": "[cười]", "chuckle": "[cười]",
    "chuckles": "[cười]", "snickers": "[cười]",
    "thở dài": "[thở dài]", "sigh": "[thở dài]", "sighs": "[thở dài]", "exhales": "[thở dài]",
    "hắng giọng": "[hắng giọng]", "clear throat": "[hắng giọng]", "clears throat": "[hắng giọng]",
}
# "Hahahaaa", "hihi", "hehehe" đọc thành chữ nghe rất máy → đổi thành tiếng cười thật
LAUGH_RE = re.compile(r"(?<!\w)(?:h[aeiê]+){2,}\w*[.!…]*", re.IGNORECASE)

TAG_RE = re.compile(r"\[([^\]]+)\]")

# tuning.json do `./rv bench --apply` sinh ra (đo so với ElevenLabs): ghi đè tempo/gain
# theo từng tag + hệ số giãn khoảng nghỉ. Không có file → dùng số mặc định ở trên.
TUNING_PATH = ROOT / "tuning.json"
GAP_SCALE = 1.0
INNER_PAUSE_SCALE = 1.0    # kéo dài khoảng lặng BÊN TRONG 1 lần đọc (nhiều câu gộp)
if TUNING_PATH.exists():
    _tn = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    for _tag, _v in (_tn.get("tags") or {}).items():
        if _tag in PROFILES:
            _s, _t, _tp, _p, _g, _eq = PROFILES[_tag]
            PROFILES[_tag] = (_s, _t, _v.get("tempo", _tp), _p, _v.get("gain", _g), _eq)
    GAP_SCALE = float(_tn.get("gap_scale", 1.0))
    INNER_PAUSE_SCALE = float(_tn.get("inner_pause_scale", 1.0))
# Nghỉ sau câu, theo dấu kết thúc câu (giây). Dòng trống = sang đoạn mới.
# Đo trên bản EL bài son (2026-09-27): nghỉ TB 0.45s, giữa các dòng 0.32-0.60s, tổng
# im lặng 6.4s/32s. Bản cũ 10.9s vì VieNeu tự để ~0.30s lặng cuối mỗi đoạn + gap cộng
# dồn → nghe "ngắt từng câu, hụt năng lượng". Nay cắt lặng mỗi đoạn (_post) rồi mới cộng gap.
GAP_COMMA, GAP_SENT, GAP_DOTS, GAP_LINE, GAP_PARA = 0.12, 0.30, 0.40, 0.42, 0.60


# ─── parse ────────────────────────────────────────────────────────────────────

def _profile(tag: str) -> str | None:
    """Tag cảm xúc → tên profile; tag lạ → None (bỏ qua, KHÔNG reset về neutral)."""
    t = ALIASES.get(tag.strip().lower(), tag.strip().lower())
    return t if t in PROFILES else None


def _lexicon() -> dict[str, str]:
    p = ROOT / "lexicon.json"
    if not p.exists():
        return {}
    return {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if not k.startswith("_")}


# Viết tắt mạng → chữ đầy đủ (đo bằng bộ chuẩn hoá VieNeu: "ko"→đọc kiểu Anh "kâu", "đc"→"đ xê",
# "j"→"giây", "vs"→"vi ét", "mn"→"em en", "zợ"→"zét ợ"). "siu/lun/hông/trùi ui" đọc đúng → giữ.
TEENCODE = {
    "ko": "không", "kh": "không", "hk": "không", "đc": "được", "dc": "được",
    "đk": "được", "j": "gì", "vs": "với", "mn": "mọi người", "mng": "mọi người", "ng": "người",
    "zợ": "vợ", "z": "vậy", "r": "rồi", "cx": "cũng", "bt": "bình thường", "ntn": "như thế nào",
    "sp": "sản phẩm", "trc": "trước", "đg": "đang", "ib": "inbox", "tr": "trời", "cmt": "comment",
}


def _norm_numbers(text: str) -> str:
    """Số kiểu bán hàng mà VieNeu đọc sai (đã đo)."""
    # 99k / 199K → "99 nghìn" (VieNeu đọc "k" thành "ca")
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s?[kK](?![a-zA-ZÀ-ỹ])", lambda m: m.group(1).replace(",", ".") + " nghìn", text)
    # ngày sale đôi 9.9 / 10.10 / 11.11 / 12.12 → "9 9" (không đọc "chấm")
    text = re.sub(r"(?<![\d.])(\d{1,2})\.(\1)(?![\d.])", r"\1 \2", text)
    # SPF50+ → "SPF 50 cộng" (liền chữ thì đọc từng chữ số "năm không")
    text = re.sub(r"\b(SPF|spf|PA)(\d+)", r"\1 \2", text)
    # x2, x3 → "gấp đôi/gấp ba"
    text = re.sub(r"(?<![\w])[xX]2(?!\d)", "gấp đôi", text)
    text = re.sub(r"(?<![\w])[xX](\d)(?!\d)", r"gấp \1", text)
    return text


def _clean(text: str, lex: dict[str, str]) -> str:
    """Chữ đưa vào TTS (SRT vẫn giữ chữ gốc). Từ điển không phân biệt hoa/thường."""
    text = _norm_numbers(text)
    for k, v in TEENCODE.items():                       # chỉ khớp NGUYÊN TỪ
        text = re.sub(rf"(?<![\w]){re.escape(k)}(?![\w])", v, text, flags=re.IGNORECASE)
    # "k" thường = "không" (anh k biết) — trừ Vitamin K / size K / K hoa
    text = re.sub(r"(?<!vitamin )(?<!size )(?<![\w])k(?=\s+[a-zà-ỹ])", "không", text)
    for k in sorted(lex, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", lex[k], text, flags=re.IGNORECASE)
    text = LAUGH_RE.sub("[cười]", text)
    text = re.sub(r"(\[cười\]\s*){2,}", "[cười] ", text)   # không cười 2 lần liền
    text = re.sub(r"(\w)\1{2,}(?=\W|$)", r"\1", text)      # nhaaaaa → nha
    text = re.sub(r"!{2,}", "!", text)
    # "nha" giữa câu thỉnh thoảng bị đọc thành "nhà" (1/6); có dấu phẩy sau → 6/6 đúng.
    # Trừ từ ghép: nha đam (mỹ phẩm!), nha khoa, nha sĩ, nha chu.
    text = re.sub(r"(?<!\w)(nha)\s+(?!đam|khoa|sĩ|chu)(?=\w)", r"\1, ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    out: list[str] = []
    for p in parts:
        if not re.search(r"\w", TAG_RE.sub("", p)):     # chỉ có [cười]… → dính vào câu trước
            if out:
                out[-1] += " " + p
            elif TAG_RE.search(p):
                out.append(p)
        else:
            out.append(p)
    return out


def _subtitle(text: str) -> str:
    """Phụ đề: bỏ tag âm thanh, giữ chữ."""
    return re.sub(r"\s+", " ", TAG_RE.sub("", text)).strip()


def gap_after(text: str) -> float:
    t = _subtitle(text).rstrip()
    if t.endswith(("...", "…")):
        return GAP_DOTS
    if t.endswith((",", ";", ":", "-", "–")):
        return GAP_COMMA
    return GAP_SENT


def parse(script: str) -> list[dict]:
    """→ [{para, tag, text}] mỗi phần tử = 1 câu.

    Luật tag cảm xúc:
      - Ở ĐẦU/GIỮA dòng: áp cho phần chữ phía sau và các dòng tiếp theo (như ElevenLabs).
      - Ở CUỐI dòng (sau nó không còn chữ): chỉ áp cho CHÍNH dòng đó, dòng sau quay về
        cảm xúc đang có — vì người viết đặt tag cuối là để mô tả câu vừa viết.
    Tag âm thanh ([giggles], [laughing]…) → [cười]/[thở dài] giữ nguyên chỗ.
    """
    out, mood, para, row = [], "neutral", 0, 0
    for raw in script.splitlines():
        if not raw.strip():
            para += 1
            continue
        # tag âm thanh → token VieNeu tại chỗ; tag lạ không phải cảm xúc → bỏ
        def sub(m):
            inner = m.group(1).strip().lower()
            if inner in SOUND_TAGS:
                return SOUND_TAGS[inner]
            return m.group(0) if _profile(inner) else " "
        line = TAG_RE.sub(sub, raw)
        emo = [(m, _profile(m.group(1))) for m in TAG_RE.finditer(line) if _profile(m.group(1))]
        # tag cảm xúc cuối dòng: sau nó chỉ còn khoảng trắng / tag âm thanh
        suffix = None
        if emo:
            m_last, p_last = emo[-1]
            if not re.search(r"\w", TAG_RE.sub("", line[m_last.end():])):
                suffix = p_last
                line = line[:m_last.start()] + line[m_last.end():]
                emo = emo[:-1]
        segs, pos, cur = [], 0, mood
        for m, p in emo:
            segs.append((cur, line[pos:m.start()]))
            cur, pos = p, m.end()
        segs.append((cur, line[pos:]))
        mood = cur                           # tag đầu/giữa dòng lan sang dòng sau
        for t, chunk in segs:
            for s in _sentences(chunk):
                out.append({"para": para, "row": row, "tag": suffix or t, "text": s.strip()})
        row += 1
    return out


def group_lines(lines: list[dict], max_chars: int = 220) -> list[dict]:
    """Gộp các câu liền nhau CÙNG dòng + CÙNG cảm xúc thành 1 lần đọc.

    Đọc từng câu rời → mỗi câu tự "reset" ngữ điệu (lên đầu, rơi cuối) nghe như đọc
    danh sách. Đọc cả cụm 1 lần → model nối mạch như người kể (giống EL đọc cả bài).
    """
    out: list[dict] = []
    for ln in lines:
        prev = out[-1] if out else None
        # Nối sang DÒNG SAU nếu dòng trước chưa "xong câu": kết thúc bằng dấu phẩy, hoặc
        # quá ngắn. Đo thật: "Thiết kế xinh xẻo," đọc riêng → «xẻo» sai thanh 5/5 lần (model
        # hạ giọng kết câu ngay chữ cuối, hỏi thành huyền); nối với câu sau → đúng 4/5.
        unfinished = prev and (_subtitle(prev["text"]).rstrip().endswith((",", ";", ":", "-", "–"))
                               or len(_subtitle(prev["text"]).split()) < 4)
        if (prev and prev["tag"] == ln["tag"] and prev.get("para") == ln.get("para")
                and (prev["row"] == ln["row"] or unfinished)
                and len(prev["text"]) + len(ln["text"]) < max_chars):
            prev["text"] += " " + ln["text"]
        else:
            out.append(dict(ln))
    return out


# ─── audio ────────────────────────────────────────────────────────────────────

def _ff(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg lỗi: {r.stderr[-600:]}")


def _stretch_pauses(path: Path, scale: float, min_run: float = 0.12, cap: float | None = None) -> None:
    """Kéo dài khoảng lặng BÊN TRONG đoạn (không đụng 2 đầu) × scale.

    Đọc gộp nhiều câu 1 lần cho liền mạch, nhưng model tự nghỉ giữa câu rất ngắn (~0.3s),
    EL nghỉ ~0.64s. Chèn thêm lặng vào giữa mỗi quãng lặng ≥ min_run.
    """
    if scale <= 1.01 and cap is None:
        return
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    hop = int(sr * 0.01)
    n = len(y) // hop
    if n < 10:
        return
    e = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(e + 1e-7)
    quiet = db < db.max() - 38
    voiced = np.where(~quiet)[0]
    if len(voiced) < 2:
        return
    first, last = voiced[0], voiced[-1]
    pieces, prev, i = [], 0, first
    while i < last:
        if quiet[i]:
            j = i
            while j < last and quiet[j]:
                j += 1
            run = (j - i) * 0.01
            if run >= min_run:
                target = run * scale if cap is None else min(run, cap)
                mid = (i + j) // 2 * hop
                if target >= run:                       # kéo dài: chèn lặng vào giữa
                    pieces.append(y[prev:mid])
                    pieces.append(np.zeros(int((target - run) * sr), dtype=np.float32))
                    prev = mid
                else:                                   # rút ngắn: cắt bớt phần giữa quãng lặng
                    cut = int((run - target) * sr)
                    pieces.append(y[prev:mid - cut // 2])
                    prev = mid + (cut - cut // 2)
            i = j
        else:
            i += 1
    pieces.append(y[prev:])
    sf.write(str(path), np.concatenate(pieces), sr)


def _post(raw: Path, out: Path, tag: str, speed: float, pace: str = "kenh") -> None:
    pc = PACES.get(pace, PACES["kenh"])
    if pc["inner_max"] is None:
        _stretch_pauses(raw, INNER_PAUSE_SCALE)
    else:
        _stretch_pauses(raw, 1.0, cap=pc["inner_max"] * pc["speed"])   # đo trước khi tăng tốc
    speed = speed * pc["speed"]
    _, _, tempo, pitch, gain, eq = PROFILES[tag]
    trim = ("silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.03,areverse,"
            "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05,areverse")
    chain = [trim, f"rubberband=tempo={tempo * speed * BASE_TEMPO:.3f}:pitch={2 ** (pitch / 12):.4f}:formant=preserved"]
    if eq:
        chain.append(eq)
    chain.append(f"volume={gain:+.1f}dB")
    _ff("-i", str(raw), "-af", ",".join(chain), "-ar", str(SR), "-ac", "1", str(out))


# Tag chưa có ref riêng → mượn ref cảm xúc gần nhất (nếu giọng có kho cảm xúc).
EMO_FALLBACK = {"amazed": "excited", "fast": "excited", "curious": "playful",
                "whispering": "playful", "serious": "confident", "dramatic": "confident"}
# Tag cần bay bổng → chấm điểm nặng độ lên-xuống giọng; thì thầm/buồn → chỉ cần giống giọng.
EXPRESSIVE = {"excited": 0.6, "playful": 0.6, "amazed": 0.6, "curious": 0.4, "dramatic": 0.4,
              "fast": 0.3, "confident": 0.3, "neutral": 0.2, "serious": 0.1}


HOT_WORDS = ("xỉu", "dữ", "luôn", "trời ơi", "liền", "quá", "siêu", "cực")
# Câu cảm thán mà tag chưa có ref "_hi" riêng → mượn excited_hi (chỉ tag năng động)
HI_FALLBACK = {"playful": "excited_hi", "curious": "excited_hi", "amazed": "excited_hi", "fast": "excited_hi"}


CTA_RE = re.compile(r"giỏ hàng|bấm (vô|vào)|múc liền|mua liền|chốt đơn|đặt hàng|link (ở|dưới)", re.IGNORECASE)


def is_cta(text: str) -> bool:
    """Câu kêu gọi mua. Đo 8 file EL: câu CTA ở mức giọng BÌNH THƯỜNG (−0.4 nửa cung so với
    cả bài) dù có '!', còn câu cảm thán khác +1.8 → Adam reo khi khen, nói chắc khi chốt."""
    return bool(CTA_RE.search(_subtitle(text)))


def is_hot(text: str) -> bool:
    """Câu cảm thán? Đo trên 74 câu EL: kết '!' → cao hơn TB cả bài +2.3 nửa cung,
    ≥2 từ cảm thán → +2.15; câu thường −0.35. EL tự lên giọng, tool phải chọn ref cao."""
    t = _subtitle(text).lower()
    return bool(re.search(r"!\s*(\[[^\]]*\]\s*)*$", t)) or "!" in t or sum(w in t for w in HOT_WORDS) >= 2


def _voice_for(voice, tag: str, text: str = ""):
    """Chọn ref: câu cảm thán → <tag>_hi (giọng Adam cao TỰ NHIÊN cắt từ câu '!' của EL);
    không thì ref cảm xúc → ref gốc."""
    if not isinstance(voice, dict):
        return voice
    emos = voice.get("emotions") or {}
    if text and is_cta(text) and tag in ("excited", "amazed", "playful", "fast") and emos.get("playful"):
        return emos["playful"]
    if text and is_hot(text):
        hi = emos.get(f"{tag}_hi") or emos.get(HI_FALLBACK.get(tag, ""))
        if hi:
            return hi
    e = emos.get(tag) or emos.get(EMO_FALLBACK.get(tag, ""))
    return e or voice


def _score(wav, sr: int, tag: str, n_chars: int, base_emb) -> tuple[float, dict]:
    """Điểm 1 bản đọc: giống giọng + lên-xuống giọng (theo tag) − phạt bản hỏng."""
    import numpy as np
    import librosa
    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    # yin nhanh ~10x pyin; lọc khung không có tiếng bằng năng lượng
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=16000, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    f0 = f0[rms > rms.max() * 0.15]
    f0std = 0.0
    if len(f0) > 10:
        st = 12 * np.log2(f0 / np.median(f0))
        st = st[np.abs(st) < 7]                   # bỏ nhảy quãng tám (lỗi pyin / tiếng rè)
        f0std = float(min(np.std(st), 4.5)) if len(st) > 10 else 0.0
    sim = 0.0
    if base_emb is not None:
        enc = vb._engine().engine._ensure_speaker_encoder()
        e = np.asarray(enc.embed(y, 16000)).reshape(-1)
        b = np.asarray(base_emb).reshape(-1)
        sim = float(e @ b / (np.linalg.norm(e) * np.linalg.norm(b)))
    dur = len(wav) / sr
    per_char = dur / max(1, n_chars)
    broken = per_char > 0.13 or per_char < 0.03        # lặp/ngậm chữ hoặc nuốt câu
    score = EXPRESSIVE.get(tag, 0.0) * f0std - (5 if broken else 0)
    hz = float(np.median(f0)) if len(f0) > 10 else 0.0
    return score, {"sim": round(sim, 3), "f0std": round(f0std, 2), "dur": round(dur, 2),
                   "broken": broken, "hz": round(hz, 1)}


_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
EXTRA_TAKES = 2
FAST = True   # dừng ở bản đầu tiên đọc đúng chữ; False = luôn sinh đủ `takes` bản rồi chọn


def _norm_vi(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", t.lower())
    t = re.sub(r"(\w)\1{2,}", r"\1", t)                  # nhaaa → nha
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _base(ch: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFD", ch)[0].replace("đ", "d")


def fold_north(words: list[str]) -> list[str]:
    """Gộp các âm đầu giọng Bắc đọc GIỐNG NHAU để chỉ bắt lỗi thật (sai thanh, sai vần).

    VieNeu phiên âm giọng Bắc: gi/d/r → z, x → s, tr → ch. "giỏ hàng" đọc thành [zɔ4] là
    ĐÚNG giọng Bắc, nhưng Whisper phải chọn 1 cách viết ("dỏ", "rõ"…) → không tính là sai.
    """
    out = []
    for w in words:
        if w.startswith("gi") and len(w) > 2 and _base(w[2]) in "aeiouy":
            w = "z" + w[2:]                      # giỏ → zỏ (gi là âm đầu)
        elif w.startswith("gi"):
            w = "z" + w[1:]                      # gì → zì (i là vần)
        w = re.sub(r"^[dr]", "z", w)
        w = re.sub(r"^tr", "ch", w)
        w = re.sub(r"^x", "s", w)
        out.append(w)
    return out


def _asr_err(wav, sr: int, *targets: str) -> tuple[float, str]:
    """Tỉ lệ sai chữ (0 = đúng hết) khi Whisper nghe lại. Giữ dấu: sai thanh = sai."""
    import difflib
    import librosa
    import mlx_whisper
    import numpy as np
    y = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    # Whisper chập chờn với đoạn bị cắt sát âm đầu / nhỏ tiếng ("Thiết"→"Kết", câu thì thầm
    # "Cây son này"→"Peace online") → chuẩn đỉnh + đệm 0.3s lặng 2 đầu trước khi nghe.
    y = y / (np.abs(y).max() + 1e-6) * 0.9
    pad = np.zeros(int(0.3 * 16000), dtype=np.float32)
    y = np.concatenate([pad, y, pad])
    heard = mlx_whisper.transcribe(y, path_or_hf_repo=_ASR_MODEL, language="vi",
                                   condition_on_previous_text=False)["text"]
    # Tính theo TỪ (WER), không theo ký tự: sai 1 dấu thanh ("này"→"nay") chỉ lệch ~2%
    # ký tự nên lọt cổng, nhưng là 1 từ sai nguyên — đúng cái tai người nghe bắt được.
    h = fold_north(_norm_vi(heard).split())
    def wer(t):
        ref = fold_north(_norm_vi(t).split())
        sm = difflib.SequenceMatcher(None, ref, h, autojunk=False)
        # Whisper tự viết số ("29", "450ml") cho chữ số đọc đúng → không tính sai
        wrong = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes()
                    if op != "equal" and not any(ch.isdigit() for x in h[j1:j2] for ch in x))
        return wrong / max(1, len(ref))
    err = min(wer(t) for t in targets if _norm_vi(t))
    return round(err, 3), heard.strip()


from concurrent.futures import ThreadPoolExecutor
_POOL = ThreadPoolExecutor(max_workers=1)   # 1 luồng sinh giọng (engine VieNeu không chia sẻ được)


def _wrong_words(target: str, heard: str) -> frozenset:
    """Tập chữ trong kịch bản bị nghe sai (đã gộp giọng Bắc)."""
    import difflib
    ref = fold_north(_norm_vi(target).split())
    h = fold_north(_norm_vi(heard).split())
    sm = difflib.SequenceMatcher(None, ref, h, autojunk=False)
    return frozenset(w for op, i1, i2, _, _ in sm.get_opcodes() if op != "equal" for w in ref[i1:i2])


def _synth(line: dict, voice, raw: Path, takes: int = 1) -> None:
    """Sinh `takes` bản với seed khác nhau, giữ bản điểm cao nhất."""
    tts = vb._engine()
    style, temp, *_ = PROFILES[line["tag"]]
    ref = _voice_for(voice, line["tag"], line["text"])
    base_emb = voice.get("speaker_emb") if isinstance(voice, dict) else None
    n_chars = len(re.sub(r"\[[^\]]*\]|\W", "", line["tts"]))
    cands = []
    max_k = max(1, takes) + (EXTRA_TAKES if takes > 1 else 0)

    def gen(k):                                 # chạy ở luồng phụ (CPU/ONNX)
        seed = line["seed"] + k * 7919
        vb._seed(seed)
        return seed, tts.infer(line["tts"], voice=ref, style=style, temperature=temp + TEMP_BOOST)

    # SONG SONG: VieNeu (CPU) sinh bản k+1 trong lúc Whisper (GPU/MLX) soát bản k.
    fut, k, limit, prev_heard = _POOL.submit(gen, 0), 0, max(1, takes), None
    while True:
        seed, wav = fut.result()
        k += 1
        if takes == 1:
            cands.append((0.0, wav, seed, {"sim": 1.0}))
            break
        nxt = _POOL.submit(gen, k) if k < max_k else None
        sc, info = _score(wav, tts.sample_rate, line["tag"], n_chars, base_emb)
        if len(_norm_vi(line["tts"]).split()) >= 2:      # câu chỉ có tiếng cười → không soát chữ
            info["err"], info["heard"] = _asr_err(wav, tts.sample_rate, line["text"], line["tts"])
        else:
            info["err"], info["heard"] = 0.0, ""
        cands.append((sc, wav, seed, info))
        wrong = _wrong_words(line["tts"], info["heard"]) if info["err"] > 0 else frozenset()
        stop = (FAST and info["err"] == 0 and not info["broken"])       # đọc đúng hết → dùng luôn
        # 2 bản liền nhau sai CÙNG chữ ("tuýp"→"tuyếp") → lỗi cố hữu (model/Whisper), thêm vô ích
        stop = stop or (FAST and bool(wrong) and prev_heard is not None and wrong <= prev_heard)
        if k >= limit:
            if limit < max_k and min(c[3]["err"] for c in cands) > 0:
                limit += 1                          # mọi bản còn sai chữ → thử thêm
            else:
                stop = True
        prev_heard = wrong if info["err"] > 0 else None
        if stop or nxt is None:
            if nxt is not None:
                nxt.cancel()                        # bản dự phòng: bỏ (nếu chưa chạy)
            break
        fut = nxt
    # 3 cổng theo thứ tự ưu tiên:
    #  1. ĐỌC ĐÚNG CHỮ: bản bay bổng hay đọc sai thanh ("căng"→"càng") — loại trước
    #  2. GIỐNG ADAM: chỉ giữ bản gần bằng bản giống nhất (≤0.04)
    #  3. NHẤN NHÁ: trong số còn lại chọn bản lên-xuống giọng nhiều nhất (theo tag)
    best_err = min(c[3]["err"] for c in cands)
    ok = [c for c in cands if c[3]["err"] <= best_err + 1e-9]      # chỉ giữ bản ít từ sai NHẤT
    top_sim = max(c[3]["sim"] for c in ok)
    ok = [c for c in ok if c[3]["sim"] >= top_sim - 0.04] or ok
    _, wav, seed, info = max(ok, key=lambda c: c[0])
    line["take_seed"], line["qa"] = seed, info
    info["takes"] = len(cands)
    line["ref"] = next((k for k, v in (voice.get("emotions") or {}).items() if v is ref), "base") \
        if isinstance(voice, dict) else "preset"
    tts.save(wav, raw)


def _voice(name: str):
    return name if not (vb.VOICES / f"{name}.json").exists() else vb._load_voice(name)


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _srt_pieces(text: str, start: float, end: float, max_chars: int = 48) -> list[tuple]:
    """Câu dài → tách theo dấu phẩy, chia thời gian theo số ký tự."""
    parts = [p.strip() for p in re.split(r"(?<=,)\s+", text) if p.strip()]
    pieces, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            pieces.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        pieces.append(cur)
    # Đoạn vẫn dài (không có dấu phẩy) → chia đều theo từ
    split = []
    for p in pieces:
        words = p.split()
        n = -(-len(p) // max_chars)
        size = -(-len(words) // n)
        split += [" ".join(words[k:k + size]) for k in range(0, len(words), size)]
    pieces = split
    total = sum(len(p) for p in pieces) or 1
    res, t = [], start
    for p in pieces:
        d = (end - start) * len(p) / total
        res.append((t, t + d, p))
        t += d
    return res


def assemble(d: Path, man: dict) -> None:
    """Ghép lines/ + khoảng nghỉ → full.wav/mp3/srt, chuẩn loudness chung 1 gain."""
    lines = man["lines"]
    concat = d / "_concat.txt"
    sil = {g: d / f"_sil_{int(g * 1000)}.wav" for g in (GAP_COMMA, GAP_SENT, GAP_DOTS, GAP_LINE, GAP_PARA)}
    gmax = PACES.get(man.get("pace", "el"), PACES["el"])["gap_max"]
    eff = {g: (min(g * GAP_SCALE, gmax) if gmax else g * GAP_SCALE) for g in sil}
    for g, p in sil.items():
        _ff("-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=mono", "-t", f"{eff[g]:.3f}", str(p))
    entries, srt, t = [], [], 0.0
    for i, ln in enumerate(lines):
        f = d / "stems" / f"{i + 1:02d}.wav"
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout)
        entries.append(f"file '{f}'")
        if _subtitle(ln["text"]):
            srt += _srt_pieces(_subtitle(ln["text"]), t, t + dur)
        ln["start"], ln["end"] = round(t, 3), round(t + dur, 3)
        t += dur
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt["para"] != ln["para"]:
                g = GAP_PARA
            elif nxt.get("row", ln.get("row")) != ln.get("row"):
                g = max(GAP_LINE, gap_after(ln["text"]))
            else:
                g = gap_after(ln["text"])
            entries.append(f"file '{sil[g]}'")
            t += eff[g]
    concat.write_text("\n".join(entries), encoding="utf-8")
    raw = d / "_full_raw.wav"
    _ff("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(raw))

    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af",
                        "loudnorm=print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    measured = float(json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])["input_i"])
    gain = TARGET_LUFS - measured
    master = f"volume={gain:+.2f}dB,alimiter=limit=0.89:level=false"
    _ff("-i", str(raw), "-af", master, str(d / "full.wav"))
    _ff("-i", str(d / "full.wav"), "-b:a", "192k", str(d / "full.mp3"))
    # lines/ cũng nhận CÙNG gain → câu lẻ kéo vào CapCut khớp âm lượng bản full
    for i in range(len(lines)):
        src = d / "stems" / f"{i + 1:02d}.wav"
        dst = d / "lines" / f"{i + 1:02d}.wav"
        _ff("-i", str(src), "-af", master, str(dst))

    (d / "full.srt").write_text("\n".join(
        f"{k}\n{_srt_time(a)} --> {_srt_time(b)}\n{txt}\n" for k, (a, b, txt) in enumerate(srt, 1)),
        encoding="utf-8")
    for p in [concat, raw, *sil.values()]:
        p.unlink(missing_ok=True)
    man["duration"], man["gain_db"] = round(t, 2), round(gain, 2)
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {d}/full.wav  ({t:.1f}s, {len(lines)} câu, {len(srt)} dòng SRT)")


def _render_line(d: Path, i: int, ln: dict, voice, speed: float, takes: int = 1, pace: str = "kenh") -> None:
    raw = d / "stems" / f"_raw_{i + 1:02d}.wav"
    t = time.time()
    _synth(ln, voice, raw, takes)
    _post(raw, d / "stems" / f"{i + 1:02d}.wav", ln["tag"], speed, pace)   # chưa master
    raw.unlink(missing_ok=True)
    qa = ln.get("qa") or {}
    extra = (f" sim={qa['sim']:.2f} lênxuống={qa['f0std']:.1f} sai={qa['err']:.0%} ({qa.get('takes', '?')} bản)"
             if "err" in qa else "")
    print(f"  {i + 1:02d} [{ln['tag']:<10}←{ln.get('ref', '-'):<9}] {time.time() - t:4.1f}s{extra}  {ln['text'][:50]}",
          flush=True)


# ─── API (CLI + web dùng chung) ───────────────────────────────────────────────

def render_text(script: str, voice: str, name: str, speed: float = 1.0, seed: int = 42,
                on_line=None, source: str = "", takes: int = 3, group: bool = True,
                pace: str = "kenh") -> dict:
    """Render cả kịch bản. on_line(i, total, line) gọi sau mỗi câu (cho thanh tiến độ)."""
    lines = parse(script)
    if group:
        lines = group_lines(lines)
    if not lines:
        raise ValueError("Kịch bản trống.")
    lex = _lexicon()
    for i, ln in enumerate(lines):
        ln["tts"], ln["seed"] = _clean(ln["text"], lex), seed + i
    d = vb.OUT / name
    for sub in ("lines", "stems"):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for old in (d / sub).glob("*.wav"):
            old.unlink()
    (d / "script.txt").write_text(script, encoding="utf-8")
    man = {"voice": voice, "speed": speed, "takes": takes, "pace": pace, "script": source, "lines": lines}
    v = _voice(voice)
    print(f"Giọng: {voice} · {len(lines)} câu")
    for i, ln in enumerate(lines):
        _render_line(d, i, ln, v, speed, takes, pace)
        if on_line:
            on_line(i + 1, len(lines), ln)
    assemble(d, man)
    return man


def load_manifest(name: str) -> dict:
    return json.loads((vb.OUT / name / "manifest.json").read_text(encoding="utf-8"))


def redo_line(name: str, line: int, *, seed: int | None = None, tag: str | None = None,
              tts: str | None = None, resynth: bool = True, subtitle: str | None = None) -> dict:
    """Sửa 1 câu (line đánh số từ 1). resynth=False: chỉ đổi phụ đề rồi ghép lại."""
    d = vb.OUT / name
    man = load_manifest(name)
    i = line - 1
    ln = man["lines"][i]
    if subtitle is not None:
        ln["text"] = subtitle.strip()
    if resynth:
        ln["seed"] = seed if seed is not None else ln["seed"] + 1000
        if tag:
            ln["tag"] = _profile(tag) or ln["tag"]
        if tts is not None:
            ln["tts"] = _clean(tts, _lexicon())
        _render_line(d, i, ln, _voice(man["voice"]), man["speed"], man.get("takes", 3), man.get("pace", "el"))
    assemble(d, man)
    return man


# ─── commands ─────────────────────────────────────────────────────────────────

def cmd_render(a) -> None:
    src = Path(a.script)
    render_text(src.read_text(encoding="utf-8"), a.voice, a.name, a.speed, a.seed,
                source=str(src.resolve()), takes=a.takes, group=not a.no_group, pace=a.pace)


def cmd_redo(a) -> None:
    redo_line(a.name, a.line, seed=a.seed, tag=a.tag, tts=a.text)


def main() -> int:
    p = argparse.ArgumentParser(description="Render kịch bản tag → audio + SRT cho CapCut")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("render")
    s.add_argument("script")
    s.add_argument("--voice", required=True, help="giọng đã clone (voices/<x>.json) hoặc preset VieNeu")
    s.add_argument("--name", required=True)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--takes", type=int, default=3, help="số bản sinh mỗi câu, giữ bản hay nhất")
    s.add_argument("--no-group", action="store_true", help="đọc từng câu rời (kiểu cũ)")
    s.add_argument("--pace", default="kenh", choices=list(PACES), help="kenh = nhịp video trên kênh (mặc định), el = file EL gốc")
    s.set_defaults(fn=cmd_render)
    s = sub.add_parser("redo", help="đọc lại 1 câu")
    s.add_argument("--name", required=True)
    s.add_argument("--line", type=int, required=True)
    s.add_argument("--seed", type=int)
    s.add_argument("--tag", help="đổi cảm xúc câu này")
    s.add_argument("--text", help="sửa chữ TTS đọc (SRT giữ nguyên)")
    s.set_defaults(fn=cmd_redo)
    a = p.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
