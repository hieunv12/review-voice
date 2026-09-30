"""SO VỚI ELEVENLABS & TỰ CHỈNH — cùng 1 kịch bản: bản EL (chuẩn) vs bản tool.

  ./rv bench samples/el_calib.mp3 samples/calib_script.txt            # đo + báo cáo
  ./rv bench samples/el_calib.mp3 samples/calib_script.txt --refs     # + lấy ref cảm xúc còn thiếu từ bản EL
  ./rv bench samples/el_calib.mp3 samples/calib_script.txt --apply    # + ghi tuning.json rồi render/đo lại

Cả 2 file đều được Whisper nghe (có mốc thời gian từng từ) rồi căn với kịch bản theo
từng CÂU → cùng đơn vị để so: tốc độ, cao độ, độ lên-xuống, độ to, khoảng nghỉ, sai chữ.
"""
from __future__ import annotations

import argparse
import difflib
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

import render as R
import voicebuild as vb

ROOT = Path(__file__).resolve().parent
SR, HOP = 16000, 160


# ─── đo ───────────────────────────────────────────────────────────────────────

def _units(script: str) -> list[dict]:
    """Đơn vị so sánh = câu (parse, chưa gộp) + danh sách từ chuẩn hoá."""
    out = []
    for u in R.parse(script):
        words = R.fold_north(R._norm_vi(R._subtitle(u["text"])).split())
        if words:
            out.append({**u, "words": words})
    return out


def _load(path: str):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def _align(y, units: list[dict]) -> tuple[list[tuple], float]:
    """Whisper (mốc từng từ) → (start, end) cho từng câu + tỉ lệ từ sai toàn bài."""
    import mlx_whisper
    res = mlx_whisper.transcribe(y, path_or_hf_repo=R._ASR_MODEL, language="vi", word_timestamps=True,
                                 condition_on_previous_text=False)
    heard = []
    for s in res["segments"]:
        for w in s.get("words", []):
            for tok in R.fold_north(R._norm_vi(w["word"]).split()):
                heard.append((tok, w["start"], w["end"]))
    ref = [(w, i) for i, u in enumerate(units) for w in u["words"]]
    sm = difflib.SequenceMatcher(None, [w for w, _ in ref], [h[0] for h in heard], autojunk=False)
    spans: dict[int, list] = {}
    wrong = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                spans.setdefault(ref[i1 + k][1], []).append(heard[j1 + k])
        else:
            wrong += max(i2 - i1, j2 - j1)
            # từ sai vẫn dùng làm mốc thời gian nếu độ dài khớp (sai thanh vẫn đúng vị trí)
            if op == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    spans.setdefault(ref[i1 + k][1], []).append(heard[j1 + k])
    out = []
    for i in range(len(units)):
        hs = spans.get(i)
        out.append((hs[0][1], hs[-1][2]) if hs and len(hs) >= max(1, len(units[i]["words"]) // 3) else None)
    return out, wrong / max(1, len(ref))


def _measure(y, span) -> dict | None:
    import librosa
    if not span:
        return None
    a, b = int(span[0] * SR), int(span[1] * SR)
    seg = y[a:b]
    if len(seg) < SR * 0.3:
        return None
    f0, vf, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    f = f0[vf & ~np.isnan(f0)]
    rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    loud = float(np.mean(db[db > db.max() - 30]))
    if len(f) < 10:
        return {"dur": span[1] - span[0], "hz": np.nan, "move": np.nan, "loud": loud}
    s = 12 * np.log2(f / np.median(f))
    s = s[np.abs(s) < 7]
    return {"dur": span[1] - span[0], "hz": float(np.median(f)), "move": float(np.std(s)), "loud": loud}


def analyse(audio: str, units: list[dict]) -> dict:
    y = _load(audio)
    spans, wer = _align(y, units)
    per = [_measure(y, sp) for sp in spans]
    # độ to tương đối so với cả bài (2 file khác mức master → so tương đối)
    louds = [p["loud"] for p in per if p]
    base = st.median(louds) if louds else 0
    for p in per:
        if p:
            p["loud_rel"] = p["loud"] - base
    pauses = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)
              if spans[i] and spans[i + 1] and spans[i + 1][0] > spans[i][1]]
    return {"spans": spans, "per": per, "wer": wer, "pauses": pauses, "dur": len(y) / SR}


def by_tag(units, res) -> dict:
    g: dict[str, dict] = {}
    for u, p in zip(units, res["per"]):
        if not p:
            continue
        d = g.setdefault(u["tag"], {"words": 0, "dur": 0.0, "hz": [], "move": [], "loud": [], "spans": []})
        d["words"] += len(u["words"])
        d["dur"] += p["dur"]
        for k in ("hz", "move"):
            if not np.isnan(p[k]):
                d[k].append(p[k])
        d["loud"].append(p["loud_rel"])
    for d in g.values():
        d["rate"] = d["words"] / max(d["dur"], 0.1)
        d["hz"] = float(np.median(d["hz"])) if d["hz"] else np.nan
        d["move"] = float(np.mean(d["move"])) if d["move"] else np.nan
        d["loud"] = float(np.mean(d["loud"]))
    return g


# ─── báo cáo + chỉnh ──────────────────────────────────────────────────────────

def report(units, el, me) -> dict:
    ge, gm = by_tag(units, el), by_tag(units, me)
    print(f"\n{'TAG':11s} │ {'tốc độ từ/s':^13s} │ {'cao độ Hz':^11s} │ {'lên-xuống st':^13s} │ {'độ to dB':^11s}")
    print(f"{'':11s} │ {'EL':>5s} {'tool':>6s} │ {'EL':>4s} {'tool':>5s} │ {'EL':>5s} {'tool':>6s} │ {'EL':>4s} {'tool':>5s}")
    sugg = {}
    for tag in ge:
        e, m = ge[tag], gm.get(tag)
        if not m:
            continue
        print(f"{tag:11s} │ {e['rate']:5.1f} {m['rate']:6.1f} │ {e['hz']:4.0f} {m['hz']:5.0f} │ "
              f"{e['move']:5.2f} {m['move']:6.2f} │ {e['loud']:+4.1f} {m['loud']:+5.1f}")
        _, _, tempo, _, gain, _ = R.PROFILES[tag]
        sugg[tag] = {
            "tempo": round(float(np.clip(tempo * e["rate"] / m["rate"], 0.8, 1.25)), 3),
            "gain": round(float(np.clip(gain + (e["loud"] - m["loud"]), -9, 4)), 1),
            "pitch_gap_st": round(12 * np.log2(e["hz"] / m["hz"]), 1) if e["hz"] and m["hz"] else 0,
        }
    pe, pm = (st.median(x["pauses"]) if x["pauses"] else 0 for x in (el, me))
    print(f"\nKhoảng nghỉ giữa câu: EL trung vị {pe:.2f}s (tổng {sum(el['pauses']):.1f}s) · "
          f"tool {pm:.2f}s (tổng {sum(me['pauses']):.1f}s)")
    print(f"Độ dài bài: EL {el['dur']:.1f}s · tool {me['dur']:.1f}s")
    print(f"Sai chữ (Whisper): EL {el['wer']:.1%} · tool {me['wer']:.1%}   "
          f"(phần EL cũng sai = lỗi của Whisper, không phải giọng)")
    miss = [t for t, s in sugg.items() if abs(s["pitch_gap_st"]) > 1.2]
    if miss:
        print("Cao độ lệch >1.2 nửa cung (nên lấy ref cảm xúc từ EL): " +
              ", ".join(f"{t} ({sugg[t]['pitch_gap_st']:+.1f})" for t in miss))
    # So TỔNG thời gian nghỉ, chỉnh một nửa quãng (căn bậc 2) mỗi vòng — theo trung vị hay vọt.
    se, sm_ = sum(el["pauses"]), sum(me["pauses"])
    gap_scale = float(np.clip(np.sqrt(se / sm_), 0.7, 1.4)) if sm_ else 1.0
    return {"tags": sugg, "gap_scale_factor": gap_scale}


def extract_refs(el_audio: str, units, el, voice: str, only_missing: bool = True) -> None:
    """Cắt các câu EL cùng tag (tổng ≤8s) → ref cảm xúc cho giọng."""
    have = set((vb._load_voice(voice).get("emotions") or {}).keys())
    tags: dict[str, list] = {}
    for u, sp in zip(units, el["spans"]):
        if sp:
            tags.setdefault(u["tag"], []).append(sp)
    for tag, spans in tags.items():
        if tag == "neutral" or (only_missing and tag in have):
            continue
        picked, total = [], 0.0
        for a, b in spans:
            if total >= vb.REF_MAX - 0.8:
                break
            a, b = max(0, a - 0.05), b + 0.08
            if b - a < 0.8:                     # câu quá ngắn ("Mấy vợ ơi!") → bỏ qua, lấy câu sau
                continue
            if total + (b - a) > vb.REF_MAX:
                b = a + (vb.REF_MAX - total)
            picked.append((a, b))
            total += b - a
        if total < 3.5:
            print(f"  ! {tag}: EL chỉ có {total:.1f}s — quá ngắn để làm ref, bỏ qua")
            continue
        rng = ",".join(f"{a:.2f}-{b:.2f}" for a, b in picked)
        vb.cmd_emo(argparse.Namespace(name=voice, tag=tag, audio=el_audio, ranges=rng, denoise=False))


def main() -> int:
    p = argparse.ArgumentParser(description="So với ElevenLabs & tự chỉnh")
    p.add_argument("el_audio")
    p.add_argument("script")
    p.add_argument("--voice", default="adam")
    p.add_argument("--name", default="bench")
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--refs", action="store_true", help="lấy ref cảm xúc CÒN THIẾU từ bản EL")
    p.add_argument("--refs-all", action="store_true", help="lấy lại ref cho MỌI tag từ bản EL")
    p.add_argument("--apply", action="store_true", help="ghi tuning.json + render/đo lại")
    p.add_argument("--no-render", action="store_true", help="dùng bản render có sẵn out/<name>")
    a = p.parse_args()

    script = Path(a.script).read_text(encoding="utf-8")
    units = _units(script)
    print(f"Kịch bản: {len(units)} câu · nghe bản EL…", flush=True)
    el = analyse(a.el_audio, units)

    if a.refs or a.refs_all:
        extract_refs(a.el_audio, units, el, a.voice, only_missing=not a.refs_all)

    def run_tool(tag=""):
        if not a.no_render or tag:
            subprocess.run([sys.executable, str(ROOT / "render.py"), "render", a.script, "--voice", a.voice,
                            "--name", a.name + tag, "--takes", str(a.takes)], check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return analyse(str(vb.OUT / (a.name + tag) / "full.wav"), units)

    print("Render bản tool + nghe…", flush=True)
    me = run_tool()
    sugg = report(units, el, me)

    if a.apply:
        cur = json.loads(R.TUNING_PATH.read_text(encoding="utf-8")) if R.TUNING_PATH.exists() else {}
        tags = cur.get("tags", {})
        for t, s in sugg["tags"].items():
            tags[t] = {"tempo": s["tempo"], "gain": s["gain"]}
        f = sugg["gap_scale_factor"]
        new = {"tags": tags, "gap_scale": round(float(cur.get("gap_scale", 1.0)) * f, 3),
               "inner_pause_scale": round(float(np.clip(float(cur.get("inner_pause_scale", 1.0)) * f, 1.0, 3.0)), 3),
               "source": {"el": a.el_audio, "script": a.script}}
        R.TUNING_PATH.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ Ghi {R.TUNING_PATH.name}: gap_scale={new['gap_scale']} inner={new['inner_pause_scale']} · " +
              " ".join(f"{t}(tốc {v['tempo']}, to {v['gain']:+})" for t, v in tags.items()))
        print("Render lại với tham số mới + đo…", flush=True)
        me2 = run_tool("_tuned")
        report(units, el, me2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
