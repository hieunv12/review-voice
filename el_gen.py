"""Gen bản ElevenLabs CHUẨN qua API (để bench so sánh) — có cache, không gen lại cùng nội dung.

  ./rv elgen samples/test_bangdo.txt            → samples/el_test_bangdo.mp3

Giọng: Adam mặc định (premade, pNInz6obpgDQGcFmaJgB). Gói Free không gọi được giọng thư viện
"Adam - Dominant, Firm" qua API, nhưng đo speaker-embedding thì 2 giọng không phân biệt được
(0.73-0.81, trong khi Dominant vs Dominant 0.70). Cài đặt khớp bản user gen trên web:
eleven_v3, stability 0.5, similarity 0.75.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICE_ID = "pNInz6obpgDQGcFmaJgB"
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}
MODEL = "eleven_v3"


def _key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY", "")
    env = ROOT / ".env"
    if not k and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ELEVENLABS_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("Thiếu ELEVENLABS_API_KEY (.env)")
    return k


def generate(script_path: str, out: str | None = None) -> Path:
    text = Path(script_path).read_text(encoding="utf-8").strip()
    out_p = Path(out) if out else ROOT / "samples" / f"el_{Path(script_path).stem}.mp3"
    sig = hashlib.sha1(json.dumps([text, VOICE_ID, MODEL, SETTINGS]).encode()).hexdigest()[:12]
    meta = out_p.with_suffix(".json")
    if out_p.exists() and meta.exists() and json.loads(meta.read_text()).get("sig") == sig:
        print(f"= đã có {out_p.name} (cache)")
        return out_p
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
        data=json.dumps({"text": text, "model_id": MODEL, "voice_settings": SETTINGS}).encode(),
        headers={"xi-api-key": _key(), "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            audio, cost = r.read(), r.headers.get("character-cost")
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs lỗi {e.code}: {e.read().decode()[:300]}")
    out_p.write_bytes(audio)
    meta.write_text(json.dumps({"sig": sig, "cost": cost, "voice": VOICE_ID, "model": MODEL}))
    print(f"✓ {out_p.name} · tốn {cost} credit")
    return out_p


if __name__ == "__main__":
    for p in sys.argv[1:]:
        generate(p)
