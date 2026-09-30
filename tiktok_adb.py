"""Đăng video lên TikTok qua điện thoại Android cắm USB (adb + uiautomator).

  python tiktok_adb.py devices                      → liệt kê máy
  python tiktok_adb.py dump [serial]                → in các nút đang có trên màn hình (để dò bước)
  python tiktok_adb.py post <video> "<caption>" [private|friends|public] [serial]

Cần: USB debugging bật, máy mở khoá, TikTok đã đăng nhập, ADBKeyBoard đã cài (gõ tiếng Việt).
Các bước tìm nút theo chữ hiển thị (SEL) chứ không theo toạ độ → TikTok đổi giao diện thì sửa SEL.
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PKG = "com.ss.android.ugc.trill"
ADB_IME = "com.android.adbkeyboard/.AdbIME"
REMOTE_DIR = "/sdcard/DCIM/TTPost"
DEBUG_DIR = Path(__file__).resolve().parent / "work" / "tiktok_post" / "debug"

# regex khớp NGUYÊN chữ (text / content-desc / resource-id) của nút, không phân biệt hoa thường
SEL = {
    "create":     r"Quay|Create|Đăng video",
    "upload":     r"Tải lên|Upload|upload_hot_area",
    "next":       r"(Tiếp|Tiếp theo|Tiếp tục|Next)( \(\d+\))?",
    "post":       r"Đăng|Post",
    "caption":    r"Thêm mô tả.*|Mô tả.*|Add description.*|Describe.*",
    "privacy":    r".*(có thể xem bài đăng này|can view this post|can watch).*",
    "priv_opt":   {"public": r"Mọi người|Everyone",
                   "friends": r"Bạn bè.*|Friends.*",
                   "private": r"Chỉ mình bạn|Chỉ mình tôi|Only me|Only you"},
    # dòng quyền riêng tư sau khi chọn xong (để kiểm tra)
    "priv_ok":    {"public": r"Ai cũng có thể xem.*|Everyone can.*",
                   "friends": r"Bạn bè.*có thể xem.*|Friends can.*",
                   "private": r"Chỉ bạn có thể xem.*|Only you can.*"},
    "mention_btn": r"Nhắc đến|Mention",
    "mention_search": r"Tìm.*|Search.*",
    "login":      r"Đăng nhập vào TikTok|Đăng ký để bắt đầu sáng tạo|Log in to TikTok|Sign up.*",
    # hộp thoại quyền Android: chỉ cho phép thứ cần để đăng (camera / micro / ảnh-video), còn lại từ chối
    "perm_need":  r".*(ảnh và video|photos and videos|chụp ảnh|take pictures|ghi âm|record audio).*",
    "allow":      r"Cho phép tất cả|Allow all|Trong khi dùng ứng dụng|While using the app|Cho phép|Allow",
    "deny":       r"Không cho phép|Don.t allow|Deny",
    # popup khác chặn đường: bấm cho qua
    "dismiss":    r"Để sau|Không phải bây giờ|Not now|Bỏ qua|Skip|Đã hiểu|Tôi đã hiểu|OK|Got it",
}
BACK = "input keyevent KEYCODE_BACK"


class PostError(RuntimeError):
    pass


def adb_bin() -> str:
    return shutil.which("adb") or str(Path.home() / "Library/Android/sdk/platform-tools/adb")


def devices() -> list[dict]:
    try:
        out = subprocess.run([adb_bin(), "devices", "-l"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    res = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        kv = dict(p.split(":", 1) for p in parts[2:] if ":" in p)
        res.append({"serial": parts[0], "state": parts[1],
                    "model": kv.get("model", "").replace("_", " ") or parts[0]})
    return res


@dataclass
class Node:
    text: str
    desc: str
    rid: str
    cls: str
    clickable: bool
    bounds: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2

    def __str__(self):
        return f"{str(list(self.bounds)):24} {'C' if self.clickable else ' '} {self.cls:14} id={self.rid:16} t={self.text[:40]!r} d={self.desc[:40]!r}"


class Phone:
    def __init__(self, serial: str | None = None, log=print):
        devs = [d for d in devices() if d["state"] == "device"]
        if serial:
            if not any(d["serial"] == serial for d in devs):
                raise PostError(f"Không thấy máy {serial} (cắm USB + bật gỡ lỗi USB chưa?)")
        elif devs:
            serial = devs[0]["serial"]
        else:
            raise PostError("Không có điện thoại nào kết nối qua adb")
        self.serial, self.log = serial, log

    # ─── adb ──────────────────────────────────────────────
    def adb(self, *args, timeout=60, check=True) -> str:
        r = subprocess.run([adb_bin(), "-s", self.serial, *args], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise PostError(f"adb {' '.join(args[:3])}: {(r.stderr or r.stdout).strip()[:200]}")
        return r.stdout

    def sh(self, cmd: str, **kw) -> str:
        return self.adb("shell", cmd, **kw)

    # ─── màn hình ─────────────────────────────────────────
    def dump(self) -> list[Node]:
        for _ in range(3):                     # uiautomator hay lỗi "could not get idle state" khi đang có animation
            xml = self.sh("uiautomator dump /sdcard/.ttpost_ui.xml >/dev/null 2>&1; cat /sdcard/.ttpost_ui.xml",
                          check=False)
            if "<hierarchy" in xml:
                break
            time.sleep(0.8)
        else:
            return []
        nodes = []
        for n in ET.fromstring(xml[xml.index("<hierarchy"):]).iter("node"):
            a = n.attrib
            b = [int(v) for v in re.findall(r"\d+", a.get("bounds", "[0,0][0,0]"))]
            nodes.append(Node(a.get("text", ""), a.get("content-desc", ""), a.get("resource-id", "").split("/")[-1],
                              a.get("class", "").split(".")[-1], a.get("clickable") == "true", tuple(b)))
        return nodes

    def find(self, pattern: str, nodes: list[Node] | None = None, cls: str | None = None) -> Node | None:
        rx = re.compile(pattern, re.I)
        for n in nodes if nodes is not None else self.dump():
            if cls and n.cls != cls:
                continue
            if n.bounds[2] - n.bounds[0] < 2 or n.bounds[3] - n.bounds[1] < 2:
                continue
            if any(s and rx.fullmatch(s.strip()) for s in (n.text, n.desc, n.rid)):
                return n
        return None

    def tap(self, target: Node | tuple[int, int]):
        x, y = target.center if isinstance(target, Node) else target
        self.sh(f"input tap {x} {y}")

    def screenshot(self, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.run([adb_bin(), "-s", self.serial, "exec-out", "screencap", "-p"],
                                        capture_output=True, timeout=30).stdout)

    def wait(self, pattern: str, timeout=20, cls: str | None = None, popups=True) -> Node:
        """Chờ tới khi có nút khớp pattern; trong lúc chờ tự bấm qua các popup quen."""
        end = time.time() + timeout
        while time.time() < end:
            nodes = self.dump()
            hit = self.find(pattern, nodes, cls)
            if hit:
                return hit
            if self.find(SEL["login"], nodes):
                raise PostError("TikTok chưa đăng nhập trên máy này")
            if popups and self._clear_popup(nodes):
                continue
            time.sleep(1)
        raise PostError(f"Không thấy nút /{pattern}/ sau {timeout}s")

    def _clear_popup(self, nodes: list[Node]) -> bool:
        btns = [x for x in nodes if x.clickable or x.cls == "Button"]
        perm = next((x for x in nodes if x.rid == "permission_message"), None)
        if perm:
            key = "allow" if re.fullmatch(SEL["perm_need"], perm.text, re.I | re.S) else "deny"
            n = self.find(SEL[key], btns)
        elif any(x.rid == "add_item_title" for x in nodes):          # "Thêm widget vào Màn hình chờ?"
            n = next((x for x in btns if x.rid == "cancel_button"), None)
        else:
            n = self.find(SEL["dismiss"], btns)
        if not n:
            return False
        self.log(f"popup: {(perm.text[:50] + ' → ') if perm else ''}{n.text or n.desc}")
        self.tap(n)
        time.sleep(1.2)
        return True

    # ─── gõ chữ (tiếng Việt qua ADBKeyBoard) ──────────────
    def type_text(self, s: str):
        if ADB_IME.split("/")[0] not in self.sh("pm list packages com.android.adbkeyboard", check=False):
            raise PostError("Chưa cài ADBKeyBoard trên máy — cần để gõ caption tiếng Việt")
        old = self.sh("settings get secure default_input_method").strip()
        if old != ADB_IME:
            self.sh(f"ime enable {ADB_IME}; ime set {ADB_IME}")
            time.sleep(0.8)
        try:
            b64 = base64.b64encode(s.encode("utf-8")).decode()
            self.sh(f"am broadcast -a ADB_INPUT_B64 --es msg {b64}")
            time.sleep(0.8)
        finally:
            if old and old != ADB_IME and old != "null":
                self.sh(f"ime set {old}", check=False)

    # ─── video ────────────────────────────────────────────
    def push_video(self, local: Path) -> str:
        remote = f"{REMOTE_DIR}/ttpost_{int(time.time())}{local.suffix.lower()}"
        self.sh(f"mkdir -p {REMOTE_DIR}; find {REMOTE_DIR} -name 'ttpost_*' -mtime +2 -delete", check=False)
        self.adb("push", str(local), remote, timeout=600)
        self.sh(f"touch {remote}; am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{remote}")
        for _ in range(20):                    # chờ MediaStore nhận file → lên đầu thư viện
            if remote.rsplit("/", 1)[1] in self.sh(
                    f"content query --uri content://media/external/video/media --projection _data "
                    f"--where \"_data LIKE '%{remote.rsplit('/', 1)[1]}'\"", check=False):
                return remote
            time.sleep(0.5)
        raise PostError("Máy chưa nhận video vào thư viện (MediaStore)")

    def pushed_videos(self) -> list[dict]:
        """Các video công cụ đã chép vào máy (REMOTE_DIR/ttpost_*)."""
        out = self.sh(f"stat -c '%s %Y %n' {REMOTE_DIR}/ttpost_* 2>/dev/null", check=False)
        res = []
        for line in out.splitlines():
            size, mtime, path = (line.split(" ", 2) + ["", ""])[:3]
            if size.isdigit() and path.startswith(REMOTE_DIR):
                res.append({"name": path.rsplit("/", 1)[1], "size": int(size), "mtime": int(mtime)})
        return sorted(res, key=lambda f: -f["mtime"])

    def clear_pushed(self) -> int:
        """Xoá hết video đã chép + gỡ khỏi thư viện ảnh (MediaStore). Trả về số file đã xoá."""
        n = len(self.pushed_videos())
        self.sh(f"rm -f {REMOTE_DIR}/ttpost_*; content delete --uri content://media/external/video/media "
                f"--where \"_data LIKE '%/TTPost/ttpost_%'\"", check=False)
        return n

    def remove_video(self, remote: str):
        self.sh(f"rm -f {remote}; am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{remote}",
                check=False)

    def check_ready(self):
        if "mWakefulness=Awake" not in self.sh("dumpsys power"):
            self.sh("input keyevent KEYCODE_WAKEUP")
            time.sleep(1)
        if re.search(r"isKeyguardShowing=true|mDreamingLockscreen=true", self.sh("dumpsys window policy", check=False)):
            raise PostError("Điện thoại đang khoá màn hình — tắt khoá màn hình hoặc mở khoá trước giờ đăng")
        if PKG not in self.sh(f"pm list packages {PKG}"):
            raise PostError("Máy chưa cài TikTok")


def post(video: Path, caption: str, privacy: str = "public", serial: str | None = None,
         log=print, tag: str = "post", dry: bool = False, hashtags: str = "", mentions: str = "") -> str:
    """Đăng 1 video. Trả về chuỗi mô tả (TikTok không trả link ngay khi đăng qua app)."""
    ph = Phone(serial, log)
    step, remote = "chuẩn bị", None

    def go(name):
        nonlocal step
        step = name
        log(name)

    try:
        go("kiểm tra máy")
        ph.check_ready()
        duration = video_duration(Path(video))
        go("chép video vào máy")
        remote = ph.push_video(Path(video))

        go("mở TikTok")
        ph.sh(f"am force-stop {PKG}")
        ph.sh(f"monkey -p {PKG} -c android.intent.category.LAUNCHER 1", check=False)
        time.sleep(5)
        go("mở thư viện")
        _open_gallery(ph)

        go("chọn video")
        _pick_newest(ph, duration)

        go("qua màn chỉnh sửa")
        for _ in range(4):                     # Tiếp → (chỉnh sửa) → Tiếp → màn đăng
            nodes = ph.dump()
            cap = ph.find(SEL["caption"], nodes, cls="EditText") or next((n for n in nodes if n.cls == "EditText"), None)
            if cap and ph.find(SEL["post"], nodes):
                break
            ph.tap(ph.wait(SEL["next"], timeout=40))
            time.sleep(3)
        else:
            raise PostError("Không tới được màn hình đăng")

        men = [m.lstrip("@") for m in mentions.split() if m.lstrip("@")]
        missed = []
        if caption.strip() or men or hashtags.strip():
            go("nhập caption")
            missed = _type_caption(ph, cap, caption.strip(), men, hashtags.strip())

        if privacy in SEL["priv_opt"]:
            go("chọn quyền riêng tư")
            _set_privacy(ph, privacy)

        btn = ph.wait(SEL["post"], timeout=15)
        if dry:
            log("dry-run: dừng ở màn hình đăng, chưa bấm Đăng")
            return "dry-run" + (f" · không tìm thấy {missed}" if missed else "")
        go("bấm Đăng")
        ph.tap(btn)
        time.sleep(5)
        nodes = ph.dump()
        if ph.find(SEL["post"], nodes) and ph.find(r"Nháp|Drafts?", nodes):
            raise PostError("Bấm Đăng nhưng vẫn ở màn hình đăng")
        log("đã gửi lên TikTok")
        for _ in range(4):                     # sau khi đăng hay bật popup (widget, danh bạ, bảo mật…)
            if not ph._clear_popup(ph.dump()):
                time.sleep(1.5)
        # không xoá file ngay: TikTok còn đọc lúc tải lên nền; push_video() dọn file > 2 ngày
        return f"Đăng qua {ph.serial}" + (f" · không tìm thấy {' '.join('@' + m for m in missed)} (ghi dạng chữ)" if missed else "")
    except Exception as e:
        shot = DEBUG_DIR / f"{tag}_{int(time.time())}.png"
        try:
            ph.screenshot(shot)
            if remote:
                ph.remove_video(remote)
        except Exception:  # noqa: BLE001
            shot = None
        msg = f"[{step}] {e}" + (f" — ảnh: {shot.relative_to(DEBUG_DIR.parent.parent.parent)}" if shot else "")
        if step == "bấm Đăng":
            msg += " — CÓ THỂ ĐÃ ĐĂNG, xem TikTok trước khi Thử lại"
        raise PostError(msg) from e


def _open_gallery(ph: Phone, timeout=60):
    """TikTok mở lại có thể ở trang chủ, màn quay, hay thẳng thư viện → nhìn màn hình rồi đi tiếp."""
    end = time.time() + timeout
    while time.time() < end:
        nodes = ph.dump()
        if ph.find(SEL["login"], nodes):
            raise PostError("TikTok chưa đăng nhập trên máy này")
        nxt = ph.find(SEL["next"], nodes)
        if ph.find(r"Tất cả|All", nodes) and ph.find(r"Video|Videos", nodes) and nxt:
            n = re.search(r"\((\d+)\)", nxt.text)
            if not n:
                return
            ph.sh(BACK)   # thư viện còn tick sẵn từ lần trước → thoát ra chọn lại
            time.sleep(2)
            continue
        hit = ph.find(SEL["upload"], nodes) or ph.find(SEL["create"], nodes)
        if not hit and (nxt or any(n.cls == "EditText" for n in nodes) and ph.find(SEL["post"], nodes)):
            ph.sh(BACK)   # TikTok khôi phục màn chỉnh sửa / màn đăng dở → lùi ra
            time.sleep(2)
            continue
        if hit:
            ph.tap(hit)
            time.sleep(3)
        elif not ph._clear_popup(nodes):
            time.sleep(1.5)
    raise PostError("Không mở được thư viện chọn video")


def video_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise PostError(f"Không đọc được thời lượng video: {path.name}") from None


def _dur_labels(sec: float) -> set[str]:
    """Các cách TikTok có thể hiện thời lượng (làm tròn lên/xuống): 00:06, 1:02:03…"""
    res = set()
    for s in {int(sec), round(sec), int(sec) + 1}:
        h, m, x = s // 3600, s % 3600 // 60, s % 60
        res.add(f"{h:02d}:{m:02d}:{x:02d}" if h else f"{m:02d}:{x:02d}")
    return res


def _inside(a: Node, b: Node) -> bool:
    return a.bounds[0] >= b.bounds[0] and a.bounds[1] >= b.bounds[1] and a.bounds[2] <= b.bounds[2] and a.bounds[3] <= b.bounds[3]


def _pick_newest(ph: Phone, duration: float):
    """Tick video vừa chép (ô đầu lưới, đúng thời lượng) rồi kiểm lại khay đã chọn — sai là dừng, không đăng nhầm."""
    labels = _dur_labels(duration)
    tab = ph.find(r"Video|Videos", ph.dump())
    if tab and tab.bounds[1] < 900:
        ph.tap(tab)
    first = None
    for _ in range(12):                        # chờ lưới ổn định + có video mới ở ô đầu
        time.sleep(1.2)
        nodes = ph.dump()
        # ô video = khung click được, cỡ ô lưới, có chứa nhãn thời lượng (màn camera nằm dưới cũng có trong dump)
        durs = [n for n in nodes if re.fullmatch(r"\d{1,2}:\d\d(:\d\d)?", n.text)]
        cells = [n for n in nodes if n.clickable and 150 < n.bounds[2] - n.bounds[0] < 500 and n.bounds[1] > 250
                 and 0.6 < (n.bounds[3] - n.bounds[1]) / max(1, n.bounds[2] - n.bounds[0]) < 1.9
                 and any(_inside(d, n) for d in durs)]
        if cells:
            first = min(cells, key=lambda n: (n.bounds[1] // 50, n.bounds[0]))
            if any(n.text in labels and _inside(n, first) for n in nodes):
                break
    else:
        raise PostError(f"Ô đầu thư viện không phải video vừa chép (cần thời lượng {sorted(labels)[0]})")
    ticks = [n for n in nodes if n.clickable and n is not first and _inside(n, first) and n.bounds[2] - n.bounds[0] < 120]
    ph.tap(ticks[0] if ticks else first)
    time.sleep(2)
    nodes = ph.dump()
    nxt = ph.find(SEL["next"], nodes)
    count = re.search(r"\((\d+)\)", nxt.text if nxt else "")
    picked = [n for n in nodes if n.text in labels and n.bounds[1] > first.bounds[3] + 400]   # nhãn ở khay dưới
    if not nxt or not count or count.group(1) != "1" or not picked:
        raise PostError("Chọn video không khớp (khay chọn khác video vừa chép) — dừng để tránh đăng nhầm")


def _set_privacy(ph: Phone, privacy: str):
    ok = SEL["priv_ok"][privacy]
    if ph.find(ok):
        return
    row = ph.wait(SEL["privacy"], timeout=8, popups=False)
    ph.tap(row)
    time.sleep(1.8)
    ph.tap(ph.wait(SEL["priv_opt"][privacy], timeout=10, popups=False))
    time.sleep(1.5)
    if not ph.find(SEL["post"]):                   # bản nào mở trang riêng thì quay lại màn đăng
        ph.sh(BACK)
        time.sleep(1.2)
    if not ph.find(ok):
        raise PostError(f"Chọn quyền riêng tư '{privacy}' không thành công")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _cursor_end(ph: Phone, box: Node):
    ph.tap(box)
    time.sleep(0.8)
    ph.sh("input keycombination KEYCODE_CTRL_LEFT KEYCODE_MOVE_END", check=False)   # Ctrl+End: cuối cả ô
    time.sleep(0.4)


def _mention(ph: Phone, handle: str) -> bool:
    """Nhắc đến thật qua nút "@ Nhắc đến" → tìm → chọn đúng tài khoản. Không khớp chính xác → False (không nhắc nhầm)."""
    ph.tap(ph.wait(SEL["mention_btn"], timeout=8, cls="Button", popups=False))
    time.sleep(1.5)
    box = ph.wait(SEL["mention_search"], timeout=8, cls="EditText", popups=False)
    ph.tap(box)
    time.sleep(0.6)
    ph.type_text(handle)
    end = time.time() + 10
    while time.time() < end:
        time.sleep(1.2)
        hit = next((n for n in ph.dump() if n.text.strip().lower() == handle.lower() and n.cls == "TextView"
                    and n.bounds[1] > box.bounds[3]), None)
        if hit:
            ph.tap(hit)
            time.sleep(2)
            return True
    for _ in range(3):                             # không thấy → lùi về màn đăng
        if ph.find(r"Nháp|Drafts?"):
            break
        ph.sh(BACK)
        time.sleep(1.2)
    return False


def _type_caption(ph: Phone, box: Node, caption: str, mentions: list[str], hashtags: str) -> list[str]:
    """Gõ caption → nhắc đến từng người → hashtag, rồi đọc lại ô caption. Trả về các @ không tìm thấy (ghi dạng chữ)."""
    missed = []
    _cursor_end(ph, box)
    if caption:
        ph.type_text(caption + " ")                # dấu cách cuối: đóng gợi ý, không để TikTok tự thay #test → #testxxx
    for m in mentions:
        if not _mention(ph, m):
            missed.append(m)
            _cursor_end(ph, ph.wait(r".*", timeout=8, cls="EditText", popups=False))
            ph.type_text(f"@{m} ")
        else:
            box = ph.wait(r".*", timeout=8, cls="EditText", popups=False)
            _cursor_end(ph, box)
    if hashtags:
        ph.type_text(hashtags + " ")
    time.sleep(1.5)
    if "mInputShown=true" in ph.sh("dumpsys input_method", check=False):
        ph.sh(BACK)       # ẩn bàn phím (chỉ khi đang hiện — BACK thừa sẽ thoát màn đăng)
        time.sleep(1)
    want = " ".join([caption, *(f"@{m}" for m in mentions), hashtags])
    got = next((n.text for n in ph.dump() if n.cls == "EditText"), "")
    if _norm(got) != _norm(want):
        raise PostError(f"Caption trên máy khác bản gốc: {got[:80]!r}")
    return missed


if __name__ == "__main__":
    # post --dry: đi tới màn hình đăng rồi dừng · --tags="#a #b" · --men="@x @y"
    opt = {k: v for k, _, v in (a[2:].partition("=") for a in sys.argv[2:] if a.startswith("--"))}
    dry = "dry" in opt
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "devices"), [a for a in sys.argv[2:] if not a.startswith("--")]
    if cmd == "devices":
        for d in devices():
            print(d)
    elif cmd == "dump":
        for n in Phone(args[0] if args else None).dump():
            if n.text or n.desc or n.clickable or n.cls == "EditText":
                print(n)
    elif cmd == "post":
        print(post(Path(args[0]), args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else "private",
                   args[3] if len(args) > 3 else None, dry=dry, hashtags=opt.get("tags", ""), mentions=opt.get("men", "")))
    else:
        print(__doc__)
        os._exit(1)
