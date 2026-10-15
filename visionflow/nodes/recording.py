"""Record frames with chosen values at the line, replay them at a desk ("Kare Kaydedici" / "Kayıttan Oynat")."""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import pathlib
import shutil
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, StopFlow, register

MANIFEST = "manifest.json"
FORMAT = "visionflow-kayit"
VALUE_PORTS = ("a", "b", "c", "d")

_clock = time.monotonic     # patched in tests
_sleep = time.sleep


def jsonable(v):
    """A value that can go into the manifest: numbers, bools, text, lists; numpy scalars and small arrays converted;
    an image (large array) becomes a short summary instead of megabytes of numbers."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return v if not isinstance(v, float) or np.isfinite(v) else str(v)
    if isinstance(v, np.generic):
        return jsonable(v.item())
    if isinstance(v, np.ndarray):
        if v.size <= 64:
            return jsonable(v.tolist())
        return {"dizi": list(v.shape), "toplam": float(np.nansum(v.astype(np.float64)))}
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x) for x in v]
    return str(v)


def to_png_image(img: np.ndarray) -> np.ndarray:
    """An image PNG can store: uint8/uint16 as they are, bool → 0/255, float scaled (0…1 → 0…255) to uint8."""
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if img.dtype in (np.uint8, np.uint16):
        return img
    a = np.nan_to_num(img.astype(np.float64))
    if img.dtype.kind == "f" and a.size and a.min() >= 0 and a.max() <= 1:
        a = a * 255
    return np.clip(a, 0, 255).astype(np.uint8)


def write_json(path: pathlib.Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(folder) -> dict:
    """The manifest of a recording folder, checked. ValueError with a Turkish message when unusable."""
    if not folder:
        raise ValueError("Kayıt klasörü seçilmedi")
    path = pathlib.Path(folder) / MANIFEST
    if not pathlib.Path(folder).is_dir():
        raise ValueError(f"Kayıt klasörü yok: {folder}")
    if not path.is_file():
        raise ValueError(f"Bu klasörde kayıt yok ({MANIFEST} bulunamadı): {folder}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{MANIFEST} okunamadı: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("frames"), list):
        raise ValueError(f"{MANIFEST} bir VisionFlow kaydı değil")
    frames = [f for f in data["frames"] if isinstance(f, dict)]
    if not frames:
        raise ValueError(f"Kayıt boş (hiç kare yok): {folder}")
    names = data.get("names") if isinstance(data.get("names"), list) else list(VALUE_PORTS)
    data["names"] = [str(n) for n in names][:len(VALUE_PORTS)] + list(VALUE_PORTS)[len(names):]
    data["frames"] = frames
    return data


@register
class FrameRecorder(Node):
    title = "Kare Kaydedici"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Gelen kareleri PNG olarak, zamanları ve seçilen değerlerle (a–d) birlikte bir klasöre kaydeder "
                   "(manifest.json). 'Tetikte' modunda sadece tetik/hata doğruyken kaydeder ve öncesindeki N kareyi "
                   "de ekler. Hattaki bir sorunu 'Kayıttan Oynat' ile masada aynen tekrar oynatmak ve "
                   "'test-akis' ile akışı sınamak için.")
    inputs = [("image", PortType.IMAGE), ("trigger", PortType.BOOL)] + [(p, PortType.ANY) for p in VALUE_PORTS]
    outputs = [("recording", PortType.BOOL), ("count", PortType.NUMBER), ("folder", PortType.TEXT)]
    params = [
        Param("folder", "str", "kayitlar/hat_kaydi", "Klasör"),
        Param("new_folder", "bool", True, "Her çalıştırmada yeni alt klasör"),
        Param("mode", "choice", "her kare", "Ne zaman", choices=["her kare", "tetikte"]),
        Param("pre", "int", 10, "Tetikten önceki kare sayısı (bellekte tutulur)", 0, 300),
        Param("post", "int", 10, "Tetikten sonraki kare sayısı", 0, 100000),
        Param("names", "str", "a,b,c,d", "Değer adları (virgülle)"),
        Param("min_free_mb", "int", 500, "En az boş disk (MB)", 0, 10_000_000),
        Param("max_files", "int", 10000, "En fazla kare dosyası", 1, 10_000_000),
    ]
    setup_params = ("folder", "new_folder", "pre")
    SAVE_EVERY = 2.0            # seconds between manifest writes (also written at the end)

    def setup(self):
        root = pathlib.Path(self.v("folder").strip() or "kayitlar/hat_kaydi")
        if self.v("new_folder"):
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder, n = root / f"kayit_{stamp}", 1
            while folder.exists():
                n += 1
                folder = root / f"kayit_{stamp}_{n}"
        else:
            folder = root
        frames = []
        if (folder / MANIFEST).is_file():                  # same folder again: continue the recording
            try:
                old = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
                frames = [f for f in old.get("frames", []) if isinstance(f, dict)]
            except (OSError, ValueError, AttributeError):
                frames = []
        self.state.update(folder=folder, frames=frames, ring=collections.deque(maxlen=max(0, self.v("pre"))),
                          t0=None, start_offset=(frames[-1].get("t", 0.0) + 0.001) if frames else 0.0,
                          seq=len(frames), post_left=0, stopped="", saved_at=0.0, dirty=False, files=sum(
                              1 for f in frames if f.get("file")))

    def _names(self):
        names = [n.strip() for n in str(self.v("names")).split(",")]
        return [(names[i] if i < len(names) and names[i] else p) for i, p in enumerate(VALUE_PORTS)]

    def _check_space(self, folder: pathlib.Path) -> str:
        if self.state["files"] >= self.v("max_files"):
            return f"En fazla kare dosyası sayısına ulaşıldı ({self.v('max_files')}), kayıt durduruldu"
        try:
            free = shutil.disk_usage(folder).free / 1e6
        except OSError:
            return ""
        if free < self.v("min_free_mb"):
            return f"Diskte yer azaldı ({free:.0f} MB < {self.v('min_free_mb')} MB), kayıt durduruldu"
        return ""

    def _write(self, item):
        st = self.state
        if st["stopped"]:
            return
        folder = st["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        t, stamp, img, values = item
        reason = self._check_space(folder)
        if reason:
            st["stopped"] = reason
            print(f"[Kare Kaydedici] {reason}")
            return
        st["seq"] += 1
        entry = {"n": st["seq"], "t": round(st["start_offset"] + t, 6), "time": stamp, "file": None,
                 "values": values}
        if img is not None:
            name = f"{st['seq']:06d}.png"
            ok, buf = cv2.imencode(".png", to_png_image(img), [cv2.IMWRITE_PNG_COMPRESSION, 1])  # fast: runs in the loop
            if not ok:
                raise RuntimeError("Kare PNG olarak kodlanamadı")
            buf.tofile(str(folder / name))        # unicode-safe on Windows
            entry["file"] = name
            st["files"] += 1
        st["frames"].append(entry)
        st["dirty"] = True

    def _save_manifest(self):
        st = self.state
        if not st.get("dirty"):
            return
        st["folder"].mkdir(parents=True, exist_ok=True)
        write_json(st["folder"] / MANIFEST, {"format": FORMAT, "version": 1, "names": self._names(),
                                             "frames": st["frames"]})
        st["dirty"], st["saved_at"] = False, _clock()

    def process(self, inputs):
        st = self.state
        img = inputs["image"]
        if img is not None and not isinstance(img, np.ndarray):
            raise ValueError("Görüntü girişi resim değil")
        trigger_mode = self.v("mode") == "tetikte"
        if trigger_mode and inputs["trigger"] is None:
            raise ValueError("'Tetikte' modunda tetik girişi bağlanmalı (ör. hatalı ürün sinyali)")
        if st["stopped"]:
            raise RuntimeError(st["stopped"])
        now = _clock()
        if st["t0"] is None:
            st["t0"] = now
        names = self._names()
        item = (now - st["t0"], dt.datetime.now().isoformat(timespec="milliseconds"),
                None if img is None else img.copy(), {names[i]: jsonable(inputs[p]) for i, p in enumerate(VALUE_PORTS)})
        if not trigger_mode:
            record = True
        elif bool(inputs["trigger"]):
            while st["ring"]:                      # the frames just before the problem
                self._write(st["ring"].popleft())
            record, st["post_left"] = True, self.v("post")
        elif st["post_left"] > 0:
            record = True
            st["post_left"] -= 1
        else:
            record = False
            if st["ring"].maxlen:
                st["ring"].append(item)
        if record:
            self._write(item)
        if st["dirty"] and (now - st["saved_at"] >= self.SAVE_EVERY or st["stopped"]):
            self._save_manifest()
        if st["stopped"]:
            raise RuntimeError(st["stopped"])
        return {"recording": record, "count": len(st["frames"]), "folder": str(st["folder"])}

    def teardown(self):
        try:
            self._save_manifest()
        except OSError as exc:
            print(f"[Kare Kaydedici] manifest yazılamadı: {exc}")


@register
class RecordingPlayer(Node):
    title = "Kayıttan Oynat"
    category = "1 · Giriş"
    description = ("'Kare Kaydedici' ile alınmış bir kaydı kaynak gibi oynatır: görüntüyü ve kaydedilen a–d "
                   "değerlerini verir. Orijinal zamanlamayla ya da olabildiğince hızlı; isterseniz başa sarar.")
    outputs = [("image", PortType.IMAGE)] + [(p, PortType.ANY) for p in VALUE_PORTS] + \
        [("frame", PortType.NUMBER), ("time", PortType.NUMBER)]
    params = [
        Param("folder", "folder", "", "Kayıt klasörü"),
        Param("speed", "choice", "orijinal", "Hız", choices=["orijinal", "en hızlı"]),
        Param("factor", "float", 1.0, "Hız çarpanı (orijinal hızda)", 0.01, 100),
        Param("loop", "bool", False, "Başa sar"),
    ]
    is_source = True
    setup_params = ("folder",)

    def setup(self):
        data = load_manifest(self.v("folder"))
        self.state.update(data=data, i=0, start=None, base_t=0.0)

    def process(self, inputs):
        st = self.state
        frames = st["data"]["frames"]
        if st["i"] >= len(frames):
            if not self.v("loop"):
                raise StopFlow()
            st["i"], st["start"] = 0, None
        entry = frames[st["i"]]
        st["i"] += 1
        try:
            t = float(entry.get("t", 0.0))
        except (TypeError, ValueError):
            t = 0.0
        if self.v("speed") == "orijinal":
            now = _clock()
            if st["start"] is None:
                st["start"], st["base_t"] = now, t
            wait = st["start"] + (t - st["base_t"]) / self.v("factor") - now
            if wait > 0:
                _sleep(min(wait, 60.0))
        img = None
        if entry.get("file"):
            path = pathlib.Path(self.v("folder")) / str(entry["file"])
            if not path.is_file():
                raise RuntimeError(f"Kayıttaki kare dosyası yok: {path.name}")
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Kare dosyası okunamadı: {path.name}")
        values = entry.get("values") if isinstance(entry.get("values"), dict) else {}
        names = st["data"]["names"]
        out = {p: values.get(names[i]) for i, p in enumerate(VALUE_PORTS)}
        out.update(image=img, frame=st["i"], time=t)
        return out
