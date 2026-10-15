"""Part traceability archive ("İzlenebilirlik arşivi"): every inspected part with its time, flow, station, serial /
barcode, lot, verdict, reason, measured values, recipe and its pictures, findable later.

Layout of an archive folder (default ``home()/izlenebilirlik``)::

    arsiv.sqlite                 one row per part (SQLite in WAL mode: the browser reads while the line writes)
    resimler/2026-10-14/…jpg     original and annotated pictures, one folder per day

``ArchiveWriter`` takes records from the flow thread and writes them on its own background thread through a bounded
queue: a slow or full disk never slows the line. When the queue is full the record is dropped, counted and a warning
is printed. Old parts are removed automatically (keep N days and / or at most N GB).

``Archive`` is the read side (search, summary, CSV / Excel / ZIP export) used by the browser window, the
"Arşivde Ara" block and ``python -m visionflow arsiv``.

The per-minute good / bad counts of the operator panel stay in production.py (``uretim_verisi.sqlite``): those are
small and kept forever for reports, while this archive is large, per part and pruned."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import pathlib
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
import zipfile
from dataclasses import dataclass

import numpy as np

ARCHIVE_DIR = "izlenebilirlik"
DB_NAME = "arsiv.sqlite"
IMG_DIR = "resimler"
SCHEMA_VERSION = 1
IMAGE_POLICIES = ("Hepsi", "Yalnız hatalı parçalar", "Hiçbiri")
BATCH = 200                     # records committed in one transaction
RETENTION_EVERY_S = 30.0        # how often the writer prunes old parts
WARN_EVERY_S = 10.0             # a full queue is reported at most this often
MAX_TEXT = 500
EXCEL_MAX_ROWS = 1_048_575

COLUMNS = ("id", "rid", "ts", "flow", "station", "serial", "lot", "ok", "reason", "vals", "recipe", "img", "ann",
           "bytes")
HEADERS = ["Kayıt no", "Zaman", "Akış", "İstasyon", "Seri no", "Lot", "Sonuç", "Sebep", "Reçete",
           "Orijinal görüntü", "İşaretli görüntü"]


def default_root() -> pathlib.Path:
    from .ai.config import home
    return home() / ARCHIVE_DIR


def fold(text) -> str:
    """Case-insensitive search key; Turkish İ / ı match i (barcodes typed in either case)."""
    return str(text or "").replace("İ", "i").replace("ı", "i").casefold()


def clean_text(value, limit: int = MAX_TEXT) -> str:
    """Text for a record field: numbers without a needless '.0', control characters removed."""
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        value = int(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value)).strip()
    return text[:limit]


def safe_name(text: str, limit: int = 40) -> str:
    """Part of a file name: letters (Turkish too) and digits kept, the rest '_'."""
    s = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")[:limit]
    return s


def values_dict(value) -> dict:
    """The 'values' input as a JSON-able dict: a dict as it is, a single number / text / bool as {"değer": …}."""
    from .nodes.recording import jsonable
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {"değerler": jsonable(list(value))}
    if isinstance(value, np.ndarray) and value.size > 64:
        raise ValueError("'values' girişine görüntü / büyük dizi değil, ölçüm değeri (sayı veya sözlük) bağlayın")
    return {"değer": jsonable(value)}


def to_jpeg_bgr(img) -> np.ndarray | None:
    """An 8-bit gray or BGR picture JPEG can store, from any image the flow carries (16-bit, float, bool,
    BGRA, 2 or >4 channels). None for an empty image."""
    if img is None:
        return None
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Görüntü bekleniyor, gelen: {type(img).__name__}")
    if img.size == 0:
        return None
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim not in (2, 3):
        raise ValueError(f"Görüntü 2 veya 3 boyutlu olmalı (gelen {img.ndim} boyut)")
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        a = np.nan_to_num(img.astype(np.float64))
        if img.dtype.kind == "f" and a.size and a.min() >= 0 and a.max() <= 1:
            a = a * 255
        elif img.dtype.kind == "f" and a.size and (a.min() < 0 or a.max() > 255):
            # a depth map (mm) or height map: stretched over its measured values, else it would be all white
            from .api_server import stretch_u8
            a = stretch_u8(img)
        img = np.clip(a, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 4:
            img = img[:, :, :3]
        elif c != 3:
            img = img[:, :, 0]
    return np.ascontiguousarray(img)


def read_image(path) -> np.ndarray | None:
    """A picture of the archive (Unicode-safe path on Windows too), None if it is missing / unreadable."""
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


# --- filters ----------------------------------------------------------------------------------------------------------

@dataclass
class Filter:
    start: float | None = None        # epoch seconds, inclusive
    end: float | None = None          # epoch seconds, exclusive
    serial: str = ""                  # contained in the serial or the lot (case-insensitive)
    verdict: bool | None = None       # True = sağlam, False = hatalı, None = all
    flow: str = ""
    station: str = ""
    recipe: str = ""
    reason: str = ""                  # contained in the reason text

    def where(self) -> tuple[str, list]:
        parts, args = [], []
        if self.start is not None:
            parts.append("ts >= ?")
            args.append(float(self.start))
        if self.end is not None:
            parts.append("ts < ?")
            args.append(float(self.end))
        if self.serial.strip():
            parts.append("(skey LIKE ? ESCAPE '\\' OR lkey LIKE ? ESCAPE '\\')")
            pat = "%" + _like(fold(self.serial.strip())) + "%"
            args += [pat, pat]
        if self.verdict is not None:
            parts.append("ok = ?")
            args.append(1 if self.verdict else 0)
        for col in ("flow", "station", "recipe"):
            v = getattr(self, col)
            if v:
                parts.append(f"{col} = ?")
                args.append(v)
        if self.reason.strip():
            parts.append("instr(vf_fold(reason), ?) > 0")
            args.append(fold(self.reason.strip()))
        return (" WHERE " + " AND ".join(parts)) if parts else "", args


def _like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def parse_date(text: str, end: bool = False) -> float:
    """'2026-10-01', '01.10.2026', '2026-10-01 08:30' → epoch seconds (local time). A date alone as the end of a
    range means the end of that day."""
    s = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
                "%d.%m.%Y"):
        try:
            d = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        if end and len(fmt) <= 8:
            d += dt.timedelta(days=1)
        return d.timestamp()
    raise ValueError(f"Tarih anlaşılamadı: '{text}' (ör. 2026-10-01 veya 01.10.2026 14:30)")


# --- read side --------------------------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rid TEXT UNIQUE NOT NULL,
    ts REAL NOT NULL,
    flow TEXT DEFAULT '', station TEXT DEFAULT '',
    serial TEXT DEFAULT '', skey TEXT DEFAULT '', lot TEXT DEFAULT '', lkey TEXT DEFAULT '',
    ok INTEGER, reason TEXT DEFAULT '', vals TEXT DEFAULT '{}', recipe TEXT DEFAULT '',
    img TEXT DEFAULT '', ann TEXT DEFAULT '', bytes INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS parts_ts ON parts (ts);
CREATE INDEX IF NOT EXISTS parts_serial ON parts (serial, ts);
CREATE INDEX IF NOT EXISTS parts_skey ON parts (skey);
"""


class Archive:
    """Read access to an archive folder. Every call opens its own short connection, so it is safe from any thread
    and while a writer (this or another process) is adding parts."""

    def __init__(self, root=None):
        self.root = pathlib.Path(root) if root else default_root()
        self.db = self.root / DB_NAME

    def exists(self) -> bool:
        return self.db.exists()

    def _connect(self, create: bool = False) -> sqlite3.Connection | None:
        if not create and not self.db.exists():
            return None
        if create:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"Arşiv klasörü oluşturulamadı ({self.root}): {exc}") from None
        try:
            con = sqlite3.connect(str(self.db), timeout=10, check_same_thread=False)
            con.create_function("vf_fold", 1, fold, deterministic=True)
            if create:
                con.execute("PRAGMA journal_mode=WAL")
                con.executescript(_SCHEMA)
                con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                con.commit()
        except sqlite3.Error as exc:
            raise ValueError(f"Arşiv veritabanı açılamadı ({self.db}): {exc}") from None
        return con

    def _rows(self, sql: str, args=()) -> list[tuple]:
        con = self._connect()
        if con is None:
            return []
        try:
            return con.execute(sql, args).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise ValueError(f"Arşiv okunamadı: {exc}") from None
        finally:
            con.close()

    def _record(self, row) -> dict:
        r = dict(zip(COLUMNS, row))
        try:
            r["values"] = json.loads(r.pop("vals") or "{}")
        except ValueError:
            r["values"] = {}
        r["ok"] = None if r["ok"] is None else bool(r["ok"])
        r["time"] = dt.datetime.fromtimestamp(r["ts"])
        r["img_path"] = str(self.root / r["img"]) if r["img"] else ""
        r["ann_path"] = str(self.root / r["ann"]) if r["ann"] else ""
        return r

    def search(self, flt: Filter | None = None, limit: int = 100, offset: int = 0, newest_first: bool = True) \
            -> list[dict]:
        where, args = (flt or Filter()).where()
        order = "DESC" if newest_first else "ASC"
        rows = self._rows(f"SELECT {', '.join(COLUMNS)} FROM parts{where} ORDER BY ts {order}, id {order} "
                          f"LIMIT ? OFFSET ?", [*args, int(limit), int(offset)])
        return [self._record(r) for r in rows]

    def iter_records(self, flt: Filter | None = None, chunk: int = 2000):
        """Every matching record, oldest first, read in chunks (for exports of any size)."""
        where, args = (flt or Filter()).where()
        last = -1
        while True:
            w = (where + " AND id > ?") if where else " WHERE id > ?"
            rows = self._rows(f"SELECT {', '.join(COLUMNS)} FROM parts{w} ORDER BY id LIMIT ?",
                              [*args, last, chunk])
            if not rows:
                return
            for r in rows:
                yield self._record(r)
            last = rows[-1][0]

    def count(self, flt: Filter | None = None) -> int:
        where, args = (flt or Filter()).where()
        rows = self._rows(f"SELECT COUNT(*) FROM parts{where}", args)
        return int(rows[0][0]) if rows else 0

    def get(self, rid: str) -> dict | None:
        rows = self._rows(f"SELECT {', '.join(COLUMNS)} FROM parts WHERE rid = ?", [str(rid)])
        return self._record(rows[0]) if rows else None

    def by_ids(self, ids) -> list[dict]:
        ids = [int(i) for i in ids]
        out = []
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            rows = self._rows(f"SELECT {', '.join(COLUMNS)} FROM parts WHERE id IN ({','.join('?' * len(part))}) "
                              "ORDER BY id", part)
            out += [self._record(r) for r in rows]
        return out

    def last_by_serial(self, serial: str, station: str = "") -> tuple[dict | None, int]:
        """The newest record of exactly this serial number (optionally of one station) and how many there are."""
        serial = clean_text(serial)
        if not serial:
            return None, 0
        extra, args = ("", [serial]) if not station else (" AND station = ?", [serial, station])
        rows = self._rows(f"SELECT {', '.join(COLUMNS)} FROM parts WHERE serial = ?{extra} "
                          "ORDER BY ts DESC, id DESC LIMIT 1", args)
        n = self._rows(f"SELECT COUNT(*) FROM parts WHERE serial = ?{extra}", args)
        return (self._record(rows[0]) if rows else None), (int(n[0][0]) if n else 0)

    def distinct(self, column: str) -> list[str]:
        if column not in ("flow", "station", "recipe"):
            raise ValueError(column)
        return [r[0] for r in self._rows(f"SELECT DISTINCT {column} FROM parts WHERE {column} != '' "
                                         f"ORDER BY {column}")]

    def summary(self, flt: Filter | None = None, top: int = 5) -> dict:
        where, args = (flt or Filter()).where()
        rows = self._rows(f"SELECT COUNT(*), SUM(ok = 1), SUM(ok = 0) FROM parts{where}", args)
        total, good, bad = (int(x or 0) for x in (rows[0] if rows else (0, 0, 0)))
        w = (where + " AND ok = 0") if where else " WHERE ok = 0"
        reasons = self._rows(f"SELECT reason, COUNT(*) AS n FROM parts{w} GROUP BY reason ORDER BY n DESC, reason "
                             "LIMIT ?", [*args, int(top)])
        judged = good + bad
        return {"count": total, "good": good, "bad": bad,
                "fail_pct": round(100.0 * bad / judged, 2) if judged else None,
                "top_reasons": [(r or "(sebep yok)", int(n)) for r, n in reasons]}

    # --- export ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _value_keys(records: list[dict]) -> list[str]:
        keys: dict[str, None] = {}
        for r in records:
            for k in r["values"]:
                keys.setdefault(k, None)
        return list(keys)

    @staticmethod
    def _row(r: dict, keys: list[str]) -> list:
        verdict = "" if r["ok"] is None else ("Sağlam" if r["ok"] else "Hatalı")
        vals = []
        for k in keys:
            v = r["values"].get(k, "")
            vals.append(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        return [r["rid"], r["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], r["flow"], r["station"], r["serial"],
                r["lot"], verdict, r["reason"], r["recipe"], r["img"], r["ann"], *vals]

    def _materialize(self, records, flt) -> list[dict]:
        return list(records) if records is not None else list(self.iter_records(flt))

    def export_csv(self, path, records=None, flt: Filter | None = None) -> int:
        """CSV for Excel (';' separated, UTF-8 with BOM so Turkish letters open right). Returns the row count."""
        recs = self._materialize(records, flt)
        keys = self._value_keys(recs)
        path = pathlib.Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                self._write_csv(f, recs, keys)
        except OSError as exc:
            raise ValueError(f"CSV yazılamadı ({path}): {exc}") from None
        return len(recs)

    def _write_csv(self, f, recs, keys):
        w = csv.writer(f, delimiter=";")
        w.writerow(HEADERS + keys)
        for r in recs:
            w.writerow(self._row(r, keys))

    def export_xlsx(self, path, records=None, flt: Filter | None = None) -> int:
        try:
            import openpyxl
        except ImportError:
            raise ValueError("Excel için 'openpyxl' paketi gerekli: pip install openpyxl "
                             "(ya da CSV olarak kaydedin)") from None
        recs = self._materialize(records, flt)
        if len(recs) > EXCEL_MAX_ROWS:
            raise ValueError(f"Excel en fazla {EXCEL_MAX_ROWS} satır alır ({len(recs)} kayıt var); filtreyi daraltın "
                             "veya CSV kullanın")
        keys = self._value_keys(recs)
        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet("Arşiv")
        ws.append(HEADERS + keys)
        bad = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        for r in recs:
            ws.append([bad.sub(" ", v) if isinstance(v, str) else v for v in self._row(r, keys)])
        try:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            wb.save(str(path))
        except OSError as exc:
            raise ValueError(f"Excel dosyası yazılamadı ({path}): {exc}") from None
        return len(recs)

    def export_zip(self, path, records=None, flt: Filter | None = None) -> tuple[int, int]:
        """ZIP with kayitlar.csv and the pictures (resimler/<kayıt>_<seri>_orijinal.jpg / _isaretli.jpg).
        Returns (records, pictures). Missing picture files are skipped."""
        recs = self._materialize(records, flt)
        keys = self._value_keys(recs)
        n_img = 0
        path = pathlib.Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                buf = io.StringIO()
                self._write_csv(buf, recs, keys)
                z.writestr("kayitlar.csv", "﻿" + buf.getvalue())
                for r in recs:
                    stem = r["rid"] + (f"_{safe_name(r['serial'])}" if safe_name(r["serial"]) else "")
                    for col, suffix in (("img", "orijinal"), ("ann", "isaretli")):
                        if r[col] and (self.root / r[col]).is_file():
                            z.write(self.root / r[col], f"resimler/{stem}_{suffix}.jpg",
                                    compress_type=zipfile.ZIP_STORED)
                            n_img += 1
        except OSError as exc:
            raise ValueError(f"ZIP yazılamadı ({path}): {exc}") from None
        return len(recs), n_img

    def export(self, path, records=None, flt: Filter | None = None) -> str:
        """Export by the file extension (.csv, .xlsx, .zip). Returns a Turkish one-line result."""
        ext = pathlib.Path(path).suffix.lower()
        if ext == ".csv":
            return f"{self.export_csv(path, records, flt)} kayıt CSV dosyasına yazıldı: {path}"
        if ext in (".xlsx", ".xlsm"):
            return f"{self.export_xlsx(path, records, flt)} kayıt Excel dosyasına yazıldı: {path}"
        if ext == ".zip":
            n, k = self.export_zip(path, records, flt)
            return f"{n} kayıt ve {k} görüntü ZIP dosyasına yazıldı: {path}"
        raise ValueError(f"Desteklenmeyen dosya türü '{ext or '(uzantı yok)'}': .csv, .xlsx veya .zip seçin")


# --- write side -------------------------------------------------------------------------------------------------------

_STOP = object()


class ArchiveWriter(Archive):
    """Writes parts on a background thread. ``submit`` never blocks: it returns the new record id, or "" when the
    queue was full and the record was dropped (counted in ``dropped``)."""

    def __init__(self, root=None, queue_size: int = 256, queue_mb: float = 512, jpeg_quality: int = 90,
                 keep_days: float = 0, max_gb: float = 0, images: str = IMAGE_POLICIES[0], clock=time.time,
                 log=print):
        super().__init__(root)
        self.queue_size = max(1, int(queue_size))
        self.queue_bytes = max(1, int(queue_mb * 1024 * 1024))
        self.jpeg_quality = int(min(100, max(1, jpeg_quality)))
        self.keep_days = float(keep_days or 0)
        self.max_gb = float(max_gb or 0)
        if images not in IMAGE_POLICIES:
            raise ValueError(f"Görüntü saklama seçenekleri: {', '.join(IMAGE_POLICIES)}")
        self.images = images
        self.clock = clock
        self.log = log
        self.dropped = 0
        self.written = 0
        self.errors = 0
        self.last_error = ""
        self._q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.Condition()
        self._pending = 0
        self._queued_bytes = 0
        self._last_warn = -1e18
        self._total_bytes = 0
        self._last_retention = -1e18
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        con = self._connect(create=True)          # a wrong / read-only folder fails here, with a clear message
        con.close()
        self._probe()

    def _probe(self):
        try:
            (self.root / IMG_DIR).mkdir(parents=True, exist_ok=True)
            probe = self.root / IMG_DIR / f".yazma_testi_{os.getpid()}"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise ValueError(f"Arşiv klasörüne yazılamıyor ({self.root}): {exc}") from None

    # --- flow thread --------------------------------------------------------------------------------------------
    def start(self):
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="izlenebilirlik-arsivi", daemon=True)
                self._thread.start()

    def submit(self, serial="", ok=None, reason="", values=None, image=None, annotated=None, flow="", station="",
               lot="", recipe="", ts: float | None = None, images: str | None = None,
               jpeg_quality: int | None = None) -> str:
        """Queue one part. ``images`` / ``jpeg_quality`` override the writer's settings for this record (blocks
        sharing a writer keep their own choices)."""
        ts = self.clock() if ts is None else float(ts)
        ok = None if ok is None else bool(ok)
        policy = images or self.images
        if policy not in IMAGE_POLICIES:
            raise ValueError(f"Görüntü saklama seçenekleri: {', '.join(IMAGE_POLICIES)}")
        keep_img = policy == IMAGE_POLICIES[0] or (policy == IMAGE_POLICIES[1] and ok is not True)
        imgs = []
        for kind, im in (("o", image), ("a", annotated)):
            if keep_img and im is not None:
                im = to_jpeg_bgr(im)                 # checks the type on the flow thread: a clear error there
                if im is not None:
                    imgs.append((kind, im))
        vals = values if isinstance(values, dict) and all(isinstance(k, str) for k in values) else \
            values_dict(values)
        try:
            vals_json = json.dumps(vals, ensure_ascii=False)
        except (TypeError, ValueError):
            from .nodes.recording import jsonable
            vals_json = json.dumps(jsonable(vals), ensure_ascii=False)
        stamp = dt.datetime.fromtimestamp(ts)
        rid = f"{stamp:%Y%m%d-%H%M%S}-{stamp.microsecond // 1000:03d}-{secrets.token_hex(3)}"
        serial, lot = clean_text(serial), clean_text(lot)
        rec = {"rid": rid, "ts": ts, "flow": clean_text(flow, 200), "station": clean_text(station, 200),
               "serial": serial, "skey": fold(serial), "lot": lot, "lkey": fold(lot),
               "ok": None if ok is None else int(ok), "reason": clean_text(reason, 2000), "vals": vals_json,
               "recipe": clean_text(recipe, 200),
               "q": int(min(100, max(1, jpeg_quality if jpeg_quality is not None else self.jpeg_quality)))}
        size = sum(im.nbytes for _, im in imgs)
        with self._lock:
            full = self._queued_bytes + size > self.queue_bytes and self._pending > 0
            if not full:
                try:
                    # the flow keeps its own arrays; the copy is what the writer encodes later
                    self._q.put_nowait((rec, [(k, im.copy()) for k, im in imgs], size))
                except queue.Full:
                    full = True
            if full:
                self.dropped += 1
                now = time.monotonic()
                if now - self._last_warn >= WARN_EVERY_S:
                    self._last_warn = now
                    self.log(f"[İzlenebilirlik] UYARI: kayıt kuyruğu dolu, parça arşivlenmedi (toplam {self.dropped} "
                             "atlandı). Disk yavaş veya dolu olabilir.")
                return ""
            self._pending += 1
            self._queued_bytes += size
        self.start()
        return rid

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until every submitted record is on disk. False on timeout."""
        end = time.monotonic() + timeout
        with self._lock:
            while self._pending:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                if self._thread is None or not self._thread.is_alive():
                    self.start()
                self._lock.wait(min(left, 0.2))
        return True

    def close(self, timeout: float = 30.0) -> bool:
        ok = self.flush(timeout)
        t = self._thread
        if t is not None and t.is_alive():
            try:
                self._q.put(_STOP, timeout=1.0)
            except queue.Full:
                pass
            t.join(timeout=5.0)
        self._thread = None
        return ok

    @property
    def pending(self) -> int:
        return self._pending

    # --- writer thread ------------------------------------------------------------------------------------------
    def _run(self):
        try:
            con = self._connect(create=True)
            con.execute("PRAGMA synchronous=NORMAL")
            self._total_bytes = int(con.execute("SELECT COALESCE(SUM(bytes), 0) FROM parts").fetchone()[0])
        except (ValueError, sqlite3.Error) as exc:
            self._fail(str(exc))
            self._discard_all()
            return
        try:
            self._maybe_retention(con)
            stop = False
            while not stop:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    self._maybe_retention(con)
                    continue
                batch = []
                while True:
                    if item is _STOP:
                        stop = True
                    else:
                        batch.append(item)
                    if stop or len(batch) >= BATCH:
                        break
                    try:
                        item = self._q.get_nowait()
                    except queue.Empty:
                        break
                if batch:
                    self._write(con, batch)
                self._maybe_retention(con)
        finally:
            con.close()

    def _discard_all(self):
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP:
                self._done(item[2])

    def _done(self, size: int, n: int = 1):
        with self._lock:
            self._pending -= n
            self._queued_bytes -= size
            self._lock.notify_all()

    def _fail(self, msg: str):
        self.errors += 1
        if msg != self.last_error:
            self.log(f"[İzlenebilirlik] HATA: {msg}")
        self.last_error = msg

    @staticmethod
    def _encode(im: np.ndarray, quality: int) -> bytes | None:
        import cv2
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    def _write(self, con, batch):
        try:
            self._write_batch(con, batch)
        except Exception as exc:  # noqa: BLE001 - the writer thread must survive anything
            self._fail(f"Arşive yazılamadı: {exc}")
        finally:
            self._done(sum(b[2] for b in batch), len(batch))

    def _write_batch(self, con, batch):
        rows = []
        for rec, imgs, _size in batch:
            nbytes = 0
            paths = {"o": "", "a": ""}
            if imgs:
                day = dt.datetime.fromtimestamp(rec["ts"]).strftime("%Y-%m-%d")
                folder = self.root / IMG_DIR / day
                stem = rec["rid"] + (f"_{safe_name(rec['serial'])}" if safe_name(rec["serial"]) else "")
                for kind, im in imgs:
                    try:
                        data = self._encode(im, rec["q"])
                        if data is None:
                            raise OSError("JPEG kodlanamadı")
                        folder.mkdir(parents=True, exist_ok=True)
                        rel = f"{IMG_DIR}/{day}/{stem}_{kind}.jpg"
                        with open(self.root / rel, "wb") as f:
                            f.write(data)
                        paths[kind] = rel
                        nbytes += len(data)
                    except Exception as exc:  # noqa: BLE001 - a picture failing must not lose the record
                        self._fail(f"Görüntü yazılamadı: {exc}")
            rows.append((rec["rid"], rec["ts"], rec["flow"], rec["station"], rec["serial"], rec["skey"], rec["lot"],
                         rec["lkey"], rec["ok"], rec["reason"], rec["vals"], rec["recipe"], paths["o"], paths["a"],
                         nbytes))
        for attempt in range(5):
            try:
                with con:
                    con.executemany("INSERT INTO parts (rid, ts, flow, station, serial, skey, lot, lkey, ok, reason, "
                                    "vals, recipe, img, ann, bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                self.written += len(rows)
                self._total_bytes += sum(r[-1] for r in rows)
                break
            except sqlite3.OperationalError as exc:      # locked by another process: try again shortly
                if attempt == 4:
                    self._fail(f"Arşive yazılamadı: {exc}")
                else:
                    time.sleep(0.2 * (attempt + 1))
            except sqlite3.Error as exc:
                self._fail(f"Arşive yazılamadı: {exc}")
                break

    # --- retention ----------------------------------------------------------------------------------------------
    def _maybe_retention(self, con):
        if not (self.keep_days > 0 or self.max_gb > 0):
            return
        now = time.monotonic()
        over = self.max_gb > 0 and self._total_bytes > self.max_gb * 1024 ** 3
        if over or now - self._last_retention >= RETENTION_EVERY_S:
            self._last_retention = now
            try:
                self._retention(con)
            except (OSError, sqlite3.Error) as exc:
                self._fail(f"Eski kayıtlar silinemedi: {exc}")

    def apply_retention(self) -> int:
        """Prune now (also called by the writer thread every 30 s). Returns the number of parts removed."""
        con = self._connect(create=True)
        try:
            self._total_bytes = int(con.execute("SELECT COALESCE(SUM(bytes), 0) FROM parts").fetchone()[0])
            return self._retention(con)
        finally:
            con.close()

    def _db_bytes(self, con) -> int:
        page = con.execute("PRAGMA page_size").fetchone()[0]
        used = con.execute("PRAGMA page_count").fetchone()[0] - con.execute("PRAGMA freelist_count").fetchone()[0]
        return int(page * used)

    def _delete(self, con, rows) -> int:
        freed = 0
        for _id, img, ann, nbytes in rows:
            for rel in (img, ann):
                if rel:
                    try:
                        (self.root / rel).unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        self._fail(f"Eski görüntü silinemedi: {exc}")
            freed += nbytes or 0
        with con:
            con.executemany("DELETE FROM parts WHERE id = ?", [(r[0],) for r in rows])
        self._total_bytes -= freed
        return len(rows)

    def _retention(self, con) -> int:
        removed = 0
        if self.keep_days > 0:
            cutoff = self.clock() - self.keep_days * 86400
            while True:
                rows = con.execute("SELECT id, img, ann, bytes FROM parts WHERE ts < ? ORDER BY ts LIMIT 1000",
                                   (cutoff,)).fetchall()
                if not rows:
                    break
                removed += self._delete(con, rows)
            # whole day folders before the cut-off (pictures whose records never made it into the database too)
            first_day = dt.datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d")
            base = self.root / IMG_DIR
            if base.is_dir():
                for d in base.iterdir():
                    if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and d.name < first_day:
                        shutil.rmtree(d, ignore_errors=True)
        if self.max_gb > 0:
            limit = self.max_gb * 1024 ** 3
            while self._total_bytes + self._db_bytes(con) > limit:
                rows = con.execute("SELECT id, img, ann, bytes FROM parts ORDER BY ts, id LIMIT 200").fetchall()
                if not rows:
                    break
                over = self._total_bytes + self._db_bytes(con) - limit
                take, freed = [], 0
                for r in rows:
                    take.append(r)
                    freed += (r[3] or 0) + 200          # ~200 bytes of database per row
                    if freed >= over:
                        break
                removed += self._delete(con, take)
        if removed:
            self._remove_empty_days()
        return removed

    def _remove_empty_days(self):
        base = self.root / IMG_DIR
        if not base.is_dir():
            return
        for d in base.iterdir():
            if d.is_dir():
                try:
                    d.rmdir()                # only succeeds when empty
                except OSError:
                    pass


# --- shared writers: blocks writing to the same folder share one thread ---------------------------------------------

_writers: dict[str, list] = {}          # resolved folder → [writer, users]
_writers_lock = threading.Lock()


def open_writer(root=None, **opts) -> ArchiveWriter:
    """The writer of this folder, shared by every block that archives into it (settings of the last one opened win
    for retention). Give it back with ``release_writer``."""
    key = str(pathlib.Path(root or default_root()).resolve())
    with _writers_lock:
        entry = _writers.get(key)
        if entry is None:
            entry = _writers[key] = [ArchiveWriter(root or default_root(), **opts), 0]
        else:
            w = entry[0]
            for name in ("jpeg_quality", "keep_days", "max_gb"):
                if name in opts:
                    setattr(w, name, type(getattr(w, name))(opts[name]))
        entry[1] += 1
        return entry[0]


def release_writer(writer: ArchiveWriter, timeout: float = 30.0) -> bool:
    """Give back a writer from ``open_writer``; the last user writes what is still queued and stops its thread."""
    key = str(writer.root.resolve())
    with _writers_lock:
        entry = _writers.get(key)
        last = entry is not None and entry[0] is writer and entry[1] <= 1
        if entry is not None and entry[0] is writer:
            entry[1] -= 1
            if last:
                del _writers[key]
    if last or entry is None or entry[0] is not writer:
        return writer.close(timeout)
    return writer.flush(timeout)


def pending_writer(root=None) -> ArchiveWriter | None:
    key = str(pathlib.Path(root or default_root()).resolve())
    with _writers_lock:
        entry = _writers.get(key)
        return entry[0] if entry else None


# --- command line: python -m visionflow arsiv -------------------------------------------------------------------------

CLI_HELP = """Kullanım: python -m visionflow arsiv [seçenekler]

  --seri ABC123               Bu seri numarasının kayıtlarını listeler (içeren; en yenisi önce)
  --disa-aktar DOSYA          Kayıtları dışa aktarır (.csv, .xlsx veya görüntülerle .zip)
  --baslangic TARİH           Bu tarihten (dahil), ör. 2026-10-01 veya 01.10.2026 08:00
  --bitis TARİH               Bu tarihe kadar (tarih yalnızsa o gün dahil)
  --sonuc saglam|hatali       Yalnız sağlam ya da hatalı parçalar
  --istasyon AD  --akis AD  --recete AD
  --klasor KLASÖR             Arşiv klasörü (varsayılan: kullanıcı klasöründe izlenebilirlik)
  --limit N                   Listelenecek en fazla kayıt (varsayılan 50)
  --ozet                      Sayı, hata yüzdesi ve en sık hata sebepleri"""


def cli(argv) -> int:
    import sys
    opts = {"--seri": None, "--disa-aktar": None, "--baslangic": None, "--bitis": None, "--sonuc": None,
            "--istasyon": "", "--akis": "", "--recete": "", "--klasor": None, "--limit": "50"}
    flags = {"--ozet": False}
    args = list(argv)
    i = 0
    try:
        while i < len(args):
            a = args[i]
            if a in ("-h", "--help"):
                print(CLI_HELP)
                return 0
            if a in flags:
                flags[a] = True
                i += 1
                continue
            if a not in opts or i + 1 >= len(args):
                raise ValueError(f"Bilinmeyen veya eksik seçenek: {a}")
            opts[a] = args[i + 1]
            i += 2
        verdict = None
        if opts["--sonuc"]:
            s = fold(opts["--sonuc"]).replace("ğ", "g")
            if s in ("saglam", "ok", "iyi"):
                verdict = True
            elif s in ("hatali", "hatalı", "nok", "kotu"):
                verdict = False
            else:
                raise ValueError("--sonuc 'saglam' veya 'hatali' olmalı")
        flt = Filter(start=parse_date(opts["--baslangic"]) if opts["--baslangic"] else None,
                     end=parse_date(opts["--bitis"], end=True) if opts["--bitis"] else None,
                     serial=opts["--seri"] or "", verdict=verdict, flow=opts["--akis"], station=opts["--istasyon"],
                     recipe=opts["--recete"])
        limit = int(opts["--limit"])
        arch = Archive(opts["--klasor"] or None)
        if not arch.exists():
            print(f"Arşiv bulunamadı: {arch.root}", file=sys.stderr)
            return 1
        if opts["--disa-aktar"]:
            print(arch.export(opts["--disa-aktar"], flt=flt))
            return 0
        if flags["--ozet"]:
            s = arch.summary(flt)
            pct = "–" if s["fail_pct"] is None else f"%{s['fail_pct']:.2f}"
            print(f"Kayıt: {s['count']}  Sağlam: {s['good']}  Hatalı: {s['bad']}  Hata oranı: {pct}")
            for reason, n in s["top_reasons"]:
                print(f"  {n:6d}  {reason}")
            return 0
        recs = arch.search(flt, limit=limit)
        if not recs:
            print("Kayıt bulunamadı.")
            return 1
        for r in recs:
            verdict = "—" if r["ok"] is None else ("SAĞLAM" if r["ok"] else "HATALI")
            vals = json.dumps(r["values"], ensure_ascii=False)
            print(f"{r['time']:%Y-%m-%d %H:%M:%S}  {r['serial'] or '-':<20} {verdict:<7} "
                  f"{r['station'] or '-'}  {r['reason']}  {vals}")
            if r["img_path"] or r["ann_path"]:
                print(f"    görüntü: {r['ann_path'] or r['img_path']}")
        total = arch.count(flt)
        if total > len(recs):
            print(f"({total} kayıttan son {len(recs)} tanesi; --limit ile artırın)")
        return 0
    except ValueError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 2
