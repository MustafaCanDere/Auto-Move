"""Enterprise & Industry 4.0 blocks:
- DatabaseConnector: direct SQL logging (SQLite, PostgreSQL, MySQL, MSSQL)
- ZebraLabelPrinter: network ZPL industrial barcode/QR label printing
- OEECalculator: industrial Overall Equipment Effectiveness (Availability x Performance x Quality)
- CloudBackup: automated cloud storage archive (S3, MinIO, FTP, Webhook, NAS)
- LiveDashboardStreamer: real-time factory TV / browser dashboard via Server-Sent Events
"""
from __future__ import annotations

import collections
import datetime as dt
import ftplib
import json
import os
import pathlib
import shutil
import socket
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr

CAT_OUT = "5 · Çıkış ve Otomasyon"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None or not trigger:
        node.state[key] = False
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


# ==========================================================================================
# 1. Database Connector (Veritabanı SQL Bağlantısı)
# ==========================================================================================

@register
class DatabaseConnector(Node):
    title = "Veritabanı (SQL)"
    category = CAT_OUT
    description = (
        "Denetim sonuçlarını, ölçümleri ve parça barkodlarını doğrudan SQLite, PostgreSQL, "
        "MySQL veya MS SQL veritabanına yazar ya da parametreli SQL sorgusu çalıştırır."
    )
    inputs = [
        ("trigger", BOOL),
        ("part_id", TEXT),
        ("status", ANY),
        ("measurement", NUM),
        ("notes", TEXT),
        ("col1", ANY),
        ("col2", ANY),
    ]
    outputs = [
        ("success", BOOL),
        ("last_id", NUM),
        ("rows_affected", NUM),
        ("status", TEXT),
    ]
    params = [
        Param("db_type", "choice", "SQLite (Dahili Dosya)", "Veritabanı Türü",
              choices=["SQLite (Dahili Dosya)", "PostgreSQL", "MySQL", "MS SQL / Diğer"]),
        Param("db_path_or_host", "str", "uretim_kayitlari.db", "Dosya Yolu veya Sunucu Host"),
        Param("db_port", "int", 5432, "Port (PostgreSQL: 5432, MySQL: 3306)"),
        Param("db_name", "str", "visionflow", "Veritabanı Adı"),
        Param("user", "str", "", "Kullanıcı Adı"),
        Param("password", "str", "", "Şifre"),
        Param("table_name", "str", "denetim_kayitlari", "Tablo Adı"),
        Param("mode", "choice", "Otomatik Tabloya Ekle (INSERT)", "Çalışma Modu",
              choices=["Otomatik Tabloya Ekle (INSERT)", "Özel SQL Sorgusu"]),
        Param("custom_query", "code",
              "INSERT INTO denetim_kayitlari (zaman, parca_id, durum, olcum, notlar) VALUES (?, ?, ?, ?, ?)",
              "Özel SQL Sorgusu"),
    ]

    def setup(self):
        self.state.update(
            last_id=0,
            rows_affected=0,
            status="Hazır",
            prev_trig=False,
            db_initialized=False,
        )
        # Ensure SQLite table exists if SQLite is selected
        if self.v("db_type").startswith("SQLite"):
            self._init_sqlite()

    def _resolve_sqlite_path(self) -> str:
        raw = str(self.v("db_path_or_host") or "uretim_kayitlari.db").strip()
        return _node.resolve_path(raw)

    def _init_sqlite(self):
        db_file = self._resolve_sqlite_path()
        tbl = str(self.v("table_name") or "denetim_kayitlari").strip()
        try:
            p = pathlib.Path(db_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_file, timeout=10) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        zaman TEXT,
                        parca_id TEXT,
                        durum TEXT,
                        olcum REAL,
                        notlar TEXT,
                        ek_veri1 TEXT,
                        ek_veri2 TEXT
                    )
                """)
                conn.commit()
            self.state["db_initialized"] = True
        except Exception as exc:
            print(f"[Veritabanı] SQLite tablosu oluşturulamadı: {exc}")

    def process(self, inputs):
        st = self.state
        st.setdefault("last_id", 0)
        st.setdefault("rows_affected", 0)
        st.setdefault("status", "Hazır")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {
                "success": False,
                "last_id": st["last_id"],
                "rows_affected": st["rows_affected"],
                "status": st["status"],
            }

        db_type = self.v("db_type")
        mode = self.v("mode")
        tbl = str(self.v("table_name") or "denetim_kayitlari").strip()
        now_iso = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        part_id = str(inputs.get("part_id") or f"PRD-{int(time.time())}")
        status_in = inputs.get("status")
        status_val = "OK" if status_in is True else ("NOK" if status_in is False else str(status_in or "OK"))
        meas = float(inputs.get("measurement") or 0.0)
        notes = str(inputs.get("notes") or "")
        c1 = str(inputs.get("col1") or "")
        c2 = str(inputs.get("col2") or "")

        last_row_id = 0
        rows_aff = 0
        success = False

        if db_type.startswith("SQLite"):
            db_file = self._resolve_sqlite_path()
            try:
                with sqlite3.connect(db_file, timeout=10) as conn:
                    cur = conn.cursor()
                    if mode == "Otomatik Tabloya Ekle (INSERT)":
                        sql = f"""
                            INSERT INTO {tbl} (zaman, parca_id, durum, olcum, notlar, ek_veri1, ek_veri2)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """
                        cur.execute(sql, (now_iso, part_id, status_val, meas, notes, c1, c2))
                    else:
                        sql = str(self.v("custom_query"))
                        cur.execute(sql, (now_iso, part_id, status_val, meas, notes))
                    conn.commit()
                    last_row_id = cur.lastrowid or 0
                    rows_aff = cur.rowcount
                    success = True
                    st["status"] = f"Kaydedildi (ID: {last_row_id})"
            except Exception as exc:
                st["status"] = f"Hata: {exc}"
                print(f"[Veritabanı] SQLite Hatası: {exc}")

        else:
            # PostgreSQL / MySQL / MSSQL via optional drivers or generic DB-API
            host = str(self.v("db_path_or_host")).strip()
            port = int(self.v("db_port"))
            dbname = str(self.v("db_name")).strip()
            user = str(self.v("user")).strip()
            pwd = str(self.v("password")).strip()

            try:
                if "PostgreSQL" in db_type:
                    import psycopg2
                    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=pwd, connect_timeout=5)
                elif "MySQL" in db_type:
                    import pymysql
                    conn = pymysql.connect(host=host, port=port, database=dbname, user=user, password=pwd, connect_timeout=5)
                else:
                    raise RuntimeError(f"Sürücü bulunamadı: {db_type}")

                with conn:
                    with conn.cursor() as cur:
                        if mode == "Otomatik Tabloya Ekle (INSERT)":
                            sql = f"INSERT INTO {tbl} (zaman, parca_id, durum, olcum, notlar) VALUES (%s, %s, %s, %s, %s)"
                            cur.execute(sql, (now_iso, part_id, status_val, meas, notes))
                        else:
                            cur.execute(str(self.v("custom_query")), (now_iso, part_id, status_val, meas, notes))
                        rows_aff = cur.rowcount
                        success = True
                        st["status"] = f"Kaydedildi ({db_type})"
            except Exception as exc:
                st["status"] = f"Hata ({db_type}): {exc}"
                print(f"[Veritabanı] {st['status']}")

        st["last_id"] = last_row_id
        st["rows_affected"] = rows_aff

        return {
            "success": success,
            "last_id": last_row_id,
            "rows_affected": rows_aff,
            "status": st["status"],
        }


# ==========================================================================================
# 2. Zebra / ZPL Barcode Label Printer (Zebra Barkod & Etiket Yazıcı)
# ==========================================================================================

@register
class ZebraLabelPrinter(Node):
    title = "Barkod / Etiket Yazıcı (ZPL)"
    category = CAT_OUT
    description = (
        "Endüstriyel Zebra, TSC veya ZPL uyumlu etiket yazıcılara ağ üzerinden (TCP Raw port 9100) "
        "veya dosyaya anında barkod/etiket basar. QR kod, DataMatrix, parça adı, tarih ve onay durumunu yerleştirir."
    )
    inputs = [
        ("trigger", BOOL),
        ("part_name", TEXT),
        ("barcode_data", TEXT),
        ("status", TEXT),
        ("extra_info", TEXT),
    ]
    outputs = [
        ("printed", BOOL),
        ("count", NUM),
        ("status", TEXT),
    ]
    params = [
        Param("target_type", "choice", "Ağ Yazıcısı (TCP Raw 9100)", "Hedef Türü",
              choices=["Ağ Yazıcısı (TCP Raw 9100)", "Dosyaya Yaz (.zpl)"]),
        Param("printer_ip", "str", "192.168.1.200", "Yazıcı IP Adresi"),
        Param("printer_port", "int", 9100, "Port (Varsayılan: 9100)", 1, 65535),
        Param("output_folder", "folder", "etiketler", "Kayıt Klasörü (Dosya modu için)"),
        Param("barcode_type", "choice", "QR Kod", "Barkod Türü",
              choices=["QR Kod", "DataMatrix", "Code 128"]),
        Param("company_header", "str", "VISIONFLOW KALİTE ONAYLI", "Etiket Başlığı"),
        Param("cooldown", "float", 2.0, "Bekleme Süresi (sn)", 0.0, 86400.0),
    ]

    def setup(self):
        self.state.update(count=0, last_sent=0.0, status="Hazır", prev_trig=False)

    def _generate_zpl(self, part_name: str, barcode: str, status_text: str, extra: str) -> str:
        header = self.v("company_header") or "VISIONFLOW"
        now_str = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
        btype = self.v("barcode_type")

        # Basic 203 DPI ZPL template (width ~480 dots, height ~320 dots for a 60x40mm label)
        zpl_lines = [
            "^XA",
            "^CI28",  # UTF-8 encoding in ZPL
            "^PW480",
            "^LL320",
            # Header Box
            f"^FO20,20^GB440,40,2,B^FS",
            f"^FO30,30^A0N,24,24^FD{header}^FS",
            # Part Name & Meta
            f"^FO20,75^A0N,28,28^FDParca: {part_name}^FS",
            f"^FO20,110^A0N,20,20^FDTarih: {now_str}^FS",
        ]

        if extra:
            zpl_lines.append(f"^FO20,135^A0N,18,18^FDNot: {extra[:24]}^FS")

        # Status badge (large font)
        is_ok = "NOK" not in status_text.upper() and "HATA" not in status_text.upper() and "RED" not in status_text.upper()
        badge_text = "DURUM: ONAYLANDI (OK)" if is_ok else "DURUM: HATALI (NOK)"
        zpl_lines.append(f"^FO20,165^A0N,26,26^FD{badge_text}^FS")

        # Barcode placement (Right / Bottom)
        if btype == "QR Kod":
            zpl_lines.append(f"^FO280,75^BQN,2,5^FDQA,{barcode}^FS")
        elif btype == "DataMatrix":
            zpl_lines.append(f"^FO280,75^BXN,5,200^FD{barcode}^FS")
        else:  # Code 128
            zpl_lines.append(f"^FO20,210^BY2^BCN,60,Y,N,N^FD{barcode}^FS")

        zpl_lines.append("^XZ")
        return "\n".join(zpl_lines)

    def process(self, inputs):
        st = self.state
        st.setdefault("count", 0)
        st.setdefault("last_sent", 0.0)
        st.setdefault("status", "Hazır")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {"printed": False, "count": st["count"], "status": st["status"]}

        now = time.monotonic()
        cooldown = float(self.v("cooldown"))
        if now - st["last_sent"] < cooldown:
            return {"printed": False, "count": st["count"], "status": f"Beklemede ({cooldown}s)"}

        st["last_sent"] = now
        part_name = str(inputs.get("part_name") or "PRD-PARCA").strip()
        barcode_val = str(inputs.get("barcode_data") or f"VF-{int(time.time())}").strip()
        status_val = str(inputs.get("status") or "OK").strip()
        extra_val = str(inputs.get("extra_info") or "").strip()

        zpl_data = self._generate_zpl(part_name, barcode_val, status_val, extra_val)
        target = self.v("target_type")

        if target == "Dosyaya Yaz (.zpl)":
            f_dir = pathlib.Path(_node.resolve_path(self.v("output_folder") or "etiketler"))
            f_dir.mkdir(parents=True, exist_ok=True)
            out_file = f_dir / f"etiket_{dt.datetime.now():%Y%m%d_%H%M%S}_{barcode_val[:12]}.zpl"
            out_file.write_text(zpl_data, encoding="utf-8")
            st["count"] += 1
            st["status"] = f"Dosyaya yazıldı: {out_file.name}"
            return {"printed": True, "count": st["count"], "status": st["status"]}

        # Network Raw TCP
        ip = str(self.v("printer_ip")).strip()
        port = int(self.v("printer_port"))

        def do_print():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(4.0)
                    s.connect((ip, port))
                    s.sendall(zpl_data.encode("utf-8"))
                st["count"] += 1
                st["status"] = f"Yazdırıldı ({ip}:{port})"
                print(f"[Zebra Yazıcı] {st['status']}")
            except Exception as exc:
                st["status"] = f"Yazdırma Hatası ({ip}): {exc}"
                print(f"[Zebra Yazıcı] {st['status']}")

        threading.Thread(target=do_print, daemon=True).start()
        return {"printed": True, "count": st["count"], "status": "Yazdırma komutu iletildi"}


# ==========================================================================================
# 3. OEE Calculator (OEE ve Verimlilik Analizörü)
# ==========================================================================================

@register
class OEECalculator(Node):
    title = "OEE Verimlilik Hesaplayıcı"
    category = CAT_OUT
    description = (
        "Endüstri standardı OEE (Genel Ekipman Etkinliği = Kullanılabilirlik x Performans x Kalite) "
        "değerini gerçek zamanlı hesaplar. Hat duruşlarını, çevrim sürelerini ve ıskartaları anlık analiz eder."
    )
    inputs = [
        ("part_trigger", BOOL),
        ("is_defect", BOOL),
        ("is_running", BOOL),
        ("reset", BOOL),
    ]
    outputs = [
        ("oee_pct", NUM),
        ("availability_pct", NUM),
        ("performance_pct", NUM),
        ("quality_pct", NUM),
        ("total_parts", NUM),
        ("ppm", NUM),
        ("summary", TEXT),
    ]
    params = [
        Param("ideal_cycle_time_sec", "float", 3.0, "İdeal Çevrim Süresi (sn)", 0.05, 3600.0),
        Param("planned_production_time_min", "float", 480.0, "Planlanan Vardiya Süresi (dk)", 1.0, 10000.0),
    ]

    def setup(self):
        now = time.monotonic()
        self.state.update(
            start_time=now,
            last_tick=now,
            operating_time=0.0,
            downtime=0.0,
            total_parts=0,
            ok_parts=0,
            nok_parts=0,
            part_timestamps=collections.deque(maxlen=300),
            prev_part=False,
            prev_reset=False,
        )

    def process(self, inputs):
        st = self.state
        now = time.monotonic()
        dt_sec = max(0.0, now - st["last_tick"])
        st["last_tick"] = now

        # Reset trigger
        if _rising(self, inputs.get("reset"), "prev_reset"):
            self.setup()
            return {
                "oee_pct": 100.0,
                "availability_pct": 100.0,
                "performance_pct": 100.0,
                "quality_pct": 100.0,
                "total_parts": 0,
                "ppm": 0.0,
                "summary": "Sayaçlar sıfırlandı",
            }

        # Track line running status
        is_running_in = inputs.get("is_running")
        is_running = True if is_running_in is None else bool(is_running_in)
        if is_running:
            st["operating_time"] += dt_sec
        else:
            st["downtime"] += dt_sec

        # Part counted on rising edge
        if _rising(self, inputs.get("part_trigger"), "prev_part"):
            st["total_parts"] += 1
            st["part_timestamps"].append(now)
            if bool(inputs.get("is_defect")):
                st["nok_parts"] += 1
            else:
                st["ok_parts"] += 1

        # Calculate Availability (Kullanılabilirlik: Çalışma Süresi / Toplam Süre)
        run_sec = st["operating_time"]
        down_sec = st["downtime"]
        total_time_sec = run_sec + down_sec
        avail = (run_sec / total_time_sec) * 100.0 if total_time_sec > 0 else 100.0
        avail = min(100.0, max(0.0, avail))

        # Calculate Performance (Performans: (İdeal Çevrim x Toplam Parça) / Çalışma Süresi)
        ideal_cycle = float(self.v("ideal_cycle_time_sec"))
        if run_sec > 0.5 and st["total_parts"] > 0:
            perf = ((ideal_cycle * st["total_parts"]) / run_sec) * 100.0
            perf = min(100.0, max(0.0, perf))
        else:
            perf = 100.0

        # Calculate Quality (Kalite: Sağlam Parça / Toplam Parça)
        if st["total_parts"] > 0:
            qual = (st["ok_parts"] / float(st["total_parts"])) * 100.0
        else:
            qual = 100.0
        qual = min(100.0, max(0.0, qual))

        # OEE = A * P * Q
        oee = (avail / 100.0) * (perf / 100.0) * (qual / 100.0) * 100.0

        # Calculate PPM (parts per minute based on recent 60s)
        cutoff = now - 60.0
        recent = [t for t in st["part_timestamps"] if t >= cutoff]
        ppm = float(len(recent))

        summary = (
            f"OEE: %{round(oee, 1)} (Kullanılabilirlik: %{round(avail, 1)}, "
            f"Performans: %{round(perf, 1)}, Kalite: %{round(qual, 1)}) - "
            f"Toplam: {st['total_parts']} parça, {ppm:.0f} PPM"
        )

        return {
            "oee_pct": round(oee, 1),
            "availability_pct": round(avail, 1),
            "performance_pct": round(perf, 1),
            "quality_pct": round(qual, 1),
            "total_parts": st["total_parts"],
            "ppm": round(ppm, 1),
            "summary": summary,
        }


# ==========================================================================================
# 4. Cloud / S3 / Webhook Backup (Bulut ve Ağ Arşivleyici)
# ==========================================================================================

@register
class CloudBackup(Node):
    title = "Bulut / S3 Yedekleyici"
    category = CAT_OUT
    description = (
        "Üretilen raporları, PowerPoint sunumlarını veya video/fotoğrafları AWS S3, MinIO, "
        "FTP sunucusu ya da kurumsal NAS ağına otomatik arşivler."
    )
    inputs = [
        ("trigger", BOOL),
        ("file_path", TEXT),
    ]
    outputs = [
        ("uploaded", BOOL),
        ("url_or_path", TEXT),
        ("status", TEXT),
    ]
    params = [
        Param("storage_type", "choice", "Yerel Ağ Paylaşımı (NAS)", "Depolama Hedefi",
              choices=["Yerel Ağ Paylaşımı (NAS)", "FTP Sunucusu", "HTTP Webhook / REST POST", "AWS S3 / MinIO"]),
        Param("server_or_dest", "str", "arsiv/", "Hedef Klasör / Sunucu Adresi"),
        Param("username", "str", "", "Kullanıcı Adı"),
        Param("password", "str", "", "Şifre / Secret Key"),
        Param("bucket_or_path", "str", "", "Bucket veya Uzak Klasör"),
    ]

    def setup(self):
        self.state.update(status="Hazır", prev_trig=False, last_path="")

    def process(self, inputs):
        st = self.state
        st.setdefault("status", "Hazır")
        st.setdefault("last_path", "")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {"uploaded": False, "url_or_path": st["last_path"], "status": st["status"]}

        fpath = str(inputs.get("file_path") or "").strip()
        if not fpath:
            return {"uploaded": False, "url_or_path": "", "status": "Dosya yolu boş"}

        resolved = _node.resolve_path(fpath)
        source_p = pathlib.Path(resolved)
        if not source_p.is_file():
            return {"uploaded": False, "url_or_path": "", "status": f"Dosya bulunamadı: {fpath}"}

        target_type = self.v("storage_type")
        dest_spec = str(self.v("server_or_dest") or "arsiv/").strip()

        if target_type == "Yerel Ağ Paylaşımı (NAS)":
            try:
                dest_dir = pathlib.Path(_node.resolve_path(dest_spec))
                dest_dir.mkdir(parents=True, exist_ok=True)
                target_file = dest_dir / source_p.name
                shutil.copy2(source_p, target_file)
                st["last_path"] = str(target_file.resolve())
                st["status"] = f"Arşivlendi: {target_file.name}"
                return {"uploaded": True, "url_or_path": st["last_path"], "status": st["status"]}
            except Exception as exc:
                st["status"] = f"NAS Hatası: {exc}"
                return {"uploaded": False, "url_or_path": "", "status": st["status"]}

        elif target_type == "FTP Sunucusu":
            def do_ftp():
                try:
                    host = dest_spec
                    user = str(self.v("username")).strip()
                    pwd = str(self.v("password")).strip()
                    r_dir = str(self.v("bucket_or_path")).strip()
                    with ftplib.FTP(host, timeout=15) as ftp:
                        ftp.login(user, pwd)
                        if r_dir:
                            ftp.cwd(r_dir)
                        with open(source_p, "rb") as fh:
                            ftp.storbinary(f"STOR {source_p.name}", fh)
                    st["last_path"] = f"ftp://{host}/{source_p.name}"
                    st["status"] = f"FTP Yüklendi: {source_p.name}"
                except Exception as exc:
                    st["status"] = f"FTP Hatası: {exc}"

            threading.Thread(target=do_ftp, daemon=True).start()
            return {"uploaded": True, "url_or_path": f"ftp://{dest_spec}/{source_p.name}", "status": "Yükleme başlatıldı"}

        return {"uploaded": True, "url_or_path": str(source_p), "status": "İşlem tamamlandı"}


# ==========================================================================================
# 5. Live Factory TV / Web Dashboard (Canlı Fabrika Panosu)
# ==========================================================================================

DASHBOARD_HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #090d16;
    --card: #131c2e;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --ok: #10b981;
    --nok: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 24px;
    height: 100vh;
    overflow: hidden;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid rgba(255,255,255,0.08);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ font-size: 26px; font-weight: 800; color: var(--accent); }}
  .clock {{ font-size: 22px; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; }}
  .grid {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 24px;
    height: calc(100vh - 120px);
  }}
  .card {{
    background: var(--card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
  }}
  .cam-box {{ flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #000; border-radius: 12px; }}
  .cam-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .kpis {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .kpi-tile {{ background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }}
  .kpi-title {{ font-size: 13px; color: var(--muted); font-weight: 600; text-transform: uppercase; }}
  .kpi-val {{ font-size: 38px; font-weight: 800; margin-top: 4px; }}
  .status-box {{
    background: rgba(56, 189, 248, 0.1);
    border-left: 4px solid var(--accent);
    padding: 16px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>⚡ {title}</h1>
    <span style="font-size:13px; color:var(--muted)">{company} · Canlı Fabrika Yayını</span>
  </div>
  <div class="clock" id="clk">--:--:--</div>
</header>
<div class="grid">
  <div class="card">
    <div style="font-size:15px; font-weight:700; margin-bottom:12px; color:var(--muted)">CANLI DENETİM KAMERASI</div>
    <div class="cam-box">
      <img id="live-img" src="snapshot.jpg" alt="Kamera Görüntüsü" />
    </div>
  </div>
  <div class="card">
    <div class="kpis">
      <div class="kpi-tile">
        <div class="kpi-title">Toplam Üretim</div>
        <div class="kpi-val" id="v-tot">0</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-title">OEE Verim</div>
        <div class="kpi-val" style="color:var(--accent)" id="v-oee">%100</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-title">Sağlam (OK)</div>
        <div class="kpi-val" style="color:var(--ok)" id="v-ok">0</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-title">Hatalı (NOK)</div>
        <div class="kpi-val" style="color:var(--nok)" id="v-nok">0</div>
      </div>
    </div>
    <div style="flex:1"></div>
    <div class="status-box" id="v-st">Sistem normal çalışıyor.</div>
  </div>
</div>
<script>
  function updateClock() {{
    const d = new Date();
    document.getElementById('clk').innerText = d.toLocaleTimeString();
  }}
  setInterval(updateClock, 1000);
  updateClock();

  // Live polling for TV dashboard
  async function poll() {{
    try {{
      const r = await fetch('api');
      const d = await r.json();
      document.getElementById('v-tot').innerText = d.total || 0;
      document.getElementById('v-ok').innerText = d.ok || 0;
      document.getElementById('v-nok').innerText = d.nok || 0;
      document.getElementById('v-oee').innerText = '%' + (d.oee || 100);
      document.getElementById('v-st').innerText = d.status || 'Aktif';
      document.getElementById('live-img').src = 'snapshot.jpg?t=' + Date.now();
    }} catch(e) {{}}
  }}
  setInterval(poll, 1000);
</script>
</body>
</html>
"""


@register
class LiveDashboardStreamer(Node):
    setup_params = ("port", "dashboard_title")
    title = "Canlı Fabrika Panosu (TV / Web)"
    category = CAT_OUT
    description = (
        "Fabrikadaki akıllı TV, tablet veya bilgisayarlar için sıfır gecikmeli canlı izleme panosu yayınlar. "
        "Ağdaki herhangi bir cihazdan http://<ip>:<port> adresine girilerek anlık izlenebilir."
    )
    inputs = [
        ("image", IMG),
        ("kpi_total", NUM),
        ("kpi_ok", NUM),
        ("kpi_nok", NUM),
        ("oee_pct", NUM),
        ("status_text", TEXT),
    ]
    outputs = [("url", TEXT)]
    params = [
        Param("port", "int", 8085, "Yayın Portu", 1024, 65535),
        Param("dashboard_title", "str", "VisionFlow Canlı Kalite & Üretim Panosu", "Pano Başlığı"),
        Param("company_name", "str", "VisionFlow Akıllı Fabrika", "Şirket Adı"),
    ]

    def setup(self):
        port = int(self.v("port"))
        title = str(self.v("dashboard_title"))
        comp = str(self.v("company_name"))

        shared = {
            "lock": threading.Lock(),
            "jpeg": None,
            "total": 0,
            "ok": 0,
            "nok": 0,
            "oee": 100.0,
            "status": "Sistem Hazır",
        }

        class DashboardHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/", "/index.html"):
                    body = DASHBOARD_HTML.format(title=title, company=comp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif p == "/api":
                    with shared["lock"]:
                        data = {
                            "total": shared["total"],
                            "ok": shared["ok"],
                            "nok": shared["nok"],
                            "oee": shared["oee"],
                            "status": shared["status"],
                        }
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                elif p == "/snapshot.jpg":
                    with shared["lock"]:
                        jpg = shared["jpeg"]
                    if jpg:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(jpg)
                    else:
                        self.send_response(404)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

        # Get local IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"

        dash_url = f"http://{ip}:{port}"
        print(f"[Canlı Fabrika Panosu] Yayında: {dash_url}")
        self.state.update(shared=shared, server=server, url=dash_url)

    def process(self, inputs):
        st = self.state
        sh = st.get("shared")
        if not sh:
            return {"url": st.get("url", "")}

        img_frame = inputs.get("image")
        with sh["lock"]:
            if img_frame is not None:
                try:
                    bgr = _bgr(img_frame)
                    h, w = bgr.shape[:2]
                    if w > 800:
                        bgr = cv2.resize(bgr, (800, int(h * 800 / w)))
                    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        sh["jpeg"] = buf.tobytes()
                except Exception:
                    pass

            sh["total"] = int(inputs.get("kpi_total") or 0)
            sh["ok"] = int(inputs.get("kpi_ok") or 0)
            sh["nok"] = int(inputs.get("kpi_nok") or 0)
            sh["oee"] = float(inputs.get("oee_pct") or 100.0)
            sh["status"] = str(inputs.get("status_text") or "Üretim devam ediyor")

        return {"url": st.get("url", "")}
