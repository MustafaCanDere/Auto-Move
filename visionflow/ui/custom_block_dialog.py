"""Custom Block Studio (Özel Blok Geliştirme Stüdyosu).
Allows visual configuration of inputs, outputs, parameters, interactive Python/OpenCV
code editing, live execution test with synthetic data, and instant one-click registration
into plugins/ and the blocks palette.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import re
import sys
import time
import traceback
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import NODE_REGISTRY, Node, Param, PortType, load_single_plugin, register


def _resolve_plugins_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        p = pathlib.Path(sys.executable).parent / "plugins"
    else:
        p = pathlib.Path.cwd() / "plugins"
        if not p.exists():
            repo_plugins = pathlib.Path(__file__).resolve().parent.parent.parent / "plugins"
            if repo_plugins.exists():
                p = repo_plugins
    p.mkdir(parents=True, exist_ok=True)
    return p


# Predefined Starter Templates
TEMPLATES = {
    "🖼️ Görüntü Filtresi (OpenCV / Renk & Morfoloji)": {
        "title": "Özel Renk & Morfoloji Filtresi",
        "class_name": "OzelRenkFiltresi",
        "category": "6 · Eklentiler",
        "description": "Seçilen HSV renk aralığını maskeler ve morfolojik temizleme uygular.",
        "inputs": [("image", "image")],
        "outputs": [("image", "image"), ("mask", "image"), ("pixel_count", "number")],
        "params": [
            ("lower_h", "int", 35, 0, 180, "Alt Ton (Hue 0-180)"),
            ("upper_h", "int", 85, 0, 180, "Üst Ton (Hue 0-180)"),
            ("min_sat", "int", 50, 0, 255, "Min Doygunluk (Sat)"),
            ("kernel_size", "int", 5, 1, 31, "Temizleme Çekirdeği (Kernel)"),
        ],
        "code": '''import cv2
import numpy as np
from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE
NUM = PortType.NUMBER


@register
class OzelRenkFiltresi(Node):
    title = "Özel Renk & Morfoloji Filtresi"
    category = "6 · Eklentiler"
    description = "Seçilen HSV renk aralığını maskeler ve morfolojik temizleme uygular."
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("mask", IMG), ("pixel_count", NUM)]
    params = [
        Param("lower_h", "int", 35, "Alt Ton (Hue 0-180)", 0, 180),
        Param("upper_h", "int", 85, "Üst Ton (Hue 0-180)", 0, 180),
        Param("min_sat", "int", 50, "Min Doygunluk (Sat)", 0, 255),
        Param("kernel_size", "int", 5, "Temizleme Çekirdeği (Kernel)", 1, 31),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        # BGR -> HSV dönüşümü
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lh = int(self.v("lower_h"))
        uh = int(self.v("upper_h"))
        sat = int(self.v("min_sat"))

        lower = np.array([lh, sat, 40], dtype=np.uint8)
        upper = np.array([uh, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morfolojik temizleme
        ksize = max(1, int(self.v("kernel_size")) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Maskelenmiş renkli çıktı
        result_img = cv2.bitwise_and(img, img, mask=mask_clean)
        count = int(cv2.countNonZero(mask_clean))

        return {
            "image": result_img,
            "mask": mask_clean,
            "pixel_count": count,
        }
''',
    },
    "🔍 Kusur & Alan Tespiti (Kontur / Eşik Kusur Bulucu)": {
        "title": "Özel Kusur ve Gözenek Dedektörü",
        "class_name": "OzelKusurDedektoru",
        "category": "6 · Eklentiler",
        "description": "Görüntü üzerindeki lekeleri, çapakları veya gözenekleri tespit eder ve işaretler.",
        "inputs": [("image", "image")],
        "outputs": [("image", "image"), ("defect_count", "number"), ("is_clean", "bool"), ("contours", "contours")],
        "params": [
            ("thresh_val", "int", 128, 0, 255, "Eşik Değeri"),
            ("min_area", "int", 50, 1, 50000, "Minimum Kusur Alanı (px)"),
            ("max_area", "int", 8000, 10, 500000, "Maksimum Kusur Alanı (px)"),
        ],
        "code": '''import cv2
import numpy as np
from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
CNT = PortType.CONTOURS


@register
class OzelKusurDedektoru(Node):
    title = "Özel Kusur ve Gözenek Dedektörü"
    category = "6 · Eklentiler"
    description = "Görüntü üzerindeki lekeleri, çapakları veya gözenekleri tespit eder ve işaretler."
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("defect_count", NUM), ("is_clean", BOOL), ("contours", CNT)]
    params = [
        Param("thresh_val", "int", 128, "Eşik Değeri", 0, 255),
        Param("min_area", "int", 50, "Minimum Kusur Alanı (px)", 1, 50000),
        Param("max_area", "int", 8000, "Maksimum Kusur Alanı (px)", 10, 500000),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = img.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY) if vis.ndim == 3 else vis
        th = int(self.v("thresh_val"))
        _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = int(self.v("min_area"))
        max_a = int(self.v("max_area"))

        found_cnts = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_a <= area <= max_a:
                found_cnts.append(c)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, f"Kusur A:{int(area)}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        count = len(found_cnts)
        is_clean = (count == 0)

        # Durum bandı
        stat_col = (0, 180, 0) if is_clean else (0, 0, 220)
        stat_txt = "TEMIZ / KUSUR YOK" if is_clean else f"KUSURLU ({count} adet tespit edildi)"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(vis, stat_txt, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stat_col, 2, cv2.LINE_AA)

        return {
            "image": vis,
            "defect_count": count,
            "is_clean": is_clean,
            "contours": found_cnts,
        }
''',
    },
    "📏 Geometrik Ölçüm & Tolerans (Piksel / mm Mesafe)": {
        "title": "Özel Parça Genişlik & Kalibre Ölçer",
        "class_name": "OzelGenislikOlcer",
        "category": "6 · Eklentiler",
        "description": "Hedef nesnenin en ve boy ölçüsünü kalibrasyon katsayısı ile mm cinsinden hesaplar.",
        "inputs": [("image", "image")],
        "outputs": [("image", "image"), ("width_mm", "number"), ("height_mm", "number"), ("is_in_tolerance", "bool")],
        "params": [
            ("mm_per_pixel", "float", 0.05, 0.001, 10.0, "Ölçek (mm/piksel)"),
            ("target_width_mm", "float", 25.0, 0.1, 1000.0, "Hedef Genişlik (mm)"),
            ("tolerance_mm", "float", 0.5, 0.01, 50.0, "İzin Verilen Tolerans (+/- mm)"),
        ],
        "code": '''import cv2
import numpy as np
from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL


@register
class OzelGenislikOlcer(Node):
    title = "Özel Parça Genişlik & Kalibre Ölçer"
    category = "6 · Eklentiler"
    description = "Hedef nesnenin en ve boy ölçüsünü kalibrasyon katsayısı ile mm cinsinden hesaplar."
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("width_mm", NUM), ("height_mm", NUM), ("is_in_tolerance", BOOL)]
    params = [
        Param("mm_per_pixel", "float", 0.05, "Ölçek (mm/piksel)", 0.001, 10.0),
        Param("target_width_mm", "float", 25.0, "Hedef Genişlik (mm)", 0.1, 1000.0),
        Param("tolerance_mm", "float", 0.5, "İzin Verilen Tolerans (+/- mm)", 0.01, 50.0),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = img.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY) if vis.ndim == 3 else vis
        _, bin_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = float(self.v("mm_per_pixel"))
        target_w = float(self.v("target_width_mm"))
        tol = float(self.v("tolerance_mm"))

        width_mm, height_mm = 0.0, 0.0
        in_tol = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            width_mm = round(w * scale, 2)
            height_mm = round(h * scale, 2)
            in_tol = abs(width_mm - target_w) <= tol

            color = (0, 255, 0) if in_tol else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"W: {width_mm} mm | H: {height_mm} mm", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        return {
            "image": vis,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "is_in_tolerance": in_tol,
        }
''',
    },
    "⚙️ Endüstriyel Karar (OK / NG Kalite Kontrol)": {
        "title": "Özel Kalite Karar Kapısı (OK/NG)",
        "class_name": "OzelKaliteKararKapisi",
        "category": "6 · Eklentiler",
        "description": "Birden fazla kalite kriterini mantıksal olarak birleştirir ve hat çıkışı verir.",
        "inputs": [("is_clean", "bool"), ("in_tolerance", "bool"), ("trigger", "bool")],
        "outputs": [("is_ok", "bool"), ("is_ng", "bool"), ("status_message", "text")],
        "params": [
            ("require_both", "bool", True, "Tüm Kriterlerin Sağlanması Zorunlu"),
            ("invert_logic", "bool", False, "Mantığı Ters Çevir (Invert)"),
        ],
        "code": '''from visionflow.core import Node, Param, PortType, register

BOOL = PortType.BOOL
TEXT = PortType.TEXT


@register
class OzelKaliteKararKapisi(Node):
    title = "Özel Kalite Karar Kapısı (OK/NG)"
    category = "6 · Eklentiler"
    description = "Birden fazla kalite kriterini mantıksal olarak birleştirir ve hat çıkışı verir."
    inputs = [("is_clean", BOOL), ("in_tolerance", BOOL), ("trigger", BOOL)]
    outputs = [("is_ok", BOOL), ("is_ng", BOOL), ("status_message", TEXT)]
    params = [
        Param("require_both", "bool", True, "Tüm Kriterlerin Sağlanması Zorunlu"),
        Param("invert_logic", "bool", False, "Mantığı Ters Çevir (Invert)"),
    ]

    def process(self, inputs):
        c1 = bool(inputs.get("is_clean", True))
        c2 = bool(inputs.get("in_tolerance", True))

        if bool(self.v("require_both")):
            ok = c1 and c2
        else:
            ok = c1 or c2

        if bool(self.v("invert_logic")):
            ok = not ok

        msg = "URUN KALITE: ONAYLANDI (OK)" if ok else f"URUN HATALI (NG) [Temiz:{c1}, Olcu:{c2}]"

        return {
            "is_ok": ok,
            "is_ng": not ok,
            "status_message": msg,
        }
''',
    },
    "🗄️ Metin & Veri Biçimlendirici (Barkod / Seri Port Dizesi)": {
        "title": "Özel Seri Veri ve Barkod Formatlayıcı",
        "class_name": "OzelVeriFormatlayici",
        "category": "6 · Eklentiler",
        "description": "Üretim verilerini, zaman damgasını ve barkodu PLC veya yazıcı için özel bir dizeye dönüştürür.",
        "inputs": [("barcode", "text"), ("is_ok", "bool"), ("counter", "number")],
        "outputs": [("formatted_string", "text"), ("plc_telegram", "text")],
        "params": [
            ("prefix", "str", "PRD-LINE1", "Ön Ek (Prefix)"),
            ("separator", "str", ";", "Ayraç Karakteri"),
            ("add_timestamp", "bool", True, "Zaman Damgası Ekle"),
        ],
        "code": '''import datetime as dt
from visionflow.core import Node, Param, PortType, register

TEXT = PortType.TEXT
BOOL = PortType.BOOL
NUM = PortType.NUMBER


@register
class OzelVeriFormatlayici(Node):
    title = "Özel Seri Veri ve Barkod Formatlayıcı"
    category = "6 · Eklentiler"
    description = "Üretim verilerini, zaman damgasını ve barkodu PLC veya yazıcı için özel bir dizeye dönüştürür."
    inputs = [("barcode", TEXT), ("is_ok", BOOL), ("counter", NUM)]
    outputs = [("formatted_string", TEXT), ("plc_telegram", TEXT)]
    params = [
        Param("prefix", "str", "PRD-LINE1", "Ön Ek (Prefix)"),
        Param("separator", "str", ";", "Ayraç Karakteri"),
        Param("add_timestamp", "bool", True, "Zaman Damgası Ekle"),
    ]

    def process(self, inputs):
        barcode = str(inputs.get("barcode") or "NO_CODE")
        ok_status = "PASS" if bool(inputs.get("is_ok", True)) else "FAIL"
        cnt = int(inputs.get("counter") or 0)

        sep = str(self.v("separator"))
        pref = str(self.v("prefix"))

        parts = [pref, barcode, ok_status, str(cnt)]
        if bool(self.v("add_timestamp")):
            parts.append(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        formatted = sep.join(parts)
        telegram = f"${formatted}#"  # PLC çerçeve formatı ($...#)

        return {
            "formatted_string": formatted,
            "plc_telegram": telegram,
        }
''',
    },
    "⚡ Boş Özel Blok (Minimal Şablon)": {
        "title": "Özel Blok Şablonu",
        "class_name": "OzelBlokSablonu",
        "category": "6 · Eklentiler",
        "description": "Kendi özel mantığınızı yazabileceğiniz temiz başlangıç bloğu.",
        "inputs": [("image", "image")],
        "outputs": [("image", "image")],
        "params": [
            ("enabled", "bool", True, 0, 0, "Aktif"),
        ],
        "code": '''from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE


@register
class OzelBlokSablonu(Node):
    title = "Özel Blok Şablonu"
    category = "6 · Eklentiler"
    description = "Kendi özel mantığınızı yazabileceğiniz temiz başlangıç bloğu."
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [
        Param("enabled", "bool", True, "Aktif"),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        # Özel kodunuzu buraya yazın
        # self.v("enabled") ile parametrelere erişebilirsiniz.
        return {"image": img}
''',
    },
}

PORT_TYPE_MAP = {
    "image": PortType.IMAGE,
    "number": PortType.NUMBER,
    "bool": PortType.BOOL,
    "text": PortType.TEXT,
    "detections": PortType.DETECTIONS,
    "contours": PortType.CONTOURS,
    "any": PortType.ANY,
}


class CustomBlockStudioDialog(QDialog):
    def __init__(self, parent=None, on_saved: Callable[[], None] | None = None):
        super().__init__(parent)
        self.setWindowTitle("🎨 Özel Blok Geliştirme Stüdyosu (Custom Block Studio)")
        self.resize(1120, 720)
        self.setMinimumSize(880, 560)
        self.on_saved = on_saved
        self.current_filepath: pathlib.Path | None = None

        self._build_ui()
        self._load_template("🖼️ Görüntü Filtresi (OpenCV / Renk & Morfoloji)")

    def _build_ui(self):
        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(12, 12, 12, 12)
        root_lay.setSpacing(10)

        # Header Banner
        header = QFrame()
        header.setObjectName("studio_header")
        header.setStyleSheet("""
            #studio_header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1e2b, stop:1 #272c3d);
                border: 1px solid #3c445c;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(8, 4, 8, 4)

        icon_lbl = QLabel("🎨")
        icon_lbl.setFont(QFont("Segoe UI Emoji", 24))
        h_lay.addWidget(icon_lbl)

        txt_lay = QVBoxLayout()
        txt_lay.setSpacing(2)
        h_title = QLabel("Özel Blok Geliştirme Stüdyosu")
        h_title.setStyleSheet("font-size: 15pt; font-weight: bold; color: #4fc3f7;")
        h_sub = QLabel("Görsel port ve parametre yapılandırıcısı, Python/OpenCV kod editörü, canlı simülasyon ve tek tıkla eklenti kaydı.")
        h_sub.setStyleSheet("font-size: 9pt; color: #b0bec5;")
        txt_lay.addWidget(h_title)
        txt_lay.addWidget(h_sub)
        h_lay.addLayout(txt_lay, 1)

        root_lay.addWidget(header)

        # Main Splitter: Left Configuration, Right Code Editor & Test
        splitter = QSplitter(Qt.Horizontal)

        # ------------------ LEFT WIDGET ------------------
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)

        # Templates Picker Box
        tpl_box = QGroupBox("Hazır Şablonlar (Starter Templates)")
        tpl_lay = QHBoxLayout(tpl_box)
        self.tpl_combo = QComboBox()
        self.tpl_combo.addItems(list(TEMPLATES.keys()))
        self.btn_load_tpl = QPushButton("📥 Şablonu Yükle")
        self.btn_load_tpl.clicked.connect(lambda: self._load_template(self.tpl_combo.currentText()))
        tpl_lay.addWidget(self.tpl_combo, 1)
        tpl_lay.addWidget(self.btn_load_tpl)
        left_lay.addWidget(tpl_box)

        # Tabs: General, Ports, Params
        self.tabs = QTabWidget()

        # Tab 1: General Info
        tab_general = QWidget()
        gen_lay = QFormLayout(tab_general)
        gen_lay.setContentsMargins(10, 10, 10, 10)
        gen_lay.setSpacing(10)

        self.txt_title = QLineEdit()
        self.txt_title.setPlaceholderText("Örn: Özel Kenar Dedektörü")
        self.txt_title.textChanged.connect(self._auto_class_name)

        self.txt_class = QLineEdit()
        self.txt_class.setPlaceholderText("Örn: OzelKenarDedektoru")

        self.combo_cat = QComboBox()
        self.combo_cat.setEditable(True)
        self.combo_cat.addItems([
            "6 · Eklentiler",
            "1 · Giriş",
            "2 · Görüntü İşleme",
            "2b · Kalite Kontrol ve Ölçüm",
            "3 · Yapay Zeka ve Tespit",
            "4 · Mantık",
            "5 · Çıkış ve Otomasyon",
            "7 · Özel Bloklarım",
        ])

        self.txt_desc = QLineEdit()
        self.txt_desc.setPlaceholderText("Bloğun işlevini anlatan kısa açıklama / ipucu")

        gen_lay.addRow("Blok Başlığı:", self.txt_title)
        gen_lay.addRow("Python Sınıf Adı:", self.txt_class)
        gen_lay.addRow("Kategori:", self.combo_cat)
        gen_lay.addRow("Açıklama (Tooltip):", self.txt_desc)
        self.tabs.addTab(tab_general, "📋 Genel Bilgiler")

        # Tab 2: Ports
        tab_ports = QWidget()
        ports_lay = QVBoxLayout(tab_ports)
        ports_lay.setContentsMargins(8, 8, 8, 8)
        ports_lay.setSpacing(6)

        # Inputs Table
        ports_lay.addWidget(QLabel("<b>Giriş Portları (Inputs):</b>"))
        self.tbl_inputs = QTableWidget(0, 2)
        self.tbl_inputs.setHorizontalHeaderLabels(["Port Adı", "Veri Tipi"])
        self.tbl_inputs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ports_lay.addWidget(self.tbl_inputs)

        btn_in_lay = QHBoxLayout()
        btn_add_in = QPushButton("➕ Giriş Ekle")
        btn_add_in.clicked.connect(lambda: self._add_port_row(self.tbl_inputs, "image", "image"))
        btn_rem_in = QPushButton("➖ Giriş Sil")
        btn_rem_in.clicked.connect(lambda: self._remove_selected_row(self.tbl_inputs))
        btn_in_lay.addWidget(btn_add_in)
        btn_in_lay.addWidget(btn_rem_in)
        ports_lay.addLayout(btn_in_lay)

        # Outputs Table
        ports_lay.addWidget(QLabel("<b>Çıkış Portları (Outputs):</b>"))
        self.tbl_outputs = QTableWidget(0, 2)
        self.tbl_outputs.setHorizontalHeaderLabels(["Port Adı", "Veri Tipi"])
        self.tbl_outputs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ports_lay.addWidget(self.tbl_outputs)

        btn_out_lay = QHBoxLayout()
        btn_add_out = QPushButton("➕ Çıkış Ekle")
        btn_add_out.clicked.connect(lambda: self._add_port_row(self.tbl_outputs, "image", "image"))
        btn_rem_out = QPushButton("➖ Çıkış Sil")
        btn_rem_out.clicked.connect(lambda: self._remove_selected_row(self.tbl_outputs))
        btn_out_lay.addWidget(btn_add_out)
        btn_out_lay.addWidget(btn_rem_out)
        ports_lay.addLayout(btn_out_lay)

        self.tabs.addTab(tab_ports, "🔌 Portlar (G/Ç)")

        # Tab 3: Parameters
        tab_params = QWidget()
        params_lay = QVBoxLayout(tab_params)
        params_lay.setContentsMargins(8, 8, 8, 8)
        params_lay.setSpacing(6)

        params_lay.addWidget(QLabel("<b>Blok Parametreleri (Kullanıcı Ayarları):</b>"))
        self.tbl_params = QTableWidget(0, 6)
        self.tbl_params.setHorizontalHeaderLabels(["Ad", "Tip", "Varsayılan", "Min", "Max", "Etiket"])
        self.tbl_params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_params.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        params_lay.addWidget(self.tbl_params)

        btn_param_lay = QHBoxLayout()
        btn_add_param = QPushButton("➕ Parametre Ekle")
        btn_add_param.clicked.connect(self._add_param_row)
        btn_rem_param = QPushButton("➖ Parametre Sil")
        btn_rem_param.clicked.connect(lambda: self._remove_selected_row(self.tbl_params))
        btn_param_lay.addWidget(btn_add_param)
        btn_param_lay.addWidget(btn_rem_param)
        params_lay.addLayout(btn_param_lay)

        self.tabs.addTab(tab_params, "⚙️ Parametreler")

        left_lay.addWidget(self.tabs, 1)

        # Sync button: visual -> code
        self.btn_gen_code = QPushButton("🔄 Ayarlardan Python Kod İskeletini Üret")
        self.btn_gen_code.setStyleSheet("font-weight: bold; padding: 7px; color: #81c784;")
        self.btn_gen_code.clicked.connect(self._generate_code_from_ui)
        left_lay.addWidget(self.btn_gen_code)

        splitter.addWidget(left_widget)

        # ------------------ RIGHT WIDGET (CODE & TEST) ------------------
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        # Code Editor Group
        code_box = QGroupBox("Python / OpenCV Kod Editörü")
        code_lay = QVBoxLayout(code_box)
        code_lay.setContentsMargins(8, 8, 8, 8)

        self.editor = QPlainTextEdit()
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.Monospace)
        self.editor.setFont(font)
        self.editor.setTabStopDistance(32)  # 4 spaces tab stop
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #171921;
                color: #e0e0e0;
                border: 1px solid #333a4d;
                border-radius: 4px;
                selection-background-color: #264f78;
            }
        """)
        code_lay.addWidget(self.editor)
        right_lay.addWidget(code_box, 3)

        # Test Console Group
        test_box = QGroupBox("Canlı Test ve Doğrulama (Simülasyon)")
        test_lay = QVBoxLayout(test_box)
        test_lay.setContentsMargins(8, 8, 8, 8)
        test_lay.setSpacing(6)

        test_bar = QHBoxLayout()
        self.btn_test = QPushButton("▶️ Kodu Test Et (Canlı Simülasyon)")
        self.btn_test.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: #ffffff; padding: 6px 14px;")
        self.btn_test.clicked.connect(self.run_test)
        test_bar.addWidget(self.btn_test)

        self.test_status_badge = QLabel("Test durumu: Beklemede")
        self.test_status_badge.setStyleSheet("color: #b0bec5; font-size: 9pt;")
        test_bar.addWidget(self.test_status_badge, 1)
        test_lay.addLayout(test_bar)

        self.test_output = QTextBrowser()
        self.test_output.setMaximumHeight(140)
        self.test_output.setFont(QFont("Consolas", 9))
        self.test_output.setStyleSheet("background-color: #11131a; color: #b0bec5; border: 1px solid #292d3e;")
        test_lay.addWidget(self.test_output)

        right_lay.addWidget(test_box, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([460, 640])
        root_lay.addWidget(splitter, 1)

        # Bottom Bar: Actions
        bot_bar = QHBoxLayout()
        self.btn_open = QPushButton("📂 Eklenti Dosyası Aç…")
        self.btn_open.clicked.connect(self.open_plugin_file)
        bot_bar.addWidget(self.btn_open)

        self.btn_open_folder = QPushButton("📁 Eklentiler Klasörünü Aç")
        self.btn_open_folder.clicked.connect(self._open_plugins_folder)
        bot_bar.addWidget(self.btn_open_folder)

        bot_bar.addStretch()

        self.btn_save = QPushButton("💾 Kaydet ve Palete Ekle")
        self.btn_save.setStyleSheet("font-weight: bold; background-color: #1565c0; color: #ffffff; padding: 8px 18px;")
        self.btn_save.clicked.connect(self.save_and_register)
        bot_bar.addWidget(self.btn_save)

        self.btn_close = QPushButton("Kapat")
        self.btn_close.clicked.connect(self.close)
        bot_bar.addWidget(self.btn_close)

        root_lay.addLayout(bot_bar)

    # ------------------ HELPER METHODS ------------------

    def _auto_class_name(self, text: str):
        if not self.txt_class.isModified() or not self.txt_class.text().strip():
            # Turkish character normalization and PascalCase
            tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
            clean = text.translate(tr_map)
            words = re.findall(r"[A-Za-z0-9]+", clean)
            p_case = "".join(w.capitalize() for w in words)
            if p_case:
                self.txt_class.setText(p_case)

    def _add_port_row(self, table: QTableWidget, name: str, ptype: str):
        row = table.rowCount()
        table.insertRow(row)

        name_item = QTableWidgetItem(name)
        table.setItem(row, 0, name_item)

        combo = QComboBox()
        combo.addItems(list(PORT_TYPE_MAP.keys()))
        idx = combo.findText(ptype.lower())
        if idx >= 0:
            combo.setCurrentIndex(idx)
        table.setCellWidget(row, 1, combo)

    def _remove_selected_row(self, table: QTableWidget):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _add_param_row(self, name="param_1", kind="int", default="10", min_v="0", max_v="100", label="Ayar 1"):
        row = self.tbl_params.rowCount()
        self.tbl_params.insertRow(row)

        self.tbl_params.setItem(row, 0, QTableWidgetItem(name))

        combo = QComboBox()
        combo.addItems(["int", "float", "bool", "str", "choice"])
        idx = combo.findText(kind)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self.tbl_params.setCellWidget(row, 1, combo)

        self.tbl_params.setItem(row, 2, QTableWidgetItem(str(default)))
        self.tbl_params.setItem(row, 3, QTableWidgetItem(str(min_v)))
        self.tbl_params.setItem(row, 4, QTableWidgetItem(str(max_v)))
        self.tbl_params.setItem(row, 5, QTableWidgetItem(str(label)))

    def _load_template(self, tpl_name: str):
        tpl = TEMPLATES.get(tpl_name)
        if not tpl:
            return

        self.txt_title.setText(tpl["title"])
        self.txt_class.setText(tpl["class_name"])
        idx = self.combo_cat.findText(tpl["category"])
        if idx >= 0:
            self.combo_cat.setCurrentIndex(idx)
        else:
            self.combo_cat.setEditText(tpl["category"])
        self.txt_desc.setText(tpl["description"])

        # Inputs
        self.tbl_inputs.setRowCount(0)
        for p_name, p_type in tpl["inputs"]:
            self._add_port_row(self.tbl_inputs, p_name, p_type)

        # Outputs
        self.tbl_outputs.setRowCount(0)
        for p_name, p_type in tpl["outputs"]:
            self._add_port_row(self.tbl_outputs, p_name, p_type)

        # Params
        self.tbl_params.setRowCount(0)
        for p in tpl["params"]:
            if len(p) == 4:
                self._add_param_row(p[0], p[1], p[2], 0, 0, p[3])
            elif len(p) == 5:
                self._add_param_row(p[0], p[1], p[2], p[3], p[4], p[0])
            elif len(p) >= 6:
                self._add_param_row(p[0], p[1], p[2], p[3], p[4], p[5])

        # Code
        self.editor.setPlainText(tpl["code"])
        self.test_output.setText("Hazır şablon yüklendi. '▶️ Kodu Test Et' butonuna basarak anında simüle edebilirsiniz.")
        self.test_status_badge.setText("Şablon hazır")
        self.test_status_badge.setStyleSheet("color: #4fc3f7;")

    def _generate_code_from_ui(self):
        title = self.txt_title.text().strip() or "Özel Blok"
        cls_name = self.txt_class.text().strip() or "OzelBlok"
        cat = self.combo_cat.currentText().strip() or "6 · Eklentiler"
        desc = self.txt_desc.text().strip() or "Özel kullanıcı bloğu."

        # Parse inputs
        in_lines = []
        for r in range(self.tbl_inputs.rowCount()):
            name = (self.tbl_inputs.item(r, 0).text() if self.tbl_inputs.item(r, 0) else f"in_{r}").strip()
            w = self.tbl_inputs.cellWidget(r, 1)
            ptype = w.currentText() if isinstance(w, QComboBox) else "any"
            in_lines.append(f'("{name}", PortType.{ptype.upper()})')

        # Parse outputs
        out_lines = []
        ret_lines = []
        for r in range(self.tbl_outputs.rowCount()):
            name = (self.tbl_outputs.item(r, 0).text() if self.tbl_outputs.item(r, 0) else f"out_{r}").strip()
            w = self.tbl_outputs.cellWidget(r, 1)
            ptype = w.currentText() if isinstance(w, QComboBox) else "any"
            out_lines.append(f'("{name}", PortType.{ptype.upper()})')
            if ptype == "image":
                ret_lines.append(f'"{name}": img,')
            elif ptype == "number":
                ret_lines.append(f'"{name}": 0,')
            elif ptype == "bool":
                ret_lines.append(f'"{name}": True,')
            elif ptype == "text":
                ret_lines.append(f'"{name}": "SONUÇ",')
            else:
                ret_lines.append(f'"{name}": None,')

        # Parse params
        param_lines = []
        for r in range(self.tbl_params.rowCount()):
            pname = (self.tbl_params.item(r, 0).text() if self.tbl_params.item(r, 0) else f"p_{r}").strip()
            w = self.tbl_params.cellWidget(r, 1)
            pkind = w.currentText() if isinstance(w, QComboBox) else "int"
            pdef = (self.tbl_params.item(r, 2).text() if self.tbl_params.item(r, 2) else "0").strip()
            pmin = (self.tbl_params.item(r, 3).text() if self.tbl_params.item(r, 3) else "0").strip()
            pmax = (self.tbl_params.item(r, 4).text() if self.tbl_params.item(r, 4) else "100").strip()
            plabel = (self.tbl_params.item(r, 5).text() if self.tbl_params.item(r, 5) else pname).strip()

            if pkind == "int":
                param_lines.append(f'        Param("{pname}", "{pkind}", {pdef}, "{plabel}", {pmin}, {pmax}),')
            elif pkind == "float":
                param_lines.append(f'        Param("{pname}", "{pkind}", {float(pdef)}, "{plabel}", {float(pmin)}, {float(pmax)}),')
            elif pkind == "bool":
                b_val = "True" if pdef.lower() in ("true", "1", "evet") else "False"
                param_lines.append(f'        Param("{pname}", "{pkind}", {b_val}, "{plabel}"),')
            else:
                param_lines.append(f'        Param("{pname}", "{pkind}", "{pdef}", "{plabel}"),')

        inputs_str = ", ".join(in_lines)
        outputs_str = ", ".join(out_lines)
        params_block = "\n".join(param_lines) if param_lines else "        # Parametre yok"
        returns_block = "\n            ".join(ret_lines) if ret_lines else '"": None,'

        code = f'''import cv2
import numpy as np
from visionflow.core import Node, Param, PortType, register


@register
class {cls_name}(Node):
    title = "{title}"
    category = "{cat}"
    description = "{desc}"
    inputs = [{inputs_str}]
    outputs = [{outputs_str}]
    params = [
{params_block}
    ]

    def process(self, inputs):
        # inputs sözlüğünden bağlı verileri alın
        img = inputs.get("image")

        # Parametrelere self.v("param_adi") ile erişebilirsiniz

        return {{
            {returns_block}
        }}
'''
        self.editor.setPlainText(code)
        self.test_output.setText("✅ Ayarlara göre yeni Python kod şablonu oluşturuldu.")
        self.test_status_badge.setText("Kod güncellendi")
        self.test_status_badge.setStyleSheet("color: #81c784;")

    def run_test(self):
        """Simulate and test the custom block execution in memory with synthetic inputs."""
        code = self.editor.toPlainText().strip()
        if not code:
            self.test_output.setText("Hata: Kod editörü boş!")
            return

        # 1. Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            err = f"❌ Sözdizimi Hatası (SyntaxError):\nSatır {e.lineno}, Sütun {e.offset}: {e.msg}\n{e.text}"
            self.test_output.setText(err)
            self.test_status_badge.setText(f"Sözdizimi hatası (Satır {e.lineno})")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return

        # 2. Namespace execution
        test_ns = {
            "cv2": cv2,
            "np": np,
            "Node": Node,
            "Param": Param,
            "PortType": PortType,
            "register": register,
            "__file__": "<custom_block_test>",
        }

        try:
            compiled = compile(code, "<custom_block_test>", "exec")
            exec(compiled, test_ns)
        except Exception as e:
            tb = traceback.format_exc()
            self.test_output.setText(f"❌ Kod Yürütme Hatası:\n{tb}")
            self.test_status_badge.setText(f"Hata: {type(e).__name__}")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return

        # 3. Locate the Node class
        node_cls = None
        for val in test_ns.values():
            if isinstance(val, type) and issubclass(val, Node) and val is not Node:
                node_cls = val
                break

        if node_cls is None:
            self.test_output.setText("❌ Hata: Kod içerisinde 'Node' sınıfından türeyen bir blok sınıfı bulunamadı!")
            self.test_status_badge.setText("Sınıf bulunamadı")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return

        # 4. Instantiate and build mock inputs
        try:
            node_inst = node_cls()
        except Exception as e:
            self.test_output.setText(f"❌ Blok başlatılırken (__init__) hata oluştu:\n{e}")
            self.test_status_badge.setText("Başlatma hatası")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return

        mock_inputs = {}
        # Generate synthetic image (640x480 with gradient and geometric shapes)
        synth_img = np.zeros((480, 640, 3), dtype=np.uint8)
        synth_img[:, :] = (40, 40, 50)  # Dark slate
        cv2.circle(synth_img, (320, 240), 90, (0, 200, 255), -1)
        cv2.rectangle(synth_img, (120, 100), (240, 220), (50, 220, 50), -1)
        cv2.putText(synth_img, "TEST FRAME", (200, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        for port_name, port_type in getattr(node_inst, "inputs", []):
            if port_type == PortType.IMAGE:
                mock_inputs[port_name] = synth_img.copy()
            elif port_type == PortType.NUMBER:
                mock_inputs[port_name] = 42.0
            elif port_type == PortType.BOOL:
                mock_inputs[port_name] = True
            elif port_type == PortType.TEXT:
                mock_inputs[port_name] = "SAMPLE_LOT_1001"
            elif port_type == PortType.CONTOURS:
                mock_inputs[port_name] = [np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)]
            elif port_type == PortType.DETECTIONS:
                mock_inputs[port_name] = [{"box": [100, 100, 150, 150], "label": "test_obj", "score": 0.98}]
            else:
                mock_inputs[port_name] = None

        # 5. Measure process() execution time
        t0 = time.perf_counter()
        try:
            result = node_inst.process(mock_inputs)
        except Exception as e:
            tb = traceback.format_exc()
            self.test_output.setText(f"❌ process(self, inputs) metodunda hata oluştu:\n{tb}")
            self.test_status_badge.setText(f"İşlem Hatası: {type(e).__name__}")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(result, dict):
            self.test_output.setText(f"❌ Hata: process() metodu sözlük (dict) döndürmelidir. Dönen tip: {type(result).__name__}")
            self.test_status_badge.setText("Geçersiz dönüş tipi")
            self.test_status_badge.setStyleSheet("color: #ef5350;")
            return

        # 6. Report outputs
        out_summary = []
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out_summary.append(f"  • {k}: Görüntü / ndarray (şekil: {v.shape}, tip: {v.dtype})")
            elif isinstance(v, (list, tuple)):
                out_summary.append(f"  • {k}: Liste ({len(v)} öğe)")
            else:
                out_summary.append(f"  • {k}: {type(v).__name__} = {repr(v)[:80]}")

        out_str = "\n".join(out_summary) if out_summary else "  (Boş sözlük döndü)"
        success_msg = (
            f"✅ Test Başarılı! Simülasyon hatasız tamamlandı.\n"
            f"⏱ Çevrim Süresi (Latency): {elapsed_ms:.2f} ms\n"
            f"📦 Üretilen Çıktılar ({len(result)} adet):\n{out_str}"
        )
        self.test_output.setText(success_msg)
        self.test_status_badge.setText(f"✅ Başarılı ({elapsed_ms:.1f} ms)")
        self.test_status_badge.setStyleSheet("color: #81c784; font-weight: bold;")

    def save_and_register(self):
        """Save Python file into plugins/ directory and register dynamically."""
        code = self.editor.toPlainText().strip()
        if not code:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek kod yok!")
            return

        cls_name = self.txt_class.text().strip() or "OzelBlok"
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        clean_cls = cls_name.translate(tr_map)
        file_stem = re.sub(r"(?<!^)(?=[A-Z])", "_", clean_cls).lower()
        file_stem = re.sub(r"[^a-z0-9_]", "", file_stem) or "ozel_blok"

        plugins_dir = _resolve_plugins_dir()
        target_path = plugins_dir / f"{file_stem}.py"

        # Check if already has a file path
        if self.current_filepath:
            target_path = self.current_filepath

        try:
            target_path.write_text(code, encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Kayıt Hatası", f"Dosya yazılamadı:\n{e}")
            return

        self.current_filepath = target_path

        # Dynamic reload
        ok, msg = load_single_plugin(target_path)
        if not ok:
            QMessageBox.warning(self, "Yükleme Uyarısı", f"Dosya kaydedildi ancak çalışma ortamına yüklenirken hata oluştu:\n{msg}")
            return

        if self.on_saved:
            self.on_saved()

        node_title = self.txt_title.text().strip() or cls_name
        QMessageBox.information(
            self,
            "Eklenti Hazır",
            f"'{node_title}' bloğu başarıyla kaydedildi ve palete eklendi!\n\n"
            f"Dosya: {target_path.name}\n"
            f"Konum: {target_path.parent}\n\n"
            f"Artık sol taraftaki 'Bloklar' panelinden sürükleyip akışınıza ekleyebilirsiniz.",
        )

    def open_plugin_file(self):
        plugins_dir = _resolve_plugins_dir()
        fname, _ = QFileDialog.getOpenFileName(self, "Eklenti Python Dosyası Seç", str(plugins_dir), "Python Dosyaları (*.py)")
        if not fname:
            return

        path = pathlib.Path(fname)
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dosya okunamadı: {e}")
            return

        self.editor.setPlainText(content)
        self.current_filepath = path
        self.test_output.setText(f"'{path.name}' dosyası yüklendi. Test edebilir veya düzenleyip kaydedebilirsiniz.")
        self.test_status_badge.setText(f"Açık: {path.name}")
        self.test_status_badge.setStyleSheet("color: #4fc3f7;")

    def _open_plugins_folder(self):
        plugins_dir = _resolve_plugins_dir()
        try:
            os.startfile(str(plugins_dir))
        except Exception as e:
            QMessageBox.information(self, "Klasör Konumu", f"Eklentiler klasörü:\n{plugins_dir}\n({e})")
