"""Smart Report Generator: generates rich HTML/PDF, Word (.docx), and Excel (.xlsx) reports
from WhatsApp, Email, AI, or ready/structured data."""
from __future__ import annotations

import base64
import csv
import datetime as dt
import io
import json
import os
import pathlib
import random
import tempfile
import time

import cv2
import numpy as np

try:
    import docx
    from docx.shared import Inches, Pt, RGBColor as DocxRGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None:
        once_key = f"{key}_once"
        if not node.state.get(once_key, False):
            node.state[once_key] = True
            return True
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


CAT_INPUT = "1 · Giriş"
CAT_OUT = "5 · Çıkış ve Otomasyon"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _frame_to_base64_jpeg(img_frame) -> str:
    """Encode OpenCV frame to base64 data URI."""
    if img_frame is None:
        return ""
    try:
        bgr = _bgr(img_frame)
        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if success:
            b64 = base64.b64encode(buf).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        print(f"[SmartReport] Görsel base64'e dönüştürülemedi: {exc}")
    return ""


def _save_temp_img(img_frame) -> str | None:
    """Save OpenCV frame to temp file and return path."""
    if img_frame is None:
        return None
    try:
        bgr = _bgr(img_frame)
        tmp = tempfile.NamedTemporaryFile(prefix="vf_report_", suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, bgr)
        return tmp.name
    except Exception:
        return None


# ==========================================================================================
# 1. Smart Report Generator (Akıllı Rapor Oluşturucu)
# ==========================================================================================

@register
class SmartReportGenerate(Node):
    title = "Akıllı Rapor Oluştur"
    category = CAT_OUT
    description = (
        "WhatsApp, e-posta veya hazır verilerden (JSON, CSV, sayaçlar) zengin HTML/PDF, Word (.docx) "
        "veya Excel (.xlsx) raporu üretir. KPI sayaç kartları, grafikler, gelen mesaj kutusu, "
        "veri tabloları ve denetim fotoğrafları içerir."
    )
    inputs = [
        ("trigger", BOOL),
        ("title", TEXT),
        ("notes", ANY),
        ("data", ANY),
        ("image", IMG),
        ("kpi_total", NUM),
        ("kpi_ok", NUM),
        ("kpi_nok", NUM),
    ]
    outputs = [
        ("path", TEXT),
        ("done", BOOL),
        ("summary", TEXT),
        ("html_path", TEXT),
        ("docx_path", TEXT),
        ("xlsx_path", TEXT),
    ]
    params = [
        Param("format", "choice", "HTML Raporu (Zengin & Yazdırılabilir)", "Rapor Formatı",
              choices=["HTML Raporu (Zengin & Yazdırılabilir)", "Word Belgesi (.docx)", "Excel Tablosu (.xlsx)", "Tüm Formatlar (HTML + Word + Excel)"]),
        Param("output_folder", "folder", "raporlar", "Kayıt Klasörü"),
        Param("filename", "str", "rapor_{zaman}", "Dosya Adı ({zaman} otomatik eklenir)"),
        Param("theme", "choice", "Modern Gece (Koyu Şık)", "Görsel Tema",
              choices=["Modern Gece (Koyu Şık)", "Kurumsal Açık (Temiz)", "Endüstriyel Amber"]),
        Param("company", "str", "VisionFlow AI Endüstriyel Çözümler", "Şirket / Tesis Adı"),
        Param("author", "str", "VisionFlow Raporlama Motoru", "Raporlayan"),
    ]

    def setup(self):
        self.state.update(
            path="",
            done=False,
            summary="",
            html_path="",
            docx_path="",
            xlsx_path="",
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("path", "")
        st.setdefault("done", False)
        st.setdefault("summary", "")
        st.setdefault("html_path", "")
        st.setdefault("docx_path", "")
        st.setdefault("xlsx_path", "")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {
                "path": st["path"],
                "done": False,
                "summary": st["summary"],
                "html_path": st["html_path"],
                "docx_path": st["docx_path"],
                "xlsx_path": st["xlsx_path"],
            }

        title = str(inputs.get("title") or "Üretim ve Kalite Kontrol Raporu").strip()
        notes_in = inputs.get("notes")
        notes_text = str(notes_in).strip() if notes_in is not None else ""
        company = str(self.v("company") or "VisionFlow").strip()
        author = str(self.v("author") or "VisionFlow Raporlama").strip()
        now_dt = dt.datetime.now()
        stamp = now_dt.strftime("%Y%m%d_%H%M%S")
        date_display = now_dt.strftime("%d.%m.%Y %H:%M:%S")

        # Parse data table
        data_in = inputs.get("data")
        rows = []
        if isinstance(data_in, list):
            rows = [r for r in data_in if isinstance(r, dict)]
        elif isinstance(data_in, dict):
            rows = [data_in]
        elif isinstance(data_in, str) and data_in.strip():
            # Try JSON then CSV
            s = data_in.strip()
            if s.startswith("[") or s.startswith("{"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        rows = [r for r in parsed if isinstance(r, dict)]
                    elif isinstance(parsed, dict):
                        rows = [parsed]
                except Exception:
                    pass
            if not rows:
                try:
                    reader = csv.DictReader(io.StringIO(s))
                    rows = list(reader)
                except Exception:
                    pass

        # Calculate KPIs
        kpi_tot = inputs.get("kpi_total")
        kpi_ok = inputs.get("kpi_ok")
        kpi_nok = inputs.get("kpi_nok")

        # If KPIs not explicitly wired, auto-derive from rows if possible
        if kpi_tot is None and rows:
            kpi_tot = len(rows)
            kpi_ok = sum(1 for r in rows if str(r.get("durum", r.get("status", "OK"))).upper() in ("OK", "SAGLAM", "PASSED", "TRUE", "1"))
            kpi_nok = kpi_tot - kpi_ok

        total_val = int(kpi_tot) if kpi_tot is not None else (len(rows) if rows else 0)
        ok_val = int(kpi_ok) if kpi_ok is not None else total_val
        nok_val = int(kpi_nok) if kpi_nok is not None else max(0, total_val - ok_val)
        yield_pct = round((ok_val / float(total_val) * 100.0), 1) if total_val > 0 else 100.0

        # Handle image
        img_frame = inputs.get("image")
        b64_img = ""
        temp_img_path = None
        if img_frame is not None:
            if isinstance(img_frame, str):
                p = pathlib.Path(_node.resolve_path(img_frame))
                if p.is_file():
                    temp_img_path = str(p)
                    with open(p, "rb") as fh:
                        b64_img = f"data:image/jpeg;base64,{base64.b64encode(fh.read()).decode('ascii')}"
            elif isinstance(img_frame, np.ndarray):
                b64_img = _frame_to_base64_jpeg(img_frame)
                temp_img_path = _save_temp_img(img_frame)

        # Output folder & base filename
        folder = self.v("output_folder") or "raporlar"
        folder_path = pathlib.Path(_node.resolve_path(folder))
        folder_path.mkdir(parents=True, exist_ok=True)

        base_name = str(self.v("filename") or "rapor_{zaman}").replace("{zaman}", stamp)
        fmt_choice = self.v("format")
        theme_choice = self.v("theme")

        html_out = ""
        docx_out = ""
        xlsx_out = ""

        # ==============================================================================
        # 1. HTML REPORT BUILDER
        # ==============================================================================
        if fmt_choice in ("HTML Raporu (Zengin & Yazdırılabilir)", "Tüm Formatlar (HTML + Word + Excel)"):
            is_dark = "Gece" in theme_choice or "Amber" in theme_choice
            is_amber = "Amber" in theme_choice

            bg_color = "#0f172a" if is_dark else "#f8fafc"
            card_bg = "#1e293b" if is_dark else "#ffffff"
            text_main = "#f8fafc" if is_dark else "#0f172a"
            text_muted = "#94a3b8" if is_dark else "#64748b"
            accent = "#f97316" if is_amber else ("#06b6d4" if is_dark else "#2563eb")
            border_color = "rgba(255,255,255,0.08)" if is_dark else "#e2e8f0"

            # Build data table HTML
            table_html = ""
            if rows:
                headers = list(rows[0].keys())
                header_th = "".join(f"<th>{h.upper()}</th>" for h in headers)
                body_trs = ""
                for r in rows:
                    tds = []
                    for h in headers:
                        val = str(r.get(h, ""))
                        val_up = val.upper()
                        if val_up in ("OK", "SAGLAM", "PASSED", "TRUE"):
                            tds.append(f'<td><span class="badge badge-ok">{val}</span></td>')
                        elif val_up in ("NOK", "RED", "HATALI", "FAILED", "FALSE"):
                            tds.append(f'<td><span class="badge badge-nok">{val}</span></td>')
                        else:
                            tds.append(f"<td>{val}</td>")
                    body_trs += f"<tr>{''.join(tds)}</tr>"
                table_html = f"""
                <div class="card">
                    <h3 class="section-title">📊 Ayrıntılı Ölçüm ve Kayıt Tablosu</h3>
                    <div class="table-wrap">
                        <table>
                            <thead><tr>{header_th}</tr></thead>
                            <tbody>{body_trs}</tbody>
                        </table>
                    </div>
                </div>
                """

            notes_html = ""
            if notes_text:
                notes_html = f"""
                <div class="card notes-card">
                    <div class="notes-header">
                        <span class="notes-icon">💬</span>
                        <span class="notes-title">Gelen Bilgi & Operatör / Yapay Zeka Notları</span>
                    </div>
                    <div class="notes-body">{notes_text.replace(chr(10), '<br>')}</div>
                </div>
                """

            img_html = ""
            if b64_img:
                img_html = f"""
                <div class="card">
                    <h3 class="section-title">📷 Denetim ve Görsel Muayene Referansı</h3>
                    <div class="img-container">
                        <img src="{b64_img}" alt="Denetim Görseli" />
                    </div>
                </div>
                """

            html_content = f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: {bg_color};
    --card: {card_bg};
    --text: {text_main};
    --muted: {text_muted};
    --accent: {accent};
    --border: {border_color};
    --ok: #10b981;
    --nok: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    padding: 32px 24px;
    line-height: 1.5;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 28px;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .brand {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); }}
  h1 {{ font-size: 28px; font-weight: 800; margin-top: 4px; }}
  .meta {{ font-size: 13px; color: var(--muted); }}
  .actions button {{
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    font-size: 14px;
    transition: opacity 0.2s;
  }}
  .actions button:hover {{ opacity: 0.85; }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    margin-bottom: 24px;
  }}
  .kpi-card {{ display: flex; flex-direction: column; justify-content: space-between; }}
  .kpi-label {{ font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; }}
  .kpi-value {{ font-size: 36px; font-weight: 800; margin-top: 8px; }}
  .kpi-ok {{ color: var(--ok); }}
  .kpi-nok {{ color: var(--nok); }}
  .kpi-yield {{ color: var(--accent); }}
  .section-title {{ font-size: 18px; font-weight: 700; margin-bottom: 16px; color: var(--text); }}
  .notes-card {{ border-left: 4px solid var(--accent); }}
  .notes-header {{ display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; margin-bottom: 8px; }}
  .notes-body {{ font-size: 15px; color: var(--text); background: rgba(0,0,0,0.08); padding: 12px 16px; border-radius: 8px; }}
  .img-container {{ text-align: center; overflow: hidden; border-radius: 8px; background: #000; }}
  .img-container img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 6px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
  th {{ background: rgba(0,0,0,0.15); padding: 12px; font-weight: 700; color: var(--muted); border-bottom: 2px solid var(--border); }}
  td {{ padding: 12px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }}
  .badge-ok {{ background: rgba(16, 185, 129, 0.18); color: var(--ok); }}
  .badge-nok {{ background: rgba(239, 68, 68, 0.18); color: var(--nok); }}
  footer {{
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 32px;
  }}
  @media print {{
    body {{ background: #fff !important; color: #000 !important; padding: 0; }}
    .actions {{ display: none; }}
    .card {{ box-shadow: none; border: 1px solid #ddd; page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <div class="brand">⚡ {company}</div>
      <h1>{title}</h1>
      <div class="meta">📅 {date_display} &nbsp;·&nbsp; 👤 {author}</div>
    </div>
    <div class="actions">
      <button onclick="window.print()">🖨️ PDF / Yazdır</button>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="card kpi-card">
      <div class="kpi-label">Toplam İşlenen</div>
      <div class="kpi-value">{total_val:,}</div>
    </div>
    <div class="card kpi-card">
      <div class="kpi-label">Sağlam Parça (OK)</div>
      <div class="kpi-value kpi-ok">{ok_val:,}</div>
    </div>
    <div class="card kpi-card">
      <div class="kpi-label">Hatalı Parça (NOK)</div>
      <div class="kpi-value kpi-nok">{nok_val:,}</div>
    </div>
    <div class="card kpi-card">
      <div class="kpi-label">Genel Verim Oranı</div>
      <div class="kpi-value kpi-yield">%{yield_pct}</div>
    </div>
  </div>

  {notes_html}
  {img_html}
  {table_html}

  <footer>
    Bu rapor VisionFlow AI Studio tarafından otomatik oluşturulmuştur. © {now_dt.year} {company}
  </footer>
</div>
</body>
</html>
"""
            html_file = folder_path / f"{base_name}.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            html_out = str(html_file.resolve())

        # ==============================================================================
        # 2. WORD (.DOCX) REPORT BUILDER
        # ==============================================================================
        if fmt_choice in ("Word Belgesi (.docx)", "Tüm Formatlar (HTML + Word + Excel)") and DOCX_AVAILABLE:
            doc = docx.Document()

            # Set margins
            for s in doc.sections:
                s.top_margin = Inches(0.8)
                s.bottom_margin = Inches(0.8)
                s.left_margin = Inches(0.8)
                s.right_margin = Inches(0.8)

            # Title & Header
            p_comp = doc.add_paragraph()
            p_comp.text = company.upper()
            p_comp.runs[0].font.size = Pt(10)
            p_comp.runs[0].font.bold = True
            p_comp.runs[0].font.color.rgb = DocxRGBColor(100, 116, 139)

            h1 = doc.add_heading(title, level=0)
            h1.runs[0].font.size = Pt(24)

            p_meta = doc.add_paragraph()
            p_meta.text = f"Tarih: {date_display}    |    Hazırlayan: {author}"
            p_meta.runs[0].font.size = Pt(10)
            p_meta.runs[0].font.italic = True
            doc.add_paragraph()

            # KPI Summary Table
            kpi_tbl = doc.add_table(rows=2, cols=4)
            kpi_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            headers = ["Toplam Parça", "Sağlam (OK)", "Hatalı (NOK)", "Verim Oranı"]
            values = [str(total_val), str(ok_val), str(nok_val), f"%{yield_pct}"]

            for c_idx in range(4):
                # Header row
                cell_h = kpi_tbl.cell(0, c_idx)
                cell_h.text = headers[c_idx]
                cell_h.paragraphs[0].runs[0].font.bold = True
                cell_h.paragraphs[0].runs[0].font.size = Pt(10)
                # Value row
                cell_v = kpi_tbl.cell(1, c_idx)
                cell_v.text = values[c_idx]
                cell_v.paragraphs[0].runs[0].font.bold = True
                cell_v.paragraphs[0].runs[0].font.size = Pt(16)

            doc.add_paragraph()

            # Notes / Incoming Message
            if notes_text:
                doc.add_heading("Gelen Mesaj ve Operatör Notları", level=2)
                p_note = doc.add_paragraph()
                p_note.text = notes_text
                p_note.runs[0].font.size = Pt(11)
                p_note.runs[0].font.italic = True
                doc.add_paragraph()

            # Inspection Image
            if temp_img_path and pathlib.Path(temp_img_path).is_file():
                doc.add_heading("Denetim Görseli", level=2)
                try:
                    doc.add_picture(temp_img_path, width=Inches(5.5))
                except Exception as exc:
                    print(f"[SmartReport Word] Görsel eklenemedi: {exc}")
                doc.add_paragraph()

            # Data Table
            if rows:
                doc.add_heading("Kayıt ve Ölçüm Detayları", level=2)
                col_keys = list(rows[0].keys())
                dt_tbl = doc.add_table(rows=len(rows) + 1, cols=len(col_keys))
                dt_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                # Header
                for ci, k in enumerate(col_keys):
                    c = dt_tbl.cell(0, ci)
                    c.text = k.upper()
                    c.paragraphs[0].runs[0].font.bold = True
                # Rows
                for ri, r in enumerate(rows):
                    for ci, k in enumerate(col_keys):
                        dt_tbl.cell(ri + 1, ci).text = str(r.get(k, ""))

            docx_file = folder_path / f"{base_name}.docx"
            doc.save(str(docx_file))
            docx_out = str(docx_file.resolve())

        # ==============================================================================
        # 3. EXCEL (.XLSX) REPORT BUILDER
        # ==============================================================================
        if fmt_choice in ("Excel Tablosu (.xlsx)", "Tüm Formatlar (HTML + Word + Excel)") and OPENPYXL_AVAILABLE:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Üretim Raporu"
            ws.views.sheetView[0].showGridLines = True

            # Colors & Fonts
            hdr_font = Font(name="Calibri", size=16, bold=True, color="1E3A8A")
            sub_font = Font(name="Calibri", size=10, italic=True, color="64748B")
            tbl_hdr_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            tbl_hdr_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
            kpi_lbl_font = Font(name="Calibri", size=10, bold=True, color="475569")
            kpi_val_font = Font(name="Calibri", size=14, bold=True, color="0F172A")
            thin_border = Border(
                left=Side(style="thin", color="CBD5E1"),
                right=Side(style="thin", color="CBD5E1"),
                top=Side(style="thin", color="CBD5E1"),
                bottom=Side(style="thin", color="CBD5E1"),
            )

            # Title
            ws["A1"] = f"{company} - {title}"
            ws["A1"].font = hdr_font
            ws["A2"] = f"Rapor Tarihi: {date_display}  |  Raporlayan: {author}"
            ws["A2"].font = sub_font

            # KPI Block
            ws["A4"] = "Toplam Parça"
            ws["B4"] = "Sağlam (OK)"
            ws["C4"] = "Hatalı (NOK)"
            ws["D4"] = "Verim %"
            for col_l in ("A", "B", "C", "D"):
                ws[f"{col_l}4"].font = kpi_lbl_font
                ws[f"{col_l}4"].border = thin_border

            ws["A5"] = total_val
            ws["B5"] = ok_val
            ws["C5"] = nok_val
            ws["D5"] = f"%{yield_pct}"
            for col_l in ("A", "B", "C", "D"):
                ws[f"{col_l}5"].font = kpi_val_font
                ws[f"{col_l}5"].border = thin_border

            start_row = 7
            if notes_text:
                ws[f"A{start_row}"] = "Operatör / Gelen Not:"
                ws[f"A{start_row}"].font = Font(bold=True)
                ws[f"B{start_row}"] = notes_text
                start_row += 2

            # Data Table
            if rows:
                col_keys = list(rows[0].keys())
                for c_idx, k in enumerate(col_keys, start=1):
                    cell = ws.cell(row=start_row, column=c_idx, value=k.upper())
                    cell.font = tbl_hdr_font
                    cell.fill = tbl_hdr_fill
                    cell.alignment = Alignment(horizontal="center")
                    cell.border = thin_border

                for r_idx, r in enumerate(rows, start=start_row + 1):
                    for c_idx, k in enumerate(col_keys, start=1):
                        cell = ws.cell(row=r_idx, column=c_idx, value=r.get(k, ""))
                        cell.border = thin_border
                        if str(r.get(k, "")).upper() in ("OK", "SAGLAM", "PASSED"):
                            cell.font = Font(color="10B981", bold=True)
                        elif str(r.get(k, "")).upper() in ("NOK", "RED", "HATALI"):
                            cell.font = Font(color="EF4444", bold=True)

            # Auto-fit column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

            xlsx_file = folder_path / f"{base_name}.xlsx"
            wb.save(str(xlsx_file))
            xlsx_out = str(xlsx_file.resolve())

        # Determine primary output path
        primary_path = html_out or docx_out or xlsx_out
        summary_str = f"Rapor hazırlandı: {total_val} parça, {ok_val} sağlam, {nok_val} hatalı, %{yield_pct} verim."

        st["path"] = primary_path
        st["done"] = True
        st["summary"] = summary_str
        st["html_path"] = html_out
        st["docx_path"] = docx_out
        st["xlsx_path"] = xlsx_out

        print(f"[Akıllı Rapor] Oluşturuldu: {primary_path}")
        return {
            "path": primary_path,
            "done": True,
            "summary": summary_str,
            "html_path": html_out,
            "docx_path": docx_out,
            "xlsx_path": xlsx_out,
        }


# ==========================================================================================
# 2. Ready Data Provider (Hazır Veri Kaynağı)
# ==========================================================================================

@register
class ReadyDataProvider(Node):
    title = "Hazır Veri Kaynağı"
    category = CAT_INPUT
    description = (
        "Sunum ve rapor bloklarını anında beslemek için hazır veri sağlar. "
        "Örnek fabrika üretim/kalite verisi üretebilir, CSV veya JSON dosyası okuyabilir "
        "ya da özel JSON/metin girdisini yapılandırılmış veri tablosuna dönüştürür."
    )
    inputs = [("trigger", BOOL)]
    outputs = [
        ("data", ANY),
        ("text_summary", TEXT),
        ("row_count", NUM),
        ("kpi_total", NUM),
        ("kpi_ok", NUM),
        ("kpi_nok", NUM),
        ("yield_pct", NUM),
    ]
    params = [
        Param("source", "choice", "Örnek Üretim & Kalite Verisi", "Veri Kaynağı",
              choices=["Örnek Üretim & Kalite Verisi", "CSV Dosyası", "JSON Dosyası", "Özel JSON Metni"]),
        Param("file_path", "path", "", "Dosya Yolu (CSV / JSON için)"),
        Param("sample_row_count", "int", 12, "Örnek Satır Sayısı", 1, 1000),
        Param("defect_rate_pct", "float", 5.0, "Hedef Hata Oranı %", 0.0, 100.0),
        Param("custom_json", "code", '[\n  {"zaman": "10:00", "istasyon": "Hat-1", "durum": "OK", "olcum_mm": 12.42},\n  {"zaman": "10:15", "istasyon": "Hat-1", "durum": "NOK", "olcum_mm": 14.18}\n]', "Özel JSON"),
    ]

    def setup(self):
        self.state.update(
            data=[],
            text_summary="",
            row_count=0,
            kpi_total=0,
            kpi_ok=0,
            kpi_nok=0,
            yield_pct=100.0,
            loaded=False,
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("loaded", False)

        trig = _rising(self, inputs.get("trigger"), "prev_trig")
        if not trig and st["loaded"]:
            return {
                "data": st["data"],
                "text_summary": st["text_summary"],
                "row_count": st["row_count"],
                "kpi_total": st["kpi_total"],
                "kpi_ok": st["kpi_ok"],
                "kpi_nok": st["kpi_nok"],
                "yield_pct": st["yield_pct"],
            }

        src = self.v("source")
        rows = []

        if src == "Örnek Üretim & Kalite Verisi":
            n_rows = max(1, int(self.v("sample_row_count")))
            defect_pct = float(self.v("defect_rate_pct"))
            stations = ["CNC-1", "Montaj-A", "Lazer-Kaynak", "Kalıp-Press", "Paketleme"]
            operators = ["Ali Y.", "Mehmet K.", "Ayşe T.", "Fatma S.", "Burak D."]
            now = dt.datetime.now()

            for i in range(n_rows):
                is_defect = (random.random() * 100.0) < defect_pct
                status = "NOK" if is_defect else "OK"
                t_stamp = (now - dt.timedelta(minutes=(n_rows - i) * 12)).strftime("%H:%M:%S")
                target_dim = 25.0
                dev = random.uniform(0.6, 1.8) if is_defect else random.uniform(-0.15, 0.15)
                dim = round(target_dim + dev, 2)
                err_type = random.choice(["Yüzey Çiziği", "Çapak", "Ölçü Aşımı", "Eksik Delik"]) if is_defect else "Yok"

                rows.append({
                    "parca_no": f"PRD-{1000 + i}",
                    "zaman": t_stamp,
                    "istasyon": stations[i % len(stations)],
                    "olcum_mm": dim,
                    "durum": status,
                    "hata_turu": err_type,
                    "operator": operators[i % len(operators)],
                })

        elif src == "CSV Dosyası":
            fpath = str(self.v("file_path")).strip()
            if fpath:
                resolved = _node.resolve_path(fpath)
                p = pathlib.Path(resolved)
                if p.is_file():
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        reader = csv.DictReader(fh)
                        rows = list(reader)

        elif src == "JSON Dosyası":
            fpath = str(self.v("file_path")).strip()
            if fpath:
                resolved = _node.resolve_path(fpath)
                p = pathlib.Path(resolved)
                if p.is_file():
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        parsed = json.load(fh)
                        if isinstance(parsed, list):
                            rows = [r for r in parsed if isinstance(r, dict)]
                        elif isinstance(parsed, dict):
                            rows = [parsed]

        elif src == "Özel JSON Metni":
            raw_json = str(self.v("custom_json")).strip()
            if raw_json:
                try:
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, list):
                        rows = [r for r in parsed if isinstance(r, dict)]
                    elif isinstance(parsed, dict):
                        rows = [parsed]
                except Exception as exc:
                    print(f"[Hazır Veri Kaynağı] JSON çözümlenemedi: {exc}")

        # Compute summary & KPIs
        total = len(rows)
        ok_cnt = sum(1 for r in rows if str(r.get("durum", r.get("status", "OK"))).upper() in ("OK", "SAGLAM", "PASSED", "TRUE", "1"))
        nok_cnt = total - ok_cnt
        y_pct = round((ok_cnt / float(total) * 100.0), 1) if total > 0 else 100.0

        summary_text = (
            f"Üretim Özeti: Toplam {total} parça incelendi. "
            f"{ok_cnt} adet sağlam (OK), {nok_cnt} adet hatalı (NOK). "
            f"Genel üretim başarı oranı %{y_pct} olarak kaydedildi."
        )

        st.update(
            data=rows,
            text_summary=summary_text,
            row_count=total,
            kpi_total=total,
            kpi_ok=ok_cnt,
            kpi_nok=nok_cnt,
            yield_pct=y_pct,
            loaded=True,
        )

        return {
            "data": rows,
            "text_summary": summary_text,
            "row_count": total,
            "kpi_total": total,
            "kpi_ok": ok_cnt,
            "kpi_nok": nok_cnt,
            "yield_pct": y_pct,
        }
