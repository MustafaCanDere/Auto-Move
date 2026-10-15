"""Tests for WhatsApp, Email, Presentation, and Smart Report generation nodes."""
import os
import pathlib
import tempfile
import numpy as np
import pytest

from visionflow.core.registry import create
from visionflow.nodes.communication import _parse_whatsapp_payload


def test_ready_data_provider():
    node = create("ReadyDataProvider")
    node.values["sample_row_count"] = 8
    node.values["defect_rate_pct"] = 12.5
    node.setup()

    # Process with trigger=True
    out = node.process({"trigger": True})
    assert out["row_count"] == 8
    assert len(out["data"]) == 8
    assert out["kpi_total"] == 8
    assert out["kpi_ok"] + out["kpi_nok"] == 8
    assert "Üretim Özeti" in out["text_summary"]


def test_presentation_generate(tmp_path):
    node = create("PresentationGenerate")
    node.values["output_folder"] = str(tmp_path)
    node.values["filename"] = "test_sunum_{zaman}.pptx"
    node.setup()

    # Dummy image
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :] = (50, 100, 150)

    # Markdown formatted text
    content = (
        "# Günlük Kalite Özeti\n"
        "## Hat 1 - Robotik Hücre\n"
        "- Toplam 500 parça başarıyla denetlendi\n"
        "- 492 adet sağlam parça sevk edildi\n"
        "- 8 parça yüzey çizik hatası nedeniyle ayrıldı\n"
        "# Eylem Planı\n"
        "- Kalıp parlatma işlemi bu gece yapılacak\n"
        "- Tolerans eşikleri güncellendi\n"
    )

    data_table = [
        {"İstasyon": "Hat-1", "Adet": 250, "Durum": "OK"},
        {"İstasyon": "Hat-2", "Adet": 250, "Durum": "OK"},
    ]

    out = node.process({
        "trigger": True,
        "title": "Üretim Değerlendirme Sunumu",
        "subtitle": "Otomasyon ve Kalite Raporu",
        "content": content,
        "image": img,
        "data": data_table,
    })

    assert out["done"] is True
    assert os.path.exists(out["path"])
    assert out["path"].endswith(".pptx")
    assert out["slide_count"] >= 3


def test_smart_report_generate(tmp_path):
    node = create("SmartReportGenerate")
    node.values["output_folder"] = str(tmp_path)
    node.values["filename"] = "test_rapor_{zaman}"
    node.values["format"] = "Tüm Formatlar (HTML + Word + Excel)"
    node.setup()

    # Dummy image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    sample_data = [
        {"parca": "P-101", "olcum": 12.4, "durum": "OK"},
        {"parca": "P-102", "olcum": 14.8, "durum": "NOK"},
    ]

    notes = "WhatsApp'tan gelen mesaj: 2. vardiya operatörü kalıp sıcaklığının normale döndüğünü bildirdi."

    out = node.process({
        "trigger": True,
        "title": "Vardiya Kalite Raporu",
        "notes": notes,
        "data": sample_data,
        "image": img,
        "kpi_total": 2,
        "kpi_ok": 1,
        "kpi_nok": 1,
    })

    assert out["done"] is True
    assert os.path.exists(out["html_path"])
    assert os.path.exists(out["docx_path"])
    assert os.path.exists(out["xlsx_path"])
    assert "1 sağlam" in out["summary"]

    # Verify HTML contents
    with open(out["html_path"], "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "Vardiya Kalite Raporu" in html_text
    assert "WhatsApp" in html_text
    assert "P-101" in html_text


def test_whatsapp_payload_parser():
    # Test Green API format
    green_data = {
        "messageData": {
            "typeMessage": "textMessage",
            "textMessageData": {"textMessage": "Hat 1 arızası giderildi."},
        },
        "senderData": {"sender": "905551234567@c.us"},
        "idMessage": "msg_9988",
    }
    p1 = _parse_whatsapp_payload(green_data)
    assert p1["text"] == "Hat 1 arızası giderildi."
    assert p1["sender"] == "905551234567"
    assert p1["id"] == "msg_9988"

    # Test Generic format
    gen_data = {"text": "Rapor hazır mı?", "from": "+905321112233"}
    p2 = _parse_whatsapp_payload(gen_data)
    assert p2["text"] == "Rapor hazır mı?"
    assert p2["sender"] == "+905321112233"


def test_whatsapp_listen_simulation():
    node = create("WhatsAppListen")
    node.values["mode"] = "Simülasyon / Test"
    node.setup()

    # Trigger a test message
    out = node.process({
        "test_trigger": True,
        "test_message": "Acil: Parça tolerans dışı!",
        "test_sender": "+905001234567",
        "next": False,
    })

    assert out["new"] is True
    assert out["text"] == "Acil: Parça tolerans dışı!"
    assert out["sender"] == "+905001234567"


def test_presentation_from_unstructured_text(tmp_path):
    """Test generating slides from a plain paragraph message (e.g. from WhatsApp or Email)."""
    node = create("PresentationGenerate")
    node.values["output_folder"] = str(tmp_path)
    node.values["theme"] = "Kurumsal Mavi"
    node.setup()

    plain_text = (
        "Hat 3 montaj hattında vardiya boyunca 340 ürün tamamlandı. "
        "Optik denetim kamerası 5 üründe conta oturma hatası tespit etti. "
        "Genel üretim verimi %98.5 seviyesinde gerçekleşti. "
        "Yarın sabah kalıp revizyonu planlanmıştır. "
        "Operatörler bakım prosedürlerini başarıyla tamamladı."
    )

    out = node.process({
        "trigger": True,
        "title": "Hat 3 Durum Değerlendirmesi",
        "subtitle": "Haftalık Vardiya Özeti",
        "content": plain_text,
    })

    assert out["done"] is True
    assert os.path.exists(out["path"])
    assert out["slide_count"] >= 2


def test_smart_report_word_and_excel_content(tmp_path):
    import docx
    import openpyxl

    node = create("SmartReportGenerate")
    node.values["output_folder"] = str(tmp_path)
    node.values["format"] = "Tüm Formatlar (HTML + Word + Excel)"
    node.setup()

    data = [
        {"İstasyon": "CNC-1", "Ölçüm": 25.04, "Durum": "OK"},
        {"İstasyon": "CNC-2", "Ölçüm": 26.89, "Durum": "NOK"},
    ]

    out = node.process({
        "trigger": True,
        "title": "Doğrulama Raporu",
        "notes": "Önemli not: Ölçüm toleransı kontrol edildi.",
        "data": data,
        "kpi_total": 2,
        "kpi_ok": 1,
        "kpi_nok": 1,
    })

    # Verify Word (.docx)
    doc = docx.Document(out["docx_path"])
    full_doc_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Doğrulama Raporu" in full_doc_text
    assert "Önemli not" in full_doc_text

    # Verify Excel (.xlsx)
    wb = openpyxl.load_workbook(out["xlsx_path"])
    ws = wb.active
    assert "CNC-1" in [ws.cell(row=i, column=1).value for i in range(1, 15)]


def test_pipeline_ready_data_to_presentation_and_report(tmp_path):
    """Test full automation chain: ReadyDataProvider -> PresentationGenerate & SmartReportGenerate."""
    provider = create("ReadyDataProvider")
    provider.values["sample_row_count"] = 6
    provider.setup()
    p_out = provider.process({"trigger": True})

    # Feed provider text_summary and data into PresentationGenerate
    pres_node = create("PresentationGenerate")
    pres_node.values["output_folder"] = str(tmp_path / "sunumlar")
    pres_node.setup()

    pres_res = pres_node.process({
        "trigger": True,
        "title": "Otomatik Günlük Sunum",
        "content": p_out["text_summary"],
        "data": p_out["data"],
    })
    assert pres_res["done"] is True
    assert os.path.exists(pres_res["path"])

    # Feed provider text_summary and data into SmartReportGenerate
    rep_node = create("SmartReportGenerate")
    rep_node.values["output_folder"] = str(tmp_path / "raporlar")
    rep_node.setup()

    rep_res = rep_node.process({
        "trigger": True,
        "title": "Otomatik Günlük Rapor",
        "notes": p_out["text_summary"],
        "data": p_out["data"],
        "kpi_total": p_out["kpi_total"],
        "kpi_ok": p_out["kpi_ok"],
        "kpi_nok": p_out["kpi_nok"],
    })
    assert rep_res["done"] is True
    assert os.path.exists(rep_res["html_path"])
