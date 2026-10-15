"""Tests for Enterprise, Industry 4.0, Multimodal VLM, and Black Box video recorder blocks."""
import os
import pathlib
import sqlite3
import numpy as np
import pytest

import visionflow.nodes
from visionflow.core.registry import create


def test_database_connector(tmp_path):
    db_file = str(tmp_path / "test_uretim.db")
    node = create("DatabaseConnector")
    node.values["db_path_or_host"] = db_file
    node.values["table_name"] = "denetim_kayitlari"
    node.setup()

    # Insert 1st part (OK)
    out1 = node.process({
        "trigger": True,
        "part_id": "PRD-5501",
        "status": "OK",
        "measurement": 24.98,
        "notes": "Tolerans dahilinde",
    })
    assert out1["success"] is True
    assert out1["last_id"] == 1

    # Reset trigger between parts (as in live flow frames)
    node.process({"trigger": False})

    # Insert 2nd part (NOK)
    out2 = node.process({
        "trigger": True,
        "part_id": "PRD-5502",
        "status": "NOK",
        "measurement": 26.50,
        "notes": "Ölçü aşımı",
    })
    assert out2["success"] is True
    assert out2["last_id"] == 2

    # Query DB to verify rows
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT parca_id, durum, olcum FROM denetim_kayitlari ORDER BY id ASC")
        rows = cur.fetchall()

    assert len(rows) == 2
    assert rows[0] == ("PRD-5501", "OK", 24.98)
    assert rows[1] == ("PRD-5502", "NOK", 26.50)


def test_zebra_label_printer(tmp_path):
    node = create("ZebraLabelPrinter")
    node.values["target_type"] = "Dosyaya Yaz (.zpl)"
    node.values["output_folder"] = str(tmp_path)
    node.values["barcode_type"] = "QR Kod"
    node.setup()

    out = node.process({
        "trigger": True,
        "part_name": "MOTOR-GOVDE-V2",
        "barcode_data": "LOT2026-99124",
        "status": "OK",
        "extra_info": "Operatör: Ali Y.",
    })

    assert out["printed"] is True
    assert out["count"] == 1

    # Check that .zpl file was created and contains ZPL commands
    zpl_files = list(tmp_path.glob("*.zpl"))
    assert len(zpl_files) == 1
    content = zpl_files[0].read_text(encoding="utf-8")
    assert "^XA" in content
    assert "^XZ" in content
    assert "MOTOR-GOVDE-V2" in content
    assert "LOT2026-99124" in content
    assert "ONAYLANDI (OK)" in content


def test_oee_calculator():
    node = create("OEECalculator")
    node.values["ideal_cycle_time_sec"] = 2.0
    node.values["planned_production_time_min"] = 10.0
    node.setup()

    # Pass 9 good parts and 1 defect part
    for i in range(10):
        is_bad = (i == 9)
        out = node.process({
            "part_trigger": True,
            "is_defect": is_bad,
            "is_running": True,
        })
        # Reset rising edge
        node.process({"part_trigger": False})

    assert out["total_parts"] == 10
    assert out["quality_pct"] == 90.0  # 9 OK / 10 Total = 90%
    assert out["oee_pct"] > 0.0
    assert "OEE:" in out["summary"]


def test_cloud_backup_nas(tmp_path):
    # Create sample file to back up
    src_file = tmp_path / "ornek_rapor.pdf"
    src_file.write_text("Örnek PDF rapor içeriği", encoding="utf-8")

    archive_dir = tmp_path / "nas_arsiv"
    node = create("CloudBackup")
    node.values["storage_type"] = "Yerel Ağ Paylaşımı (NAS)"
    node.values["server_or_dest"] = str(archive_dir)
    node.setup()

    out = node.process({
        "trigger": True,
        "file_path": str(src_file),
    })

    assert out["uploaded"] is True
    assert os.path.exists(out["url_or_path"])
    assert pathlib.Path(out["url_or_path"]).name == "ornek_rapor.pdf"


def test_black_box_recorder(tmp_path):
    node = create("BlackBoxRecorder")
    node.values["output_folder"] = str(tmp_path)
    node.values["fps"] = 10.0
    node.values["pre_trigger_sec"] = 0.5   # 5 frames
    node.values["post_trigger_sec"] = 0.5  # 5 frames
    node.setup()

    img = np.zeros((120, 160, 3), dtype=np.uint8)

    # Feed 8 frames to populate ring buffer
    for _ in range(8):
        node.process({"image": img, "trigger": False})

    # Trigger alarm/defect
    out_trig = node.process({"image": img, "trigger": True})
    assert out_trig["is_recording"] is True

    # Feed post-trigger frames to complete recording
    final_out = None
    for _ in range(7):
        final_out = node.process({"image": img, "trigger": False})

    assert final_out is not None
    assert final_out["total_recordings"] == 1
    assert final_out["video_path"] != ""


def test_vlm_inspector_setup():
    node = create("VisualInspectorAI")
    node.setup()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Process without trigger
    out = node.process({"image": img, "trigger": False})
    assert out["done"] is False
    assert "Denetim" in out["analysis"]
