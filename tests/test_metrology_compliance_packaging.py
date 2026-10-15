"""Unit tests for Professional Sub-pixel Metrology, Camera Calibration,
FDA 21 CFR Part 11 Audit Trail, Model Drift Monitoring, DPM Barcode Grading, and Packaging Completeness."""
import json
import os
import cv2
import numpy as np
import pytest

import visionflow.nodes
from visionflow.core.registry import create


def test_subpixel_caliper_gauge():
    node = create("SubpixelCaliperGauge")
    node.values["measurement_mode"] = "Kumpas Genişliği (Caliper)"
    node.values["nominal_mm"] = 10.0
    node.values["scale_mm_per_px"] = 0.1
    node.values["caliper_line"] = "20,50,180,50"
    node.setup()

    # Image with a white bar in the middle (from x=50 to x=150 -> 100px width = 10.0mm)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, 50:150] = 255

    out = node.process({"image": img, "trigger": True})
    assert out["dimension_mm"] > 0.0
    assert 9.5 <= out["dimension_mm"] <= 10.5
    assert out["is_ok"] is True
    assert out["overlay_image"] is not None


def test_camera_calibrator_undistort():
    node = create("CameraCalibrator")
    node.values["mode"] = "Manuel Katsayılar (Undistort)"
    node.values["k1"] = -0.1
    node.setup()

    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    out = node.process({"image": img, "calibrate_trigger": False})

    assert out["calibrated_image"] is not None
    assert out["calibrated_image"].shape == (100, 100, 3)


def test_audit_trail_logger_sha256_chain(tmp_path):
    log_file = tmp_path / "fda_audit_trail.json"
    node = create("AuditTrailLogger")
    node.values["audit_log_file"] = str(log_file)
    node.values["current_user"] = "Uygulama_Muhendisi"
    node.setup()

    # First event
    out1 = node.process({
        "log_trigger": True,
        "action": "Tolerans Eşiği Güncelleme",
        "details": "Nominal değer 25.0 mm'den 25.5 mm'ye çekildi.",
    })
    assert out1["logged"] is True
    assert out1["total_records"] == 1
    assert out1["is_chain_valid"] is True
    hash1 = out1["last_hash"]

    # Toggle trigger
    node.process({"log_trigger": False})

    # Second event
    out2 = node.process({
        "log_trigger": True,
        "action": "Reçete Yükleme",
        "details": "Reçete: Ürün_A_Kalıp_2",
    })
    assert out2["logged"] is True
    assert out2["total_records"] == 2
    assert out2["is_chain_valid"] is True
    hash2 = out2["last_hash"]
    assert hash1 != hash2

    # Verify log file JSON contents
    with open(log_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
    assert records[0]["hash"] == hash1
    assert records[1]["prev_hash"] == hash1
    assert records[1]["hash"] == hash2


def test_model_drift_monitor(tmp_path):
    node = create("ModelDriftMonitor")
    node.values["min_acceptable_confidence"] = 80.0
    node.values["drift_alert_pct"] = 25.0
    node.values["window_size"] = 15
    node.values["save_retraining_samples"] = True
    node.values["retrain_folder"] = str(tmp_path / "egitim_havuzu")
    node.setup()

    img = np.zeros((80, 80, 3), dtype=np.uint8)

    # Feed 10 high-confidence samples
    for _ in range(10):
        out_good = node.process({"confidence": 95.0, "is_defect": False, "image": img})
    assert out_good["drift_detected"] is False

    # Feed 5 low-confidence samples (< 80%)
    last_out = None
    for _ in range(5):
        last_out = node.process({"confidence": 65.0, "is_defect": True, "image": img})

    assert last_out is not None
    # 5 out of 15 samples = 33.3% > 25.0% threshold -> drift alarm triggers!
    assert last_out["drift_detected"] is True
    assert last_out["saved_for_retraining"] >= 1


def test_dpm_barcode_grader():
    node = create("DPMBarcodeGrader")
    node.values["min_passing_grade"] = "Grade C (Kabul Edilebilir)"
    node.values["manual_barcode"] = "DPM-TEST-449"
    node.setup()

    # Create high-contrast synthetic barcode image
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[20:100, 20:100] = 255  # high symbol contrast

    out = node.process({"image": img, "trigger": True})
    assert out["decoded_text"] == "DPM-TEST-449"
    assert out["grade_score"] >= 2.0
    assert out["is_pass"] is True
    assert out["enhanced_image"] is not None


def test_packaging_completeness_checker():
    node = create("PackagingCompletenessChecker")
    node.values["rows"] = 1
    node.values["cols"] = 3
    node.values["start_x"] = 10
    node.values["start_y"] = 10
    node.values["slot_width"] = 40
    node.values["slot_height"] = 40
    node.values["gap_x"] = 10
    node.values["gap_y"] = 10
    node.values["fill_threshold_pct"] = 20.0
    node.setup()

    # Image with slots 1 and 3 filled (white), slot 2 empty (black)
    img = np.zeros((60, 160, 3), dtype=np.uint8)
    img[10:50, 10:50] = 255    # Slot 1: filled
    # Slot 2 (x=60..100) is empty (0)
    img[10:50, 110:150] = 255  # Slot 3: filled

    out = node.process({"image": img, "trigger": True})
    assert out["total_slots"] == 3
    assert out["filled_slots"] == 2
    assert out["missing_slots"] == 1
    assert "2" in out["missing_indices"]
    assert out["is_complete"] is False
    assert out["overlay_image"] is not None
