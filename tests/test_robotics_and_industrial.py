"""Unit tests for Robotics Guidance, SPC Analysis, Label Compliance, Multi-Camera Stitching,
Thermal Vision, Auto-Lighting, and Virtual PLC Simulator blocks."""
import datetime as dt
import time
import cv2
import numpy as np
import pytest

import visionflow.nodes
from visionflow.core.registry import create


def test_robot_guidance():
    node = create("RobotGuidance")
    node.values["scale_mm_per_px"] = 0.5
    node.values["origin_x_px"] = 100.0
    node.values["origin_y_px"] = 100.0
    node.values["offset_z_mm"] = 150.0
    node.setup()

    img = np.zeros((300, 300, 3), dtype=np.uint8)

    out = node.process({
        "image": img,
        "x_px": 200.0,  # (200 - 100) * 0.5 = 50.0 mm
        "y_px": 300.0,  # (300 - 100) * 0.5 = 100.0 mm
        "angle_deg": 45.0,
        "trigger": True,
    })

    assert out["x_mm"] == 50.0
    assert out["y_mm"] == 100.0
    assert out["rz_deg"] == 45.0
    assert out["pick_ready"] is True
    assert "movel" in out["command_str"]
    assert out["overlay_image"] is not None


def test_spc_analyzer():
    node = create("SPCAnalyzer")
    node.values["usl"] = 26.0
    node.values["lsl"] = 24.0
    node.values["window_size"] = 20
    node.setup()

    # Feed 10 steady measurements around 25.0
    measurements = [25.02, 24.98, 25.01, 24.99, 25.00, 25.03, 24.97, 25.02, 24.99, 25.01]
    last_out = None
    for m in measurements:
        last_out = node.process({"measurement": m, "sample_trigger": True})
        node.process({"sample_trigger": False})

    assert last_out is not None
    assert last_out["sample_count"] == 10
    assert 24.95 <= last_out["mean"] <= 25.05
    assert last_out["cp"] > 1.33   # High process capability
    assert last_out["cpk"] > 1.33
    assert last_out["in_control"] is True


def test_fast_template_aligner():
    node = create("FastTemplateAligner")
    node.setup()

    # Create synthetic image with a rotated white rectangle on black background
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    rect_pts = np.array([[60, 40], [140, 40], [140, 160], [60, 160]], dtype=np.int32)
    cv2.fillPoly(img, [rect_pts], (255, 255, 255))

    out = node.process({"image": img, "trigger": True})
    assert out["found"] is True
    assert out["aligned_image"] is not None
    assert out["aligned_image"].shape == (200, 200, 3)


def test_label_compliance_checker():
    node = create("LabelComplianceChecker")
    node.values["check_barcode_lot"] = True
    node.setup()

    # Future date: should pass
    future_date = (dt.date.today() + dt.timedelta(days=120)).strftime("%d.%m.%Y")
    text_ok = f"ILAC A.S. / SKT: {future_date} / LOT: L77889"
    out_ok = node.process({
        "ocr_text": text_ok,
        "barcode_data": "L77889",
    })
    assert out_ok["is_compliant"] is True
    assert out_ok["is_expired"] is False
    assert out_ok["lot_match"] is True

    # Expired date: should fail
    past_date = "01.01.2020"
    text_exp = f"GIDA SAN / TETT: {past_date} / PARTI: B123"
    out_exp = node.process({
        "ocr_text": text_exp,
        "barcode_data": "B123",
    })
    assert out_exp["is_compliant"] is False
    assert out_exp["is_expired"] is True


def test_multi_camera_stitcher():
    node = create("MultiCameraStitcher")
    node.values["layout"] = "Yatay Yan Yana"
    node.setup()

    im1 = np.ones((100, 120, 3), dtype=np.uint8) * 50
    im2 = np.ones((100, 140, 3), dtype=np.uint8) * 150

    out = node.process({"image_1": im1, "image_2": im2})
    assert out["success"] is True
    assert out["stitch_width"] == 120 + 140
    assert out["stitch_height"] == 100


def test_thermal_inspector():
    node = create("ThermalInspector")
    node.values["alarm_threshold_temp"] = 80.0
    node.values["temp_min_scale"] = 20.0
    node.values["temp_max_scale"] = 100.0
    node.setup()

    # Image with a very bright (hot) spot
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50, 50] = (255, 255, 255)  # Hottest pixel = 100°C

    out = node.process({"image": img})
    assert out["max_temp"] == 100.0
    assert out["alarm_active"] is True
    assert out["hotspot_x"] == 50
    assert out["hotspot_y"] == 50


def test_auto_lighting_controller():
    node = create("AutoLightingController")
    node.values["target_luminance"] = 150.0
    node.values["tolerance"] = 5.0
    node.setup()

    # Dark image (luminance 50) -> should increase brightness PWM
    dark_img = np.ones((80, 80, 3), dtype=np.uint8) * 50
    out = node.process({"image": dark_img})

    assert out["current_brightness"] == 50.0
    assert out["adjustment_needed"] > 0
    assert out["pwm_output"] > 128
    assert "LIGHT:SET:" in out["serial_command"]


def test_virtual_plc_simulator():
    node = create("VirtualPLCSimulator")
    node.values["parts_per_minute"] = 600.0  # Fast rate for testing (10 parts/sec)
    node.values["nominal_measurement"] = 30.0
    node.values["tolerance"] = 0.5
    node.setup()

    # Wait a fraction of second for simulated part to arrive
    time.sleep(0.12)
    out = node.process({"reject_cmd": False})

    assert out["total_parts"] >= 1
    assert "SIM-" in out["part_id"]
    assert 28.0 <= out["sim_measurement"] <= 32.0
    assert out["conveyor_running"] is True
