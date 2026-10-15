import numpy as np
import pytest
import cv2
import pathlib

from visionflow.core import NODE_REGISTRY
import visionflow.nodes  # Ensure all nodes are registered


def test_registry_contains_new_futuristic_nodes():
    expected = [
        "DigitalProductPassport",
        "WeldSeamBatteryInspector",
        "SelfHealingVisionTuner",
        "ARProjectionGuidance",
        "AcousticVisionFusion",
        "CADSurfaceDeviationHeatmap",
    ]
    for name in expected:
        assert name in NODE_REGISTRY, f"Node {name} missing from NODE_REGISTRY"


def test_digital_product_passport(tmp_path):
    cls = NODE_REGISTRY["DigitalProductPassport"]
    node = cls()
    node.values["export_folder"] = str(tmp_path)
    node.setup()

    # Without trigger: created should be False
    res_idle = node.process({"trigger": False})
    assert res_idle["created"] is False

    # With trigger: should generate HTML passport and QR
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_trig = node.process({
        "trigger": True,
        "part_id": "TEST-BATTERY-POUCH-001",
        "status": True,
        "measurements": {"thickness_mm": 7.42, "length_mm": 210.5},
        "image": test_img,
    })

    assert res_trig["created"] is True
    assert res_trig["passport_qr"] is not None
    assert len(res_trig["passport_hash"]) == 64  # SHA-256
    assert pathlib.Path(res_trig["passport_url"]).exists()
    assert "TEST-BATTERY-POUCH-001" in pathlib.Path(res_trig["passport_url"]).read_text(encoding="utf-8")


def test_weld_seam_battery_inspector():
    cls = NODE_REGISTRY["WeldSeamBatteryInspector"]
    node = cls()
    node.values["nominal_width_mm"] = 5.0
    node.values["width_tolerance_mm"] = 1.0
    node.values["scale_mm_per_px"] = 0.05
    node.setup()

    # Synthetic image with a clean weld seam bead
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 80), (250, 120), (200, 200, 200), -1)  # 40px width ~ 2.0mm

    res = node.process({"image": img, "trigger": True})
    assert "seam_width_mm" in res
    assert "porosity_defect_count" in res
    assert "is_pass" in res
    assert res["overlay_image"] is not None
    assert res["overlay_image"].shape == img.shape


def test_self_healing_vision_tuner():
    cls = NODE_REGISTRY["SelfHealingVisionTuner"]
    node = cls()
    node.values["target_contrast"] = 70.0
    node.setup()

    # Low contrast washed-out / dark image
    dark_img = np.ones((200, 200, 3), dtype=np.uint8) * 30
    res = node.process({"image": dark_img})

    assert res["healed_image"] is not None
    assert res["contrast_score"] >= 0.0
    assert res["blurriness_score"] >= 0.0
    assert "auto_gamma_applied" in res
    assert isinstance(res["healing_active"], bool)


def test_ar_projection_guidance():
    cls = NODE_REGISTRY["ARProjectionGuidance"]
    node = cls()
    node.values["display_mode"] = "Kamera Üstü AR (HUD)"
    node.setup()

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Step 1
    res1 = node.process({"image": img, "trigger_next_step": False, "reset": False})
    assert res1["step_number"] == 1
    assert "Adım 1" in res1["step_text"]
    assert res1["all_steps_done"] is False
    assert res1["ar_projection_image"].shape == (480, 640, 3)

    # Advance to next step
    res2 = node.process({"image": img, "trigger_next_step": True, "reset": False})
    assert res2["step_number"] == 2

    # Reset
    res_reset = node.process({"image": img, "trigger_next_step": False, "reset": True})
    assert res_reset["step_number"] == 1


def test_acoustic_vision_fusion():
    cls = NODE_REGISTRY["AcousticVisionFusion"]
    node = cls()
    node.values["simulation_mode"] = True
    node.values["temp_alarm_c"] = 80.0
    node.setup()

    # Nominal healthy machine state
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    res_healthy = node.process({
        "image": img,
        "temperature_c": 45.0,
        "rpm": 1450.0,
    })

    assert res_healthy["health_index_pct"] >= 70.0
    assert res_healthy["bearing_fault_detected"] is False
    assert res_healthy["vibration_rms_g"] > 0
    assert res_healthy["fusion_hud_image"] is not None

    # Overheated machine with faulty vibration
    res_fault = node.process({
        "image": img,
        "temperature_c": 95.0,  # Above 80°C alarm
        "rpm": 1450.0,
    })

    assert res_fault["bearing_fault_detected"] is True
    assert res_fault["health_index_pct"] < res_healthy["health_index_pct"]


def test_cad_surface_deviation_heatmap():
    cls = NODE_REGISTRY["CADSurfaceDeviationHeatmap"]
    node = cls()
    node.values["tolerance_mm"] = 0.5
    node.values["scale_mm_per_px"] = 0.05
    node.setup()

    # Part image: circle with small burr bump
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, (220, 220, 220), -1)

    res = node.process({"image": img})
    assert res["deviation_heatmap"] is not None
    assert "max_positive_dev_mm" in res
    assert "max_negative_dev_mm" in res
    assert "surface_defect_pct" in res
    assert isinstance(res["is_within_tolerance"], (bool, np.bool_))
    assert "tolerance_report" in res
