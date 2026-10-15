import numpy as np
import pytest
import cv2

from visionflow.core import NODE_REGISTRY
import visionflow.nodes  # Ensure all nodes are registered


def test_registry_contains_advanced_and_safety_nodes():
    expected = [
        "LaserLine3DProfiler",
        "GlueBeadDispenseInspector",
        "DeltaESpectrophotometer",
        "VirtualSafetyLightCurtain",
        "DotPeenMetalReader",
        "GoldenMasterDifferencer",
    ]
    for name in expected:
        assert name in NODE_REGISTRY, f"Node {name} missing from NODE_REGISTRY"


def test_laser_line_3d_profiler():
    cls = NODE_REGISTRY["LaserLine3DProfiler"]
    node = cls()
    node.values["target_height_mm"] = 5.0
    node.values["scale_z_mm_per_px"] = 0.05
    node.setup()

    # Synthetic image with horizontal laser line
    # Background baseline at y=200, step in center at y=100 (diff = 100px * 0.05 = 5.0mm)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Baseline left
    img[195:205, :100] = 255
    # Step center
    img[95:105, 100:300] = 255
    # Baseline right
    img[195:205, 300:] = 255

    res = node.process({"image": img, "trigger": True})
    assert "profile_height_mm" in res
    assert "flatness_deviation_mm" in res
    assert res["heightmap_image"] is not None
    assert res["is_in_spec"] is True
    assert res["profile_points_json"].startswith("[")


def test_glue_bead_dispense_inspector():
    cls = NODE_REGISTRY["GlueBeadDispenseInspector"]
    node = cls()
    node.values["nominal_width_mm"] = 4.0
    node.values["min_width_mm"] = 2.0
    node.values["max_width_mm"] = 6.0
    node.values["scale_mm_per_px"] = 0.05
    node.setup()

    # Image with continuous white horizontal glue bead (height=80px -> ~4.0mm)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 110), (380, 190), (255, 255, 255), -1)

    res = node.process({"image": img, "trigger": True})
    assert res["bead_continuity_pass"] is True
    assert res["gap_count"] == 0
    assert res["avg_width_mm"] > 0
    assert res["overlay_image"] is not None


def test_delta_e_spectrophotometer():
    cls = NODE_REGISTRY["DeltaESpectrophotometer"]
    node = cls()
    node.values["max_acceptable_delta_e"] = 2.5
    node.values["roi_center_x"] = 100
    node.values["roi_center_y"] = 100
    node.values["roi_radius_px"] = 20
    node.setup()

    # Image with a specific solid color
    img = np.full((200, 200, 3), (136, 91, 46), dtype=np.uint8)  # BGR #2E5B88 in RGB is (46, 91, 136)

    res = node.process({"image": img})
    assert "delta_e_00" in res
    assert res["delta_e_00"] >= 0.0
    assert res["is_color_ok"] is True
    assert res["color_comparison_image"] is not None
    assert "Mükemmel" in res["match_grade"] or "Kabul" in res["match_grade"]


def test_virtual_safety_light_curtain():
    cls = NODE_REGISTRY["VirtualSafetyLightCurtain"]
    node = cls()
    node.values["warning_zone_rect"] = "50,50,350,350"
    node.values["danger_zone_rect"] = "150,150,250,250"
    node.values["min_intruder_area_px"] = 100
    node.values["latch_alarm"] = False
    node.setup()

    # 1. Background clean frames
    empty_frame = np.full((400, 400, 3), 100, dtype=np.uint8)
    for _ in range(5):
        node.process({"image": empty_frame})

    res_clean = node.process({"image": empty_frame})
    assert res_clean["estop_active"] is False
    assert res_clean["warning_slow_active"] is False

    # 2. Intruder in Danger Zone (center 200, 200)
    danger_frame = empty_frame.copy()
    cv2.circle(danger_frame, (200, 200), 25, (255, 255, 255), -1)
    res_danger = node.process({"image": danger_frame})

    assert res_danger["estop_active"] is True
    assert res_danger["intrusion_detected"] is True
    assert res_danger["safety_overlay_image"] is not None


def test_dot_peen_metal_reader():
    cls = NODE_REGISTRY["DotPeenMetalReader"]
    node = cls()
    node.values["expected_regex"] = r"^VIN.*$"
    node.setup()

    # Synthetic dot-peen image: multiple dots in a line
    img = np.full((150, 400, 3), 50, dtype=np.uint8)
    for dot_x in range(40, 360, 30):
        cv2.circle(img, (dot_x, 75), 8, (230, 230, 230), -1)

    res = node.process({"image": img, "trigger": True})
    assert "read_text" in res
    assert res["confidence_pct"] > 0
    assert res["preprocessed_image"] is not None
    assert res["annotated_image"] is not None


def test_golden_master_differencer():
    cls = NODE_REGISTRY["GoldenMasterDifferencer"]
    node = cls()
    node.values["min_ssim_threshold"] = 0.90
    node.setup()

    # Identical live and golden image
    gold = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(gold, (40, 40), (160, 160), (180, 180, 180), -1)

    res = node.process({"image": gold, "golden_master_image": gold})
    assert res["ssim_score"] >= 0.98
    assert res["defect_count"] == 0
    assert res["is_identical"] is True
    assert res["diff_heatmap"] is not None
    assert res["defect_mask"] is not None
