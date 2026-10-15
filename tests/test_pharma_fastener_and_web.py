import numpy as np
import pytest
import cv2
import time

from visionflow.core import NODE_REGISTRY
import visionflow.nodes  # Ensure all nodes are registered


def test_registry_contains_pharma_and_web_nodes():
    expected = [
        "PharmaVialCapInspector",
        "ThreadFastenerInspector",
        "LiquidParticulateInspector",
        "ConveyorBeltSorter",
        "MicroCrackSurfaceFilter",
        "WebRollDefectMapper",
    ]
    for name in expected:
        assert name in NODE_REGISTRY, f"Node {name} missing from NODE_REGISTRY"


def test_pharma_vial_cap_inspector():
    cls = NODE_REGISTRY["PharmaVialCapInspector"]
    node = cls()
    node.values["nominal_fill_height_mm"] = 25.0
    node.values["fill_tolerance_mm"] = 2.0
    node.values["scale_mm_per_px"] = 0.1
    node.values["vial_roi"] = "50,50,250,350"
    node.setup()

    # Image with vial, cap at top, liquid at bottom
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    # Cap (top)
    img[50:110, 50:250] = 180
    # Liquid fill (from y=200 to y=350 -> 150px = 15.0mm)
    img[200:350, 50:250] = 120

    res = node.process({"image": img, "trigger": True})
    assert "fill_level_mm" in res
    assert "cap_present" in res
    assert "stopper_skew_deg" in res
    assert "overall_pass" in res
    assert res["overlay_image"] is not None


def test_thread_fastener_inspector():
    cls = NODE_REGISTRY["ThreadFastenerInspector"]
    node = cls()
    node.values["nominal_pitch_mm"] = 1.25
    node.values["nominal_major_dia_mm"] = 8.0
    node.values["scale_mm_per_px"] = 0.05
    node.setup()

    # Synthetic bolt silhouette: rectangle with serrated edges
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (220, 250), (255, 255, 255), -1)

    res = node.process({"image": img, "trigger": True})
    assert "thread_pitch_mm" in res
    assert "major_diameter_mm" in res
    assert "minor_diameter_mm" in res
    assert "is_thread_pass" in res
    assert "thread_type_detected" in res
    assert res["annotated_image"] is not None


def test_liquid_particulate_inspector():
    cls = NODE_REGISTRY["LiquidParticulateInspector"]
    node = cls()
    node.values["min_particle_area_px"] = 5
    node.values["max_allowed_particles"] = 0
    node.setup()

    # Frame 1: Clean fluid
    f1 = np.full((200, 200, 3), 100, dtype=np.uint8)
    node.process({"image": f1, "reset_sequence": True})

    # Frame 2: Particle moved in fluid
    f2 = f1.copy()
    cv2.circle(f2, (100, 100), 4, (255, 255, 255), -1)
    res = node.process({"image": f2})

    assert res["particulate_count"] >= 1
    assert res["is_pure_pass"] is False
    assert res["particle_trajectories_image"] is not None
    assert "Kirlilik" in res["alert_summary"]


def test_conveyor_belt_sorter():
    cls = NODE_REGISTRY["ConveyorBeltSorter"]
    node = cls()
    node.values["conveyor_speed_m_per_min"] = 60.0  # 1000 mm/s
    node.values["sensor_to_ejector_dist_mm"] = 50.0  # 50 ms transit
    node.values["ejection_pulse_ms"] = 80.0
    node.setup()

    # Clean initial state
    res1 = node.process({"part_detected": False, "is_defective": False})
    assert res1["ejector_blow_signal"] is False

    # Defective part detected at camera
    res_detect = node.process({"part_detected": True, "is_defective": True})
    assert res_detect["total_rejected"] == 1
    assert res_detect["active_queue_count"] == 1

    # Wait for transit time (~0.05s) and check ejection signal
    time.sleep(0.06)
    res_eject = node.process({"part_detected": False, "is_defective": False})
    assert res_eject["ejector_blow_signal"] is True
    assert res_eject["conveyor_monitor_image"] is not None


def test_micro_crack_surface_filter():
    cls = NODE_REGISTRY["MicroCrackSurfaceFilter"]
    node = cls()
    node.values["crack_threshold"] = 20.0
    node.values["min_crack_length_px"] = 15
    node.setup()

    # Image with a thin dark hairline crack
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    cv2.line(img, (50, 100), (150, 100), (30, 30, 30), 1)

    res = node.process({"image": img})
    assert res["crack_detected"] is True
    assert res["max_crack_length_px"] >= 15
    assert res["crack_ridge_image"] is not None
    assert res["overlay_image"] is not None
    assert "Çatlak" in res["defect_severity"]


def test_web_roll_defect_mapper():
    cls = NODE_REGISTRY["WebRollDefectMapper"]
    node = cls()
    node.values["nominal_roll_length_m"] = 1000.0
    node.values["roll_width_mm"] = 1500.0
    node.values["max_allowed_defects_per_100m"] = 3.0
    node.setup()

    # Reset roll
    res_init = node.process({"reset_new_roll": True})
    assert res_init["total_roll_defects"] == 0

    # Defect at 250m, 600mm width
    res_def1 = node.process({
        "defect_detected": True,
        "defect_width_pos_mm": 600.0,
        "current_roll_length_m": 250.0,
    })
    assert res_def1["total_roll_defects"] == 1
    assert res_def1["roll_defect_map_image"] is not None
    assert "Grade A" in res_def1["roll_quality_grade"] or "Grade B" in res_def1["roll_quality_grade"]
    assert "600" in res_def1["defect_positions_json"]
