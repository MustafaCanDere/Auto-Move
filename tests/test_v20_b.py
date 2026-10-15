"""2.0 (b): more machine-vision blocks: feature matching, Hough lines, caliper, blob defects, fill level, colour area,
histogram comparison, auto alignment, tray grid check."""
import json
import math
import pathlib

import cv2
import numpy as np
import pytest

from visionflow.check import check
from visionflow.core import Graph, create, load_builtin
from visionflow.nodes import vision_more as vm

load_builtin()
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
NEW = ["FeatureMatch", "HoughLines", "Caliper", "BlobDefects", "FillLevel", "ColorArea", "HistogramCompare",
       "AutoAlign", "GridInspect"]


def node(name, **values):
    n = create(name)
    n.values.update(values)
    assert n.safe_setup(), n.setup_error
    return n


def textured(w=200, h=160, seed=1):
    rng = np.random.default_rng(seed)
    obj = np.full((h, w, 3), 200, np.uint8)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        xs, ys = rng.integers(0, w, 2), rng.integers(0, h, 2)
        cv2.line(obj, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), c, 2)
    cv2.putText(obj, "AB12", (20, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return obj


def place(obj, angle, scale, center=(320, 240), size=(640, 480), seed=2):
    rng = np.random.default_rng(seed)
    scene = (np.full((size[1], size[0], 3), 90, np.uint8) + rng.integers(0, 20, (size[1], size[0], 3),
                                                                          dtype=np.uint8))
    h, w = obj.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += center[0] - w / 2
    M[1, 2] += center[1] - h / 2
    warped = cv2.warpAffine(obj, M, size)
    m = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, size)
    scene[m > 0] = warped[m > 0]
    return scene


def bar_image(left, right, h=100, w=300, lo=40, hi=200):
    """A bright vertical bar from x = left to x = right (pixel i covering i..i+1), anti-aliased by area
    downsampling so the edges are at sub-pixel positions."""
    k = 20
    big = np.full((h * k, w * k), lo, np.uint8)
    big[:, int(round(left * k)):int(round(right * k))] = hi
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def tray(empty=(2, 7), rows=3, cols=4, cell=60, origin=(20, 30)):
    img = np.full((origin[1] + rows * cell + 20, origin[0] + cols * cell + 20, 3), 235, np.uint8)
    for r in range(rows):
        for c in range(cols):
            x, y = origin[0] + c * cell, origin[1] + r * cell
            cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), (150, 150, 150), 2)
            if r * cols + c + 1 not in empty:
                cv2.circle(img, (x + cell // 2, y + cell // 2), cell // 3, (30, 120, 200), -1)
    return img


# --- registration / common input handling ----------------------------------------------------------------------------

def test_blocks_registered_with_turkish_titles():
    titles = {create(n).title for n in NEW}
    assert {"Özellik Eşleştir (Nesne Bul)", "Çizgi Bul (Hough)", "Kenar Ölç (Kumpas)", "Leke / Kusur Bul (Blob)",
            "Doluluk Seviyesi", "Renk Alanı %", "Histogram Karşılaştır", "Otomatik Hizala",
            "Sayma Izgarası (Tepsi Kontrol)"} == titles
    for n in NEW:
        b = create(n)
        assert b.category == vm.CAT and b.description


@pytest.mark.parametrize("name", NEW)
def test_none_and_wrong_input_give_turkish_error(name):
    n = node(name)
    with pytest.raises(ValueError, match="bağlı değil"):
        n.process({k: None for k, _ in n.inputs})
    with pytest.raises(ValueError, match="resim bağlanmalı"):
        n.process({**{k: None for k, _ in n.inputs}, "image": "merhaba"})


def test_to_bgr8_handles_all_kinds():
    g = np.full((10, 12), 128, np.uint8)
    assert vm.to_bgr8(g).shape == (10, 12, 3)
    assert vm.to_bgr8(np.full((10, 12), 128 * 257, np.uint16))[0, 0, 0] == 128
    assert vm.to_bgr8(np.full((10, 12), 0.5, np.float32))[0, 0, 0] == 128
    assert vm.to_bgr8(np.full((10, 12), 200.0, np.float64))[0, 0, 0] == 200
    f = np.full((10, 12), np.nan, np.float32)
    assert vm.to_bgr8(f).max() == 0
    assert vm.to_bgr8(np.zeros((10, 12, 4), np.uint8)).shape == (10, 12, 3)
    assert vm.to_bgr8(np.zeros((10, 12, 1), np.uint8)).shape == (10, 12, 3)
    assert vm.to_bgr8(np.ones((10, 12), bool))[0, 0, 0] == 255
    with pytest.raises(ValueError, match="boş"):
        vm.to_bgr8(np.zeros((0, 5), np.uint8))


def test_parse_roi():
    assert vm.parse_roi("", (100, 200)) == (0, 0, 200, 100)
    assert vm.parse_roi("150,50,100,100", (100, 200)) == (150, 50, 50, 50)
    with pytest.raises(ValueError, match="dışında"):
        vm.parse_roi("500,500,10,10", (100, 200))
    with pytest.raises(ValueError, match="4 sayı"):
        vm.parse_roi("1,2,3", (100, 200))
    with pytest.raises(ValueError, match="sayı olmalı"):
        vm.parse_roi("a,b,c,d", (100, 200))


# --- 1. feature match ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ORB (hızlı)", "SIFT (daha sağlam, yavaş)"])
def test_feature_match_finds_rotated_scaled_object(method):
    obj = textured()
    scene = place(obj, 30, 1.3, center=(300, 250))
    n = node("FeatureMatch", method=method)
    r = n.process({"image": scene, "reference": obj})
    assert r["found"] and r["count"] == 1
    assert abs(r["angle"] - 30) < 2
    assert abs(r["scale"] - 1.3) < 0.05
    assert abs(r["x"] - 300) < 4 and abs(r["y"] - 250) < 4
    d = r["detections"][0]
    assert set(d) >= {"box", "center", "angle", "scale", "score", "poly"}
    assert r["image"].shape == scene.shape


def test_feature_match_negative_angle_and_reference_file(tmp_path):
    obj = textured(seed=5)
    p = tmp_path / "ref.png"
    cv2.imwrite(str(p), obj)
    n = node("FeatureMatch", reference=str(p))
    r = n.process({"image": place(obj, -45, 0.8), "reference": None})
    assert r["found"] and abs(r["angle"] + 45) < 2 and abs(r["scale"] - 0.8) < 0.05


def test_feature_match_not_found_and_errors():
    obj = textured()
    n = node("FeatureMatch")
    blank = np.full((480, 640, 3), 90, np.uint8)
    r = n.process({"image": blank, "reference": obj})
    assert not r["found"] and r["count"] == 0 and r["angle"] == 0.0
    with pytest.raises(ValueError, match="Referans resim yok"):
        n.process({"image": blank, "reference": None})
    with pytest.raises(ValueError, match="özellik noktası yok"):
        n.process({"image": blank, "reference": np.full((50, 50, 3), 128, np.uint8)})


def test_feature_match_two_objects_and_odd_inputs():
    obj = textured()
    scene = place(obj, 0, 0.8, center=(170, 240), size=(700, 480))
    second = place(obj, 90, 0.8, center=(520, 240), size=(700, 480), seed=3)
    scene[:, 350:] = second[:, 350:]
    n = node("FeatureMatch", max_objects=3)
    r = n.process({"image": scene, "reference": obj})
    assert r["count"] == 2
    angles = sorted(d["angle"] for d in r["detections"])
    assert abs(angles[0]) < 3 and abs(angles[1] - 90) < 3
    # gray 16-bit scene and BGRA reference still work
    g16 = cv2.cvtColor(place(obj, 30, 1.0), cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257
    r = node("FeatureMatch").process({"image": g16, "reference": cv2.cvtColor(obj, cv2.COLOR_BGR2BGRA)})
    assert r["found"] and abs(r["angle"] - 30) < 2


# --- 2. Hough lines --------------------------------------------------------------------------------------------------

def test_hough_lines_angle_and_length():
    img = np.zeros((300, 400), np.uint8)
    a = math.radians(20)
    p0 = (50, 250)
    p1 = (int(round(50 + 300 * math.cos(a))), int(round(250 - 300 * math.sin(a))))
    cv2.line(img, p0, p1, 255, 3)
    cv2.line(img, (50, 30), (150, 30), 255, 3)                    # a shorter horizontal one
    r = node("HoughLines").process({"image": img})
    assert r["found"] and r["count"] >= 2
    assert abs(r["angle"] - 20) < 1.5
    assert abs(r["length"] - 300) < 15
    d = r["detections"][0]
    assert d["line"] and d["angle"] == r["angle"]
    # angle filter keeps only the horizontal one
    r = node("HoughLines", angle_min=-5, angle_max=5).process({"image": img})
    assert r["count"] >= 1 and all(abs(d["angle"]) <= 5 for d in r["detections"])
    assert abs(r["length"] - 100) < 10
    # ROI: only the top part
    r = node("HoughLines", roi="0,0,400,60").process({"image": img})
    assert r["count"] >= 1 and all(d["box"][1] < 60 for d in r["detections"])
    # vertical line is +90, nothing found on a blank picture
    v = np.zeros((300, 400), np.uint8)
    cv2.line(v, (200, 20), (200, 280), 255, 3)
    assert abs(abs(node("HoughLines").process({"image": v})["angle"]) - 90) < 1
    r = node("HoughLines").process({"image": np.zeros((100, 100, 4), np.uint8)})
    assert not r["found"] and r["count"] == 0


# --- 3. caliper ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("left,right", [(100.3, 137.6), (120.0, 180.5), (99.75, 111.25)])
def test_caliper_measures_bar_width_subpixel(left, right):
    img = bar_image(left, right)
    r = node("Caliper", line="50,50,250,50").process({"image": img, "px_per_mm": None})
    assert r["found"]
    assert abs(r["distance"] - (right - left)) < 0.5
    # OpenCV pixel centres are at whole numbers: a boundary at `left` in the drawing lies at left - 0.5
    assert abs(r["edge1"] - (left - 0.5 - 50)) < 0.25
    (x1, y1), (x2, y2) = r["points"]
    assert abs(x1 - (left - 0.5)) < 0.25 and abs(x2 - (right - 0.5)) < 0.25 and y1 == pytest.approx(50)


def test_caliper_mm_reverse_direction_and_polarity():
    img = bar_image(100.0, 140.0)
    r = node("Caliper", line="250,50,50,50", px_per_mm=4.0).process({"image": img, "px_per_mm": None})
    assert r["found"] and abs(r["distance"] - 10.0) < 0.15 and abs(r["distance_px"] - 40) < 0.5
    # the input wins over the setting
    r = node("Caliper", line="50,50,250,50", px_per_mm=4.0).process({"image": img, "px_per_mm": 8.0})
    assert abs(r["distance"] - 5.0) < 0.1
    # only falling edges: there is just one, so nothing to measure
    r = node("Caliper", line="50,50,250,50", edge1="açıktan koyuya", edge2="açıktan koyuya").process(
        {"image": img, "px_per_mm": None})
    assert not r["found"] and r["edges"] == 2 and abs(r["edge1"] - 89.5) < 0.25


def test_caliper_outer_pairing_diagonal_and_inputs():
    img = np.full((200, 300), 40, np.uint8)
    img[:, 100:120] = 200
    img[:, 160:200] = 200
    first = node("Caliper", line="50,100,250,100").process({"image": img, "px_per_mm": None})
    outer = node("Caliper", line="50,100,250,100", pairing="en dış kenarlar").process(
        {"image": img, "px_per_mm": None})
    assert abs(first["distance"] - 20) < 0.5 and abs(outer["distance"] - 100) < 0.5 and first["edges"] == 4
    # a diagonal line through a vertical bar measures width / cos(angle)
    r = node("Caliper", line="50,50,250,150").process({"image": img, "px_per_mm": None})
    assert abs(r["distance"] - 20 / math.cos(math.atan2(100, 200))) < 0.5
    # 16-bit and float input give the same answer
    r16 = node("Caliper", line="50,100,250,100").process({"image": img.astype(np.uint16) * 257, "px_per_mm": None})
    rf = node("Caliper", line="50,100,250,100").process({"image": img / 255.0, "px_per_mm": None})
    assert r16["distance"] == pytest.approx(first["distance"], abs=0.05)
    assert rf["distance"] == pytest.approx(first["distance"], abs=0.05)
    flat = node("Caliper").process({"image": np.full((480, 640), 90, np.uint8), "px_per_mm": None})
    assert not flat["found"] and flat["distance"] == 0
    with pytest.raises(ValueError, match="dışında"):
        node("Caliper", line="0,0,5000,0").process({"image": img, "px_per_mm": None})
    with pytest.raises(ValueError, match="çok kısa"):
        node("Caliper", line="10,10,11,10").process({"image": img, "px_per_mm": None})
    with pytest.raises(ValueError, match="4 sayı"):
        node("Caliper", line="10,10").process({"image": img, "px_per_mm": None})


# --- 4. blobs --------------------------------------------------------------------------------------------------------

def test_blob_defects_on_uneven_surface():
    h, w = 300, 400
    surface = np.tile(np.linspace(110, 190, w), (h, 1)).astype(np.uint8)     # uneven light
    img = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)
    cv2.circle(img, (100, 100), 5, (40, 40, 40), -1)
    cv2.circle(img, (300, 200), 8, (60, 60, 60), -1)
    cv2.circle(img, (200, 250), 6, (255, 255, 255), -1)                       # a light one
    r = node("BlobDefects").process({"image": img})
    assert r["count"] == 2 and not r["ok"]
    assert abs(r["largest_area"] - math.pi * 64) < 0.25 * math.pi * 64
    assert r["total_area"] == pytest.approx(sum(d["area"] for d in r["detections"]))
    assert r["mask"].shape == (h, w) and r["mask"].max() == 255
    assert node("BlobDefects", polarity="açık leke").process({"image": img})["count"] == 1
    assert node("BlobDefects", polarity="ikisi").process({"image": img})["count"] == 3
    assert node("BlobDefects", min_area=150).process({"image": img})["count"] == 1
    assert node("BlobDefects", roi="0,0,200,150").process({"image": img})["count"] == 1
    # a scratch is not round
    cv2.line(img, (20, 280), (150, 280), (40, 40, 40), 2)
    assert node("BlobDefects", min_circularity=0.6).process({"image": img})["count"] == 2
    clean = node("BlobDefects").process({"image": surface.astype(np.uint16) * 257})
    assert clean["ok"] and clean["count"] == 0 and clean["largest_area"] == 0
    big = node("BlobDefects", bg_size=151).process({"image": img})               # big median on a small copy
    assert big["count"] >= 2


def test_blob_global_background():
    img = np.full((100, 100), 150, np.uint8)
    cv2.rectangle(img, (40, 40), (59, 59), 50, -1)
    r = node("BlobDefects", background="tüm yüzeyin ortalaması").process({"image": img})
    assert r["count"] == 1 and r["largest_area"] == 400


# --- 5. fill level ---------------------------------------------------------------------------------------------------

def bottle(level_pct, h=200):
    img = np.full((300, 200, 3), 30, np.uint8)
    x0, y0, w = 50, 50, 100
    img[y0:y0 + h, x0:x0 + w] = 220
    top = y0 + int(round(h * (1 - level_pct / 100)))
    img[top:y0 + h, x0:x0 + w] = (40, 90, 140)
    return img, f"{x0 + 10},{y0},{w - 20},{h}"


@pytest.mark.parametrize("pct", [65, 80, 92])
def test_fill_level_edge_method(pct):
    img, roi = bottle(pct)
    r = node("FillLevel", roi=roi).process({"image": img})
    assert r["found"] and abs(r["level"] - pct) < 1.0
    assert abs(r["level_px"] - pct * 2) < 2
    assert r["ok"] == (70 <= r["level"] <= 95)


def test_fill_level_threshold_method_and_missing():
    img, roi = bottle(50)
    r = node("FillLevel", roi=roi, method="parlaklık eşiği").process({"image": img})
    assert r["found"] and abs(r["level"] - 50) < 1 and not r["ok"]
    light = node("FillLevel", roi=roi, method="parlaklık eşiği", liquid="sıvı daha açık").process(
        {"image": 255 - img})
    assert abs(light["level"] - 50) < 1
    empty = np.full((300, 200, 3), 220, np.uint8)
    r = node("FillLevel", roi=roi).process({"image": empty})
    assert not r["found"] and not r["ok"] and r["level"] == 0
    r = node("FillLevel", roi=roi, method="parlaklık eşiği").process({"image": empty})
    assert not r["found"]


# --- 6. colour area --------------------------------------------------------------------------------------------------

def test_color_area_percent():
    img = np.full((100, 200, 3), 128, np.uint8)
    img[:, :100] = (0, 200, 0)                                    # green left half
    r = node("ColorArea").process({"image": img})
    assert r["percent"] == pytest.approx(50.0) and r["ok"] and r["pixels"] == 10000
    assert r["mask"].shape == (100, 200)
    r = node("ColorArea", roi="50,0,100,100", min_pct=60).process({"image": img})
    assert r["percent"] == pytest.approx(50.0) and not r["ok"]
    r = node("ColorArea", roi="0,0,100,100").process({"image": cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)})
    assert r["percent"] == pytest.approx(100.0)
    red = np.full((10, 10, 3), (0, 0, 220), np.uint8)
    assert node("ColorArea", h_min=170, h_max=10).process({"image": red})["percent"] == 100.0
    assert node("ColorArea").process({"image": np.full((10, 10), 128, np.uint8)})["percent"] == 0.0


# --- 7. histogram compare --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", vm.HIST_METHOD)
def test_histogram_compare(method):
    ref = textured()
    same = node("HistogramCompare", method=method).process({"image": ref.copy(), "reference": ref})
    assert same["similarity"] > 0.99 and same["ok"]
    other = np.full_like(ref, (200, 50, 50))
    diff = node("HistogramCompare", method=method).process({"image": other, "reference": ref})
    assert diff["similarity"] < 0.5 and not diff["ok"]
    gray = node("HistogramCompare", method=method, space="gri").process(
        {"image": cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), "reference": ref})
    assert gray["similarity"] > 0.99


def test_histogram_compare_roi_and_file(tmp_path):
    ref = np.full((100, 200, 3), 128, np.uint8)
    ref[:, 100:] = (0, 0, 255)
    p = tmp_path / "r.png"
    cv2.imwrite(str(p), ref)
    img = ref.copy()
    img[:, :100] = (255, 0, 0)                     # left half changed
    n = node("HistogramCompare", reference=str(p))
    assert node("HistogramCompare", reference=str(p), roi="100,0,100,100").process(
        {"image": img, "reference": None})["similarity"] > 0.99
    assert n.process({"image": img, "reference": None})["similarity"] < 0.8
    # a reference that is already the cut-out part is used whole
    r = node("HistogramCompare", roi="100,0,100,100").process({"image": img, "reference": ref[:, 100:]})
    assert r["similarity"] > 0.99


# --- 8. auto align ---------------------------------------------------------------------------------------------------

def scene_ref():
    return place(textured(300, 220), 0, 1.0, center=(320, 240))


def shifted(img, dx, dy, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("method", vm.ALIGN_METHOD)
def test_auto_align_recovers_shift_and_rotation(method):
    ref = scene_ref()
    img = shifted(ref, 12.0, -7.0, 3.0)
    r = node("AutoAlign", method=method).process({"image": img, "reference": ref})
    assert r["ok"]
    assert abs(r["dx"] - 12) < 1 and abs(r["dy"] + 7) < 1 and abs(r["angle"] - 3) < 0.3
    inner = (slice(100, 380), slice(120, 520))
    err = np.abs(r["image"][inner].astype(int) - ref[inner].astype(int)).mean()
    raw = np.abs(img[inner].astype(int) - ref[inner].astype(int)).mean()
    assert err < raw / 3


def test_auto_align_large_shift_translation_and_failure():
    ref = scene_ref()
    img = shifted(ref, 60.0, 35.0, 0.0)
    r = node("AutoAlign", motion="sadece kaydırma").process({"image": img, "reference": ref})
    assert r["ok"] and abs(r["dx"] - 60) < 1 and abs(r["dy"] - 35) < 1 and r["angle"] == 0
    assert not node("AutoAlign", max_shift=20).process({"image": img, "reference": ref})["ok"]
    noise = np.random.default_rng(0).integers(0, 255, ref.shape, dtype=np.uint8)
    r = node("AutoAlign", method=vm.ALIGN_METHOD[1]).process({"image": np.full_like(ref, 90), "reference": noise})
    assert not r["ok"] and r["image"].shape == ref.shape
    # gray float input of another size still gives an image of the reference's size
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (320, 240)).astype(np.float32) / 255
    r = node("AutoAlign").process({"image": small, "reference": ref})
    assert r["image"].shape == ref.shape and r["ok"]
    assert abs(r["dx"] - 60) < 2 and abs(r["dy"] - 35) < 2
    # ECC cannot converge on a flat picture: not ok, no crash
    r = node("AutoAlign").process({"image": np.full_like(ref, 90), "reference": ref})
    assert not r["ok"] and r["dx"] == 0


# --- 9. tray grid ----------------------------------------------------------------------------------------------------

def test_grid_inspect_finds_empty_cells():
    img = tray(empty=(2, 7))
    r = node("GridInspect", roi="20,30,240,180", rows=3, cols=4, threshold=200).process({"image": img})
    assert r["empty_cells"] == "2, 7" and r["empty_count"] == 2 and r["filled_count"] == 10 and not r["ok"]
    assert len(r["cells"]) == 12 and r["cells"][1] is False and r["cells"][0] is True
    assert [d["cell"] for d in r["detections"]] == [2, 7]
    assert r["detections"][0]["box"] == (80, 30, 60, 60)
    full = node("GridInspect", roi="20,30,240,180", threshold=200).process({"image": tray(empty=())})
    assert full["ok"] and full["empty_cells"] == "" and full["empty_count"] == 0


def test_grid_inspect_colour_and_light_modes():
    img = tray(empty=(1, 12))
    hsv = dict(h_min=5, h_max=25, s_min=100, s_max=255, v_min=100, v_max=255)   # the orange-ish items
    r = node("GridInspect", roi="20,30,240,180", method="dolu = renk aralığında (HSV)", **hsv).process(
        {"image": img})
    assert r["empty_cells"] == "1, 12"
    inv = node("GridInspect", roi="20,30,240,180", method="dolu = açık", threshold=55).process(
        {"image": 255 - img})
    assert inv["empty_cells"] == "1, 12"
    g16 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257
    assert node("GridInspect", roi="20,30,240,180", threshold=200).process({"image": g16})["empty_cells"] == "1, 12"
    with pytest.raises(ValueError, match="çok küçük"):
        node("GridInspect", roi="0,0,5,5", rows=10, cols=10).process({"image": img})


def test_tray_example_runs():
    path = EXAMPLES / "tepsi_kontrol.json"
    g = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert not g.warnings
    assert not [i for i in check(g, None) if i.level == "hata"]
    g.setup()
    res = g.run_once()
    grid = next(n for n in g.nodes.values() if n.type_name == "GridInspect")
    out = res[grid.id]
    assert out.error is None, out.error
    assert out.outputs["empty_cells"] == "3, 8" and not out.outputs["ok"]


def test_feature_match_rejects_mirror_image():
    obj = textured()
    r = node("FeatureMatch").process({"image": place(cv2.flip(obj, 1), 0, 1.0), "reference": obj})
    assert not r["found"]
