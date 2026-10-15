"""v2.3 (b): 3D / depth vision — depth sources (RealSense, Orbbec, files, synthetic scene), ground plane, height,
volume, 3D objects, flatness, profile, point cloud files and the 3D viewer."""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import types

import cv2
import numpy as np
import pytest

import visionflow.nodes  # noqa: F401
from visionflow import depth3d as D
from visionflow import templates as T
from visionflow.core import StopFlow, registry
from visionflow.core.graph import Graph

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def mk(type_name, **values):
    n = registry.create(type_name)
    n.values.update(values)
    assert n.safe_setup(), n.setup_error
    return n


def run(n, **inputs):
    full = {name: None for name, _ in n.inputs}
    full.update(inputs)
    return n.process(full)


@pytest.fixture(scope="module")
def scene():
    """The full test scene on a tilted floor, with noise and dropped pixels."""
    return D.synthetic_scene(640, 480, 800.0, tilt_x=12.0, tilt_y=-8.0, noise=0.3, dropout=0.02,
                             scene=D.SCENES[1], seed=3)


def angle_between(n1, n2) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(n1, n2))))))


# --- plane, heights, volume, objects ---------------------------------------------------------------------------------

def test_ground_plane_fit_on_tilted_noisy_floor(scene):
    n = mk("FindGroundPlane")
    out = run(n, depth=scene["depth"], intrinsics=scene["intrinsics"])
    plane, truth = out["plane"], scene["plane"]
    assert angle_between(plane["normal"], truth["normal"]) < 0.5
    assert abs(plane["d"] - truth["d"]) < 1.0
    assert abs(out["tilt"] - truth["tilt"]) < 0.5 and 10 < out["tilt"] < 20
    assert out["inliers"] > 70 and out["rms"] < 1.0
    h = out["height"]
    assert h.dtype == np.float32 and h.shape == scene["depth"].shape
    assert np.isnan(h[np.isnan(scene["depth"])]).all()                 # invalid stays invalid
    floor = h[np.isfinite(h) & (np.abs(h) < 5)]
    assert abs(float(np.median(floor))) < 0.3
    assert out["image"].shape == (480, 640, 3) and out["image"].dtype == np.uint8


def test_plane_locked_on_first_frame():
    a = D.synthetic_scene(320, 240, scene=D.SCENES[3])
    b = D.synthetic_scene(320, 240, scene=D.SCENES[3], distance=850.0)
    n = mk("FindGroundPlane", mode="ilk karede sabitle")
    p1 = run(n, depth=a["depth"], intrinsics=a["intrinsics"])["plane"]
    out = run(n, depth=b["depth"], intrinsics=b["intrinsics"])
    assert out["plane"] == p1 and abs(float(np.nanmedian(out["height"])) + 50) < 0.5   # floor 50 mm lower
    assert n.safe_setup()                                               # a new run learns again
    assert abs(run(n, depth=b["depth"], intrinsics=b["intrinsics"])["plane"]["d"] - 850) < 0.5


def test_objects_heights_volumes_and_count(scene):
    plane = run(mk("FindGroundPlane"), depth=scene["depth"], intrinsics=scene["intrinsics"])["plane"]
    out = run(mk("Find3DObjects"), depth=scene["depth"], intrinsics=scene["intrinsics"], plane=plane,
              image=scene["image"])
    solid = [o for o in scene["objects"] if o["kind"] != "hole"]
    assert out["count"] == len(solid) == 4                              # the hole is not an object
    dets = out["detections"]
    for o in solid:                                                     # match by height
        d = min(dets, key=lambda d: abs(d["height"] - o["h"]))
        assert abs(d["height"] - o["h"]) < 1.0, (o, d)
        assert abs(d["volume"] * 1000 - D.object_volume(o)) / D.object_volume(o) < 0.03, (o, d)
        assert set(d) >= {"box", "score", "class_id", "label", "height", "volume", "area_mm2"}
    assert abs(out["max_height"] - 60) < 1.0
    assert out["mask"].shape == scene["depth"].shape and out["image"].shape == scene["image"].shape
    # the detections work with the existing blocks
    drawn = run(mk("DrawDetections"), image=scene["image"], detections=dets)["image"]
    assert drawn.shape == scene["image"].shape
    tall = run(mk("FilterDetections"), detections=[d for d in dets if d["height"] > 45])
    assert tall["count"] == 2


def test_height_and_volume_blocks_in_roi():
    sc = D.synthetic_scene(640, 480, noise=0.3, scene=D.SCENES[2], seed=1)
    plane = run(mk("FindGroundPlane"), depth=sc["depth"], intrinsics=sc["intrinsics"])
    box = sc["objects"][0]
    hm = run(mk("MeasureHeight"), height=plane["height"])
    assert hm["found"] and abs(hm["height"] - box["h"]) < 1.0 and hm["max"] >= hm["height"] >= hm["mean"]
    med = run(mk("MeasureHeight", stat="medyan"), height=plane["height"])
    assert abs(med["height"] - box["h"]) < 0.5
    # a region of bare floor: nothing above it
    empty = run(mk("MeasureHeight", roi="0,0,80,80"), height=plane["height"])
    assert not empty["found"] and empty["height"] == 0.0
    vol = run(mk("MeasureVolume"), depth=sc["depth"], intrinsics=sc["intrinsics"], plane=plane["plane"])
    assert vol["found"] and abs(vol["volume"] * 1000 / D.object_volume(box) - 1) < 0.03
    assert abs(vol["area"] - box["sx"] * box["sy"]) / (box["sx"] * box["sy"]) < 0.03
    # the mask input limits the volume: the left half of the picture holds about half of the box
    mask = np.zeros(sc["depth"].shape, np.uint8)
    mask[:, :320] = 255
    half = run(mk("MeasureVolume"), depth=sc["depth"], intrinsics=sc["intrinsics"], plane=plane["plane"], mask=mask)
    assert 0.3 < half["volume"] / vol["volume"] < 0.7
    none = run(mk("MeasureVolume", roi="0,0,60,60"), depth=sc["depth"], intrinsics=sc["intrinsics"],
               plane=plane["plane"])
    assert none["volume"] == 0.0 and not none["found"]


def test_total_volume_matches_truth(scene):
    plane = run(mk("FindGroundPlane"), depth=scene["depth"], intrinsics=scene["intrinsics"])["plane"]
    vol = run(mk("MeasureVolume"), depth=scene["depth"], intrinsics=scene["intrinsics"], plane=plane)["volume"]
    truth = sum(D.object_volume(o) for o in scene["objects"]) / 1000
    assert abs(vol / truth - 1) < 0.03


def test_hole_is_below_the_plane(scene):
    plane = run(mk("FindGroundPlane"), depth=scene["depth"], intrinsics=scene["intrinsics"])
    h = plane["height"]
    assert np.nanmin(h) < -25                                           # the 30 mm deep hole's bottom


def test_invalid_and_wrong_inputs():
    sc = D.synthetic_scene(160, 120, scene=D.SCENES[2])
    intr = sc["intrinsics"]
    nan = np.full((120, 160), np.nan, np.float32)
    with pytest.raises(ValueError, match="yeterli geçerli"):
        run(mk("FindGroundPlane"), depth=nan, intrinsics=intr)
    with pytest.raises(ValueError, match="bağlı değil"):
        run(mk("FindGroundPlane"), depth=sc["depth"])
    with pytest.raises(ValueError, match="tek kanallı"):
        run(mk("FindGroundPlane"), depth=sc["image"], intrinsics=intr)
    with pytest.raises(ValueError, match="bağlı değil"):
        run(mk("FindGroundPlane"), intrinsics=intr)
    with pytest.raises(ValueError, match="anlaşılmadı"):
        run(mk("MeasureVolume"), depth=sc["depth"], intrinsics=intr, plane="düzlem")
    with pytest.raises(ValueError, match="anlaşılmadı"):
        run(mk("Find3DObjects"), depth=sc["depth"], intrinsics={"fx": 1}, plane=sc["plane"])
    with pytest.raises(ValueError, match="geçersiz"):
        D.intrinsics({"fx": 0, "fy": 1, "cx": 0, "cy": 0})
    with pytest.raises(ValueError, match="Yükseklik girişi"):
        run(mk("MeasureHeight"))
    with pytest.raises(ValueError, match="dışında"):
        run(mk("MeasureVolume", roi="5000,5000,10,10"), depth=sc["depth"], intrinsics=intr, plane=sc["plane"])
    # raw 16-bit depth: 0 means no measurement
    u16 = np.nan_to_num(sc["depth"]).astype(np.uint16)
    u16[:10] = 0
    d = D.as_depth(u16)
    assert d.dtype == np.float32 and np.isnan(d[:10]).all() and np.isfinite(d[20:]).all()
    # all-invalid depth: nothing found, no crash
    out = run(mk("Find3DObjects"), depth=nan, intrinsics=intr, plane=sc["plane"])
    assert out["count"] == 0 and out["detections"] == [] and out["max_height"] == 0.0
    assert run(mk("MeasureVolume"), depth=nan, intrinsics=intr, plane=sc["plane"])["volume"] == 0.0
    # intrinsics of another picture size are scaled
    half = D.intrinsics(intr, (60, 80))
    assert half["fx"] == pytest.approx(intr["fx"] / 2) and half["width"] == 80


def test_heavy_dropout_still_measures():
    sc = D.synthetic_scene(640, 480, noise=0.3, dropout=0.3, scene=D.SCENES[2], seed=5)
    plane = run(mk("FindGroundPlane"), depth=sc["depth"], intrinsics=sc["intrinsics"])
    assert angle_between(plane["plane"]["normal"], sc["plane"]["normal"]) < 0.5
    filt = run(mk("DepthFilter", smooth="yok", fill=100), depth=sc["depth"])
    assert filt["valid"] > 98
    for depth in (sc["depth"], filt["depth"]):                          # with and without the holes filled
        objs = run(mk("Find3DObjects"), depth=depth, intrinsics=sc["intrinsics"], plane=plane["plane"])
        assert objs["count"] == 1
        assert abs(objs["detections"][0]["height"] - 40) < 1.0
        assert abs(objs["detections"][0]["volume"] * 1000 / D.object_volume(sc["objects"][0]) - 1) < 0.03


def test_flatness_and_tilt():
    flat = D.synthetic_scene(320, 240, tilt_x=5.0, noise=0.2, scene=D.SCENES[3], seed=2)
    out = run(mk("Flatness", trim=0.5), depth=flat["depth"], intrinsics=flat["intrinsics"])
    assert out["flatness"] < 2.0 and out["rms"] < 0.3 and abs(out["tilt"] - 5.0) < 0.3
    ref = run(mk("FindGroundPlane"), depth=flat["depth"], intrinsics=flat["intrinsics"])["plane"]
    rel = run(mk("Flatness"), depth=flat["depth"], intrinsics=flat["intrinsics"], plane=ref)
    assert rel["tilt"] < 0.3
    # a 40 mm box on the surface: no longer flat; with the outlier cut the floor is still found
    bumpy = D.synthetic_scene(320, 240, scene=D.SCENES[2])
    b = run(mk("Flatness", trim=0.0), depth=bumpy["depth"], intrinsics=bumpy["intrinsics"])
    assert b["flatness"] > 30
    c = run(mk("Flatness", trim=0.0, outlier=2.0), depth=bumpy["depth"], intrinsics=bumpy["intrinsics"])
    assert abs(c["max_dev"] - 40) < 1.0 and c["tilt"] < 0.2
    with pytest.raises(ValueError, match="yeterli"):
        run(mk("Flatness"), depth=np.full((50, 50), np.nan, np.float32), intrinsics=bumpy["intrinsics"])


def test_depth_profile_across_box():
    sc = D.synthetic_scene(640, 480, scene=D.SCENES[2])
    out = run(mk("DepthProfile", line="20,240,620,240"), depth=sc["depth"], intrinsics=sc["intrinsics"])
    assert abs((out["max"] - out["min"]) - 40) < 0.5                     # floor 800, box top 760
    assert out["image"].shape == (240, 640, 3)
    expected = 600 / sc["intrinsics"]["fx"] * 800                        # both ends on the floor at 800 mm
    assert abs(out["length"] - expected) < 1.0
    assert len(out["values"]) >= 600
    hmap = run(mk("FindGroundPlane"), depth=sc["depth"], intrinsics=sc["intrinsics"])["height"]
    prof = run(mk("DepthProfile", line="20,240,620,240", kind="yükseklik"), depth=hmap)
    assert abs(prof["max"] - 40) < 0.5 and abs(prof["min"]) < 0.5
    with pytest.raises(ValueError, match="dışında"):
        run(mk("DepthProfile", line="0,0,5000,10"), depth=sc["depth"])
    nan = np.full((100, 100), np.nan, np.float32)
    assert run(mk("DepthProfile", line="0,50,99,50"), depth=nan)["values"][0] is None


# --- colouring and filtering -----------------------------------------------------------------------------------------

def test_colorize_depth():
    sc = D.synthetic_scene(320, 240, scene=D.SCENES[2])
    d = sc["depth"].copy()
    d[:20, :20] = np.nan
    out = run(mk("ColorizeDepth", legend=False), depth=d)
    img = out["image"]
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert (img[:20, :20] == 0).all() and img[100:140, 140:180].any()
    assert out["min"] < out["max"]
    fixed = run(mk("ColorizeDepth", vmin=700, vmax=900), depth=d.astype(np.float64))
    assert fixed["min"] == 700 and fixed["max"] == 900
    u16 = run(mk("ColorizeDepth", cmap="Gri"), depth=np.nan_to_num(d).astype(np.uint16))
    assert (u16["image"][:20, :20] == 0).all()
    with pytest.raises(ValueError, match="tek kanallı"):
        run(mk("ColorizeDepth"), depth=sc["image"])
    with pytest.raises(ValueError, match="bağlı değil"):
        run(mk("ColorizeDepth"))


def test_depth_filter_range_median_bilateral_holes():
    rng = np.random.default_rng(0)
    d = np.full((100, 120), 800.0, np.float32) + rng.normal(0, 2, (100, 120)).astype(np.float32)
    d[10:13, 10:13] = np.nan                                             # small hole: filled
    d[50:90, 50:100] = np.nan                                            # big hole: stays
    d[0, :] = 300                                                        # too near: clipped
    out = run(mk("DepthFilter", near=500, far=1000, smooth="yok", fill=20), depth=d)
    o = out["depth"]
    assert np.isfinite(o[10:13, 10:13]).all() and abs(float(o[11, 11]) - 800) < 10
    assert np.isnan(o[60:80, 60:90]).all() and np.isnan(o[0]).all()
    med = run(mk("DepthFilter", smooth="medyan 5", fill=0), depth=d)["depth"]
    assert np.nanstd(med[20:45, 20:45]) < np.nanstd(d[20:45, 20:45]) / 1.5
    assert np.isnan(med[60:80, 60:90]).all()
    bil = run(mk("DepthFilter", smooth="bilateral", sigma=10, fill=0), depth=d)["depth"]
    assert np.nanstd(bil[20:45, 20:45]) < np.nanstd(d[20:45, 20:45])
    assert np.isnan(bil[60:80, 60:90]).all()
    allnan = run(mk("DepthFilter"), depth=np.full((20, 20), np.nan, np.float32))
    assert allnan["valid"] == 0.0


# --- files ----------------------------------------------------------------------------------------------------------

def test_ply_round_trip_binary_and_ascii(tmp_path):
    sc = D.synthetic_scene(160, 120, noise=0.5, dropout=0.05, scene=D.SCENES[1], seed=4)
    pts, (vs, us) = D.depth_to_points(sc["depth"], sc["intrinsics"])
    for binary in (True, False):
        p = D.write_ply(str(tmp_path / f"b{binary}.ply"), pts, sc["image"][vs, us], binary=binary,
                        intr=sc["intrinsics"], size=(160, 120))
        ply = D.read_ply(p)
        assert len(ply["points"]) == len(pts) and ply["size"] == (160, 120)
        assert np.allclose(ply["points"], pts, atol=1e-3)
        assert (ply["colors"] == sc["image"][vs, us]).all()
        depth, color, intr = D.load_depth_file(p)
        assert intr["fx"] == pytest.approx(sc["intrinsics"]["fx"], rel=1e-6)
        same = np.isfinite(sc["depth"])
        assert (np.isfinite(depth) == same).all()
        assert np.allclose(depth[same], sc["depth"][same], atol=1e-2)
        assert (color[same] == sc["image"][same]).all()


def test_foreign_ply_big_endian_with_faces(tmp_path):
    pts = np.array([[0, 0, 1.0], [0.1, 0, 1.0], [0, 0.1, 1.2]], ">f8")
    head = ("ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty double x\nproperty double y\n"
            "property double z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n")
    p = tmp_path / "x.ply"
    p.write_bytes(head.encode() + pts.tobytes() + b"\x03" + np.array([0, 1, 2], ">i4").tobytes())
    ply = D.read_ply(str(p))
    assert np.allclose(ply["points"], pts) and ply["colors"] is None and ply["intrinsics"] is None
    depth, _, intr = D.load_depth_file(str(p), unit="m", size=(64, 48))
    assert depth.shape == (48, 64) and np.nanmax(depth) == pytest.approx(1200) and np.isfinite(depth).sum() == 3
    asc = tmp_path / "a.ply"
    asc.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
                   "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                   "0 0 500 255 0 0\n10 0 500 0 0 255\n", encoding="ascii")
    a = D.read_ply(str(asc))
    assert a["colors"].tolist() == [[0, 0, 255], [255, 0, 0]]          # read as BGR
    for bad, msg in ((b"merhaba", "PLY dosyası değil"),
                     (b"ply\nformat ascii 1.0\nelement vertex 5\nproperty float x\nproperty float y\n"
                      b"property float z\nend_header\n1 2 3\n", "eksik"),
                     (b"ply\nformat ascii 1.0\nelement face 1\nend_header\n", "vertex")):
        (tmp_path / "bad.ply").write_bytes(bad)
        with pytest.raises(ValueError, match=msg):
            D.read_ply(str(tmp_path / "bad.ply"))


def test_png_npy_round_trip_and_errors(tmp_path):
    sc = D.synthetic_scene(160, 120, scene=D.SCENES[2])
    p = D.save_depth_png(str(tmp_path / "ğüş" / "d.png"), sc["depth"], sc["intrinsics"])
    depth, color, intr = D.load_depth_file(p)
    assert np.nanmax(np.abs(depth - sc["depth"])) <= 0.5 and intr["fx"] == pytest.approx(sc["intrinsics"]["fx"])
    assert color is None
    cv2.imencode(".png", sc["image"])[1].tofile(str(tmp_path / "ğüş" / "d_renk.png"))
    assert D.load_depth_file(p)[1].shape == (120, 160, 3)                    # the colour picture next to it
    np.save(tmp_path / "m.npy", sc["depth"] / 1000.0)
    dm, _, im = D.load_depth_file(str(tmp_path / "m.npy"), unit="m", hfov=60)
    assert np.allclose(dm, sc["depth"], atol=1e-2, equal_nan=True) and im["fx"] == pytest.approx(sc["intrinsics"]["fx"])
    xyz = D.depth_to_xyz(sc["depth"], sc["intrinsics"])
    np.save(tmp_path / "xyz.npy", xyz)
    dx, _, ix = D.load_depth_file(str(tmp_path / "xyz.npy"))
    assert np.allclose(dx, sc["depth"], equal_nan=True) and ix["fx"] == pytest.approx(sc["intrinsics"]["fx"], rel=1e-3)
    pts, _ = D.depth_to_points(sc["depth"], sc["intrinsics"])
    np.save(tmp_path / "cloud.npy", pts)
    dc, _, _ = D.load_depth_file(str(tmp_path / "cloud.npy"), size=(160, 120))
    assert np.allclose(dc, sc["depth"], atol=1e-3, equal_nan=True)
    cv2.imwrite(str(tmp_path / "b8.png"), np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="16 bit"):
        D.load_depth_file(str(tmp_path / "b8.png"))
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        (tmp_path / "x.txt").write_text("1")
        D.load_depth_file(str(tmp_path / "x.txt"))
    with pytest.raises(ValueError, match="bulunamadı"):
        D.load_depth_file(str(tmp_path / "yok.png"))
    np.save(tmp_path / "s.npy", np.array(["a"]))
    with pytest.raises(ValueError, match="sayı dizisi"):
        D.load_depth_file(str(tmp_path / "s.npy"))


def test_depth_camera_from_file_and_folder(tmp_path):
    sc = D.synthetic_scene(160, 120, scene=D.SCENES[2])
    single = D.save_depth_png(str(tmp_path / "tek.png"), sc["depth"], sc["intrinsics"])
    cam = mk("DepthCamera", driver="Dosyadan", path=single)
    assert not cam.is_source                                            # one file: like a picture file
    out = run(cam)
    assert out["depth"].shape == (120, 160) and out["image"].shape == (120, 160, 3)
    assert out["intrinsics"]["fx"] == pytest.approx(sc["intrinsics"]["fx"])
    assert D.latest(cam.id)[1] is out["depth"]                          # the viewer can find it
    folder = tmp_path / "seri"
    for i in range(3):
        D.save_depth_png(str(folder / f"{i}.png"), sc["depth"] + i, sc["intrinsics"])
    cv2.imencode(".png", sc["image"])[1].tofile(str(folder / "0_renk.png"))      # not a depth frame
    cam = mk("DepthCamera", driver="Dosyadan", path=str(folder), loop=False)
    assert cam.is_source
    meds = [float(np.nanmedian(run(cam)["depth"])) for _ in range(3)]
    assert meds[1] - meds[0] == pytest.approx(1, abs=0.6) and meds[2] - meds[1] == pytest.approx(1, abs=0.6)
    with pytest.raises(StopFlow):
        run(cam)
    bad = registry.create("DepthCamera")
    bad.values.update(driver="Dosyadan", path="")
    assert not bad.safe_setup() and "seçilmedi" in bad.setup_error
    bad.values.update(path=str(tmp_path / "bos"))
    (tmp_path / "bos").mkdir()
    assert not bad.safe_setup() and "derinlik dosyası yok" in bad.setup_error


def test_save_point_cloud_block_and_read_back(tmp_path):
    sc = D.synthetic_scene(160, 120, scene=D.SCENES[2])
    n = mk("SavePointCloud", folder=str(tmp_path / "out"))
    args = dict(depth=sc["depth"], image=sc["image"], intrinsics=sc["intrinsics"])
    assert run(n, trigger=False, **args)["path"] == ""
    p = run(n, trigger=True, **args)["path"]
    assert p.endswith(".ply") and os.path.isfile(p)
    assert run(n, trigger=True, **args)["path"] == ""                   # only on the rising edge
    cam = mk("DepthCamera", driver="Dosyadan", path=p)
    back = run(cam)
    ok = np.isfinite(sc["depth"])
    assert np.allclose(back["depth"][ok], sc["depth"][ok], atol=1e-2) and (back["image"][ok] == sc["image"][ok]).all()
    png = mk("SavePointCloud", folder=str(tmp_path / "png"), format="PNG 16 bit + JSON", every_n=2)
    assert run(png, **args)["path"] == ""
    pp = run(png, **args)["path"]
    assert pp.endswith(".png") and json.loads(pathlib.Path(pp).with_suffix(".json").read_text())["fx"] > 0
    assert run(mk("SavePointCloud", every_n=1), depth=None)["path"] == ""


# --- depth cameras (fake SDK modules) --------------------------------------------------------------------------------

class _Intr:
    def __init__(self, fx, fy, cx, cy, w, h):
        self.fx, self.fy, self.ppx, self.ppy, self.cx, self.cy, self.width, self.height = fx, fy, cx, cy, cx, cy, w, h


def fake_realsense(no_color=False, timeouts=0):
    rs = types.ModuleType("pyrealsense2")
    rs.log = []
    rs.stream = types.SimpleNamespace(depth="depth", color="color")
    rs.format = types.SimpleNamespace(z16="z16", bgr8="bgr8")
    rs.camera_info = types.SimpleNamespace(serial_number="serial", name="name")
    rs.timeouts = timeouts
    W, H = 64, 48
    raw = np.full((H, W), 800, np.uint16)
    raw[0, 0] = 0
    raw_m = (raw * 1).astype(np.uint16)

    class Dev:
        def __init__(self, sn):
            self.sn = sn

        def get_info(self, what):
            return self.sn if what == "serial" else "Intel RealSense D435"

    class context:
        def query_devices(self):
            return [Dev("111"), Dev("222")]

    class config:
        def __init__(self):
            self.streams, self.device = [], None

        def enable_device(self, sn):
            self.device = sn

        def enable_stream(self, *a):
            self.streams.append(a)

    class Sensor:
        def get_depth_scale(self):
            return 0.001                                               # 1 unit = 1 mm

    class Profile:
        def get_device(self):
            return types.SimpleNamespace(first_depth_sensor=lambda: Sensor())

    class VProfile:
        def __init__(self, aligned):
            self.intrinsics = _Intr(610, 611, 31.5, 23.5, W, H) if aligned else _Intr(380, 381, 32, 24, W, H)

        def as_video_stream_profile(self):
            return self

    class DepthFrame:
        def __init__(self, aligned):
            self.profile = VProfile(aligned)

        def get_data(self):
            return raw_m

        def __bool__(self):
            return True

    class ColorFrame:
        def get_data(self):
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :, 2] = 200
            return img

        def __bool__(self):
            return True

    class Frames:
        def __init__(self, aligned=False, color=True):
            self.aligned, self.color = aligned, color

        def get_depth_frame(self):
            return DepthFrame(self.aligned)

        def get_color_frame(self):
            return ColorFrame() if self.color else None

    class pipeline:
        def __init__(self, ctx=None):
            self.cfg, self.running = None, False
            rs.log.append(("pipeline", self))

        def start(self, cfg):
            if no_color and any(s[0] == "color" for s in cfg.streams):
                raise RuntimeError("Couldn't resolve requests")
            self.cfg, self.running = cfg, True
            rs.log.append(("start", cfg.device, len(cfg.streams)))
            return Profile()

        def wait_for_frames(self, timeout):
            if rs.timeouts:
                rs.timeouts -= 1
                raise RuntimeError("Frame didn't arrive within 1000")
            return Frames(color=len(self.cfg.streams) > 1)

        def stop(self):
            self.running = False
            rs.log.append(("stop", self.cfg.device))

    class align:
        def __init__(self, to):
            rs.log.append(("align", to))

        def process(self, frames):
            return Frames(aligned=True, color=frames.color)

    rs.context, rs.config, rs.pipeline, rs.align = context, config, pipeline, align
    return rs


def test_realsense_driver_alignment_intrinsics_sharing_teardown(monkeypatch):
    rs = fake_realsense()
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    a = mk("DepthCamera", driver="Intel RealSense", serial="222", width=64, height=48)
    b = mk("DepthCamera", driver="Intel RealSense", serial="222", width=64, height=48)
    assert [e for e in rs.log if e[0] == "start"] == [("start", "222", 2)]      # one device for both blocks
    assert ("align", "color") in rs.log
    out = run(a)
    assert out["depth"].dtype == np.float32 and out["depth"][1, 1] == pytest.approx(800.0)
    assert np.isnan(out["depth"][0, 0])                                 # 0 = no measurement
    assert out["intrinsics"]["fx"] == 610 and out["intrinsics"]["cx"] == 31.5    # colour camera's (aligned)
    assert out["image"].shape == (48, 64, 3) and out["image"][0, 0, 2] == 200
    a.teardown()
    assert not [e for e in rs.log if e[0] == "stop"]                    # b still uses it
    run(b)
    b.teardown()
    assert [e for e in rs.log if e[0] == "stop"] == [("stop", "222")]
    assert D.SharedDevice.open_count() == 0
    # without alignment: the depth camera's own intrinsics
    rs2 = fake_realsense()
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs2)
    c = mk("DepthCamera", driver="Intel RealSense", align=False)
    assert not [e for e in rs2.log if e[0] == "align"]
    assert run(c)["intrinsics"]["fx"] == 380 and [e for e in rs2.log if e[0] == "start"][0][1] == "111"
    c.teardown()
    bad = registry.create("DepthCamera")
    bad.values.update(driver="Intel RealSense", serial="999")
    assert not bad.safe_setup() and "999" in bad.setup_error and "111, 222" in bad.setup_error
    assert D.SharedDevice.open_count() == 0


def test_realsense_without_colour_and_timeouts(monkeypatch):
    rs = fake_realsense(no_color=True, timeouts=1)
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    cam = mk("DepthCamera", driver="Intel RealSense")
    assert [e for e in rs.log if e[0] == "start"] == [("start", "111", 1)]      # depth only
    with pytest.raises(RuntimeError, match="kare gelmedi"):
        run(cam)
    out = run(cam)
    assert out["image"].shape == (48, 64, 3)                            # coloured depth instead of a picture
    assert out["intrinsics"]["fx"] == 380                               # nothing to align to
    rs.timeouts = 3
    for _ in range(3):
        with pytest.raises(RuntimeError):
            run(cam)
    assert cam.state["dev"] is None and [e for e in rs.log if e[0] == "stop"]    # dropped after 3 misses
    assert run(cam)["depth"].shape == (48, 64)                          # and reconnected
    cam.teardown()
    assert D.SharedDevice.open_count() == 0


def test_missing_sdk_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    for driver, pkg in (("Intel RealSense", "pyrealsense2"), ("Orbbec", "pyorbbecsdk")):
        cam = registry.create("DepthCamera")
        cam.values["driver"] = driver
        assert not cam.safe_setup() and f"pip install {pkg}" in cam.setup_error
        cam.teardown()
    assert D.SharedDevice.open_count() == 0


def fake_orbbec(color_fmt="RGB"):
    ob = types.ModuleType("pyorbbecsdk")
    ob.log = []
    W, H = 32, 24
    ob.OBSensorType = types.SimpleNamespace(DEPTH_SENSOR="depth", COLOR_SENSOR="color")
    ob.OBFormat = types.SimpleNamespace(Y16="Y16", RGB="RGB", BGR="BGR", MJPG="MJPG", YUYV="YUYV")
    ob.OBAlignMode = types.SimpleNamespace(SW_MODE="sw", HW_MODE="hw")

    class DevList:
        def get_count(self):
            return 1

        def get_device_serial_number_by_index(self, i):
            return "AY123"

        def get_device_by_index(self, i):
            return "dev0"

    class Context:
        def query_devices(self):
            return DevList()

    class PList:
        def __init__(self, sensor):
            self.sensor = sensor

        def get_video_stream_profile(self, w, h, fmt, fps):
            raise RuntimeError("no such profile")

        def get_default_video_stream_profile(self):
            return ("profile", self.sensor)

    class Config:
        def __init__(self):
            self.streams, self.align = [], None
            ob.log.append(("config", self))

        def enable_stream(self, p):
            self.streams.append(p)

        def set_align_mode(self, m):
            self.align = m

    class DFrame:
        def get_width(self):
            return W

        def get_height(self):
            return H

        def get_data(self):
            d = np.full((H, W), 5000, np.uint16)
            d[0, 0] = 0
            return d.tobytes()

        def get_depth_scale(self):
            return 0.1                                                  # 0.1 mm per unit

    class CFrame:
        def get_width(self):
            return W

        def get_height(self):
            return H

        def get_format(self):
            return color_fmt

        def get_data(self):
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :, 0] = 250                                          # red in RGB
            if color_fmt == "MJPG":
                return cv2.imencode(".jpg", img[:, :, ::-1].copy())[1].tobytes()
            return img.tobytes()

    class Frames:
        def get_depth_frame(self):
            return DFrame()

        def get_color_frame(self):
            return CFrame()

    class Param:
        rgb_intrinsic = _Intr(500, 501, 15.5, 11.5, W, H)
        depth_intrinsic = _Intr(300, 301, 16, 12, W, H)

    class Pipeline:
        def __init__(self, dev):
            self.dev = dev
            ob.log.append(("pipeline", dev))

        def get_stream_profile_list(self, sensor):
            return PList(sensor)

        def start(self, cfg):
            ob.log.append(("start", tuple(cfg.streams), cfg.align))

        def wait_for_frames(self, t):
            return Frames()

        def get_camera_param(self):
            return Param()

        def stop(self):
            ob.log.append(("stop",))

    ob.Context, ob.Config, ob.Pipeline = Context, Config, Pipeline
    return ob


@pytest.mark.parametrize("fmt", ["RGB", "MJPG"])
def test_orbbec_driver(monkeypatch, fmt):
    ob = fake_orbbec(fmt)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", ob)
    cam = mk("DepthCamera", driver="Orbbec", width=32, height=24)
    start = [e for e in ob.log if e[0] == "start"][0]
    assert start[1] == (("profile", "depth"), ("profile", "color")) and start[2] == "sw"
    out = run(cam)
    assert out["depth"][1, 1] == pytest.approx(500.0) and np.isnan(out["depth"][0, 0])
    assert out["intrinsics"]["fx"] == 500                               # aligned to colour
    assert out["image"][5, 5, 2] > 200 and out["image"][5, 5, 0] < 40   # RGB → BGR
    cam.teardown()
    assert ("stop",) in ob.log and D.SharedDevice.open_count() == 0
    cam2 = mk("DepthCamera", driver="Orbbec", align=False)
    assert run(cam2)["intrinsics"]["fx"] == 300
    cam2.teardown()


# --- synthetic scene block, viewer, example ------------------------------------------------------------------------------

def test_test_scene_block_varies_and_caches():
    n = mk("DepthTestScene", width=160, height=120, scene="tek kutu")
    assert not n.is_source
    a, b = run(n), run(n)
    assert np.array_equal(a["depth"], b["depth"], equal_nan=True) and a["truth"][0]["h"] == 40
    a["depth"][:] = 0                                                    # outputs are copies
    assert np.nanmax(run(n)["depth"]) > 700
    v = mk("DepthTestScene", width=160, height=120, scene="tek kutu", animate=True, vary=5)
    assert v.is_source
    hs = {run(v)["truth"][0]["h"] for _ in range(4)}
    assert len(hs) == 4 and all(35 <= h <= 45 for h in hs)


def test_render_cloud():
    sc = D.synthetic_scene(160, 120, scene=D.SCENES[1])
    pts, cols = D.cloud_for_view(sc["depth"], sc["intrinsics"], sc["image"], sc["plane"])
    a = D.render_cloud(pts, cols, (200, 150), yaw=0, pitch=0)
    b = D.render_cloud(pts, cols, (200, 150), yaw=60, pitch=40)
    assert a.shape == (150, 200, 3) and not np.array_equal(a, b)
    assert len(np.unique(a.reshape(-1, 3), axis=0)) > 20
    empty = D.render_cloud(np.zeros((0, 3)), None, (50, 40))
    assert (empty == empty[0, 0]).all()
    cam = D.cloud_for_view(sc["depth"], sc["intrinsics"], sc["image"], None, "kamera rengi", max_points=500)
    assert len(cam[0]) <= 600 and cam[1].dtype == np.uint8
    out = run(mk("PointCloudView", width=120, height=90), depth=sc["depth"], intrinsics=sc["intrinsics"],
              plane=sc["plane"], image=sc["image"])
    assert out["image"].shape == (90, 120, 3)
    with pytest.raises(ValueError, match="anlaşılmadı"):
        run(mk("PointCloudView"), depth=sc["depth"], intrinsics=sc["intrinsics"], plane=[1, 2])


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_viewer_window_offscreen(qapp, tmp_path):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from visionflow.ui.depth_viewer import DepthViewerDialog
    node = registry.create("DepthTestScene")
    node.values.update(width=160, height=120, scene="tek kutu")
    D.forget(node.id)
    dlg = DepthViewerDialog(None, node, lambda *a: True)
    try:
        dlg.resize(500, 400)
        dlg.show()
        qapp.processEvents()
        assert not dlg.refresh() and "Henüz" in dlg.status.text()
        assert dlg.grab_once() and len(dlg.view.points) > 1000
        img = dlg.view.grab().toImage()
        assert img.width() == dlg.view.width()
        colours = {img.pixel(x, y) for x in range(0, img.width(), 7) for y in range(0, img.height(), 7)}
        assert len(colours) > 10                                        # a cloud was drawn, not a blank
        yaw = dlg.view.yaw
        c = QPointF(100, 100)
        dlg.view.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, c, c, Qt.LeftButton, Qt.LeftButton,
                                             Qt.NoModifier))
        m = QPointF(160, 120)
        dlg.view.mouseMoveEvent(QMouseEvent(QMouseEvent.MouseMove, m, m, Qt.LeftButton, Qt.LeftButton,
                                            Qt.NoModifier))
        dlg.view.mouseReleaseEvent(QMouseEvent(QMouseEvent.MouseButtonRelease, m, m, Qt.LeftButton, Qt.NoButton,
                                               Qt.NoModifier))
        assert dlg.view.yaw != yaw
        z = dlg.view.zoom
        dlg.view.wheelEvent(QWheelEvent(c, c, QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                                        Qt.NoScrollPhase, False))
        assert dlg.view.zoom > z
        img2 = dlg.view.grab().toImage()
        assert img2 != img
        dlg.view.mouseDoubleClickEvent(None)
        assert dlg.view.yaw == -30.0 and dlg.view.zoom == 1.0
        p = dlg.save_ply(str(tmp_path / "v.ply"))
        assert len(D.read_ply(p)["points"]) == len(np.flatnonzero(np.isfinite(D.latest(node.id)[1])))
        # live: a newer frame of the block is picked up by the timer
        n0 = dlg._shown_n
        sc = D.synthetic_scene(80, 60, scene=D.SCENES[3])
        D.publish(node.id, sc["depth"], None, sc["intrinsics"])
        dlg._tick()
        assert dlg._shown_n == n0 + 1 and len(dlg.view.points) <= 80 * 60
        for mode in range(3):
            dlg.mode.setCurrentIndex(mode)
        # a block that cannot produce frames by itself
        view_node = registry.create("PointCloudView")
        other = DepthViewerDialog(None, view_node, lambda *a: True)
        assert not other.grab_once() and "akışı" in other.status.text()
        other.close()
    finally:
        dlg.close()
        qapp.processEvents()


def test_viewer_opens_from_block_button(qapp):
    from visionflow.ui.main_window import MainWindow
    from PySide6.QtWidgets import QPushButton
    w = MainWindow()
    w._confirm_discard = lambda: True
    try:
        g = Graph()
        n = g.add_new("DepthCamera")
        w.scene.load_graph(g)
        w._loaded(None)
        w.props.show_node(w.scene.graph.nodes[n.id])
        btn = w.props.findChild(QPushButton, "block_dialog")
        assert btn is not None and "3B" in btn.text()
        dlg = w.open_block_dialog(n.id)
        assert type(dlg).__name__ == "DepthViewerDialog" and dlg.node.id == n.id
        dlg.close()
    finally:
        w.stop()
        w.close()
        qapp.processEvents()


def test_example_template_runs_with_ok_and_nok():
    t = T.load_template(ROOT / "examples" / "derinlik_hacim_olcum.json")
    assert t["new"] and set(t["categories"]) <= set(T.CATEGORIES) and t["questions"]
    assert "derinlik_hacim_olcum" in [x["name"] for x in T.list_templates() if T.matches(x, "Ölçüm", "hacim")]
    from visionflow.check import check
    g = T.apply_answers(t, {"yukseklik": 40, "tol": 1.5, "hacim": 280, "hacim_tol": 3})
    assert not [i for i in check(g) if i.level == "hata"]
    assert g.nodes["yukseklik_tol"].values["low"] == pytest.approx(38.5)
    assert g.nodes["hacim_tol"].values["high"] == pytest.approx(288.4)
    rows = []

    def on_frame(n, res):
        errors = [(k, r.error) for k, r in res.items() if r.error]
        assert not errors, errors
        truth = res["sahne"].outputs["truth"][0]
        det = res["nesne"].outputs["detections"]
        rows.append((truth["h"], truth["volume"], det[0]["height"], res["hacim"].outputs["volume"],
                     res["sonuc_ve"].outputs["result"]))
    g.run(max_frames=6, on_frame=on_frame)
    assert len(rows) == 6
    for th, tv, mh, mv, ok in rows:
        assert abs(mh - th) < 1.0 and abs(mv / tv - 1) < 0.03
    assert {r[4] for r in rows} == {True, False}                        # some parts good, some not
    data = json.loads((ROOT / "examples" / "derinlik_hacim_olcum.json").read_text(encoding="utf-8"))
    assert not Graph.from_dict(data).warnings


def test_blocks_registered_with_category():
    from visionflow.ui.theme import CATEGORY_COLORS
    for name in ("DepthCamera", "DepthTestScene", "ColorizeDepth", "DepthFilter", "FindGroundPlane",
                 "MeasureHeight", "MeasureVolume", "Find3DObjects", "Flatness", "DepthProfile", "SavePointCloud",
                 "PointCloudView"):
        cls = registry.NODE_REGISTRY[name]
        assert cls.title and cls.description and cls.category in CATEGORY_COLORS
