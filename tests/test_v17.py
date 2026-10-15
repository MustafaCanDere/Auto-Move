"""1.7: YOLO pose, YOLO OBB (rotated boxes), joint angle and PatchCore deep anomaly detection."""
import math
import sys

import cv2
import numpy as np
import pytest

from visionflow.core import create, load_builtin
from visionflow.core.registry import NODE_REGISTRY
from visionflow.nodes import patchcore as pc
from visionflow.nodes.ai_more import KPT_NAMES, kpt_shape, obb_postprocess, pose_postprocess

load_builtin()
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


def const_model(path, outputs, meta=None):
    """An ONNX model that ignores its 640x640 input and returns fixed tensors (enough to test the decoding)."""
    nodes = [helper.make_node("Constant", [], [name], value=helper.make_tensor(
        name + "_v", TensorProto.FLOAT, arr.shape, arr.flatten().tolist())) for name, arr in outputs.items()]
    graph = helper.make_graph(nodes, "fake",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info(n, TensorProto.FLOAT, a.shape) for n, a in outputs.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for k, v in (meta or {}).items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, path)
    return str(path)


def test_blocks_registered():
    for t in ("YoloPose", "YoloOBB", "PoseAngle", "PatchCore"):
        assert t in NODE_REGISTRY
    assert len(NODE_REGISTRY) >= 130


# --- pose ----------------------------------------------------------------------------------------------------------

def pose_tensor(n=8400):
    raw = np.zeros((1, 56, n), np.float32)
    raw[0, :5, 0] = (320, 320, 100, 200, 0.9)                 # one person in the middle
    raw[0, :5, 1] = (322, 321, 100, 200, 0.6)                 # the same person again: removed by NMS
    for j in range(17):
        raw[0, 5 + 3 * j:8 + 3 * j, 0] = (300 + j, 250 + 5 * j, 0.9 if j != 4 else 0.1)
    return raw


def test_pose_decoding():
    dets = pose_postprocess(pose_tensor(), 0.25, 0.45, 0.5, (0, 140), ["person"])
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "person" and d["box"] == (540, 160, 200, 400)
    assert len(d["keypoints"]) == 17
    assert d["keypoints"][0] == (600.0, 220.0, pytest.approx(0.9))   # (300 - 0) / 0.5, (250 - 140) / 0.5
    assert d["keypoints"][4][2] == pytest.approx(0.1)


def test_pose_end_to_end_and_bad_shape():
    raw = np.zeros((1, 300, 57), np.float32)
    raw[0, 0, :6] = (100, 100, 200, 300, 0.8, 0)
    raw[0, 0, 6:] = 50
    dets = pose_postprocess(raw, 0.25, 0.45, 1.0, (0, 0), ["person"])
    assert len(dets) == 1 and dets[0]["box"] == (100, 100, 100, 200) and dets[0]["keypoints"][3][:2] == (50.0, 50.0)
    with pytest.raises(ValueError, match="poz modeli değil"):
        pose_postprocess(np.zeros((1, 84, 100), np.float32), 0.25, 0.45, 1.0, (0, 0), ["person"])


def test_kpt_shape_from_metadata():
    assert kpt_shape("[17, 3]") == (17, 3) and kpt_shape("[4, 2]") == (4, 2) and kpt_shape(None) == (17, 3)


def test_pose_block_with_onnx(tmp_path):
    path = const_model(tmp_path / "pose.onnx", {"output0": pose_tensor()}, {"kpt_shape": "[17, 3]",
                                                                             "names": "{0: 'person'}"})
    n = create("YoloPose")
    n.values.update(model=path)
    assert n.safe_setup(), n.setup_error
    out = n.process({"image": np.zeros((640, 640, 3), np.uint8)})
    assert out["count"] == 1 and out["detections"][0]["keypoints"][0][:2] == (300.0, 250.0)
    drawn = create("DrawDetections").process({"image": np.zeros((640, 640, 3), np.uint8),
                                              "detections": out["detections"]})["image"]
    assert drawn[250, 300].any()                                # a joint dot is drawn
    assert drawn[260 - 1:260 + 2, 270:370].any()               # and the box


def test_custom_keypoint_model(tmp_path):
    raw = np.zeros((1, 4 + 1 + 4 * 2, 100), np.float32)        # 4 keypoints of (x, y): no visibility value
    raw[0, :5, 0] = (100, 100, 40, 40, 0.9)
    raw[0, 5:, 0] = (90, 90, 110, 90, 110, 110, 90, 110)
    path = const_model(tmp_path / "kp.onnx", {"output0": raw}, {"kpt_shape": "[4, 2]", "names": "{0: 'kutu'}"})
    n = create("YoloPose")
    n.values.update(model=path)
    assert n.safe_setup()
    d = n.process({"image": np.zeros((640, 640, 3), np.uint8)})["detections"][0]
    assert d["label"] == "kutu" and [k[:2] for k in d["keypoints"]] == [(90, 90), (110, 90), (110, 110), (90, 110)]


def joints(**pts):
    kp = [(0.0, 0.0, 0.0)] * 17
    for name, (x, y) in pts.items():
        kp[KPT_NAMES.index(name)] = (x, y, 0.9)
    return kp


def test_pose_angle():
    n = create("PoseAngle")
    straight = joints(sağ_omuz=(0, 0), sağ_dirsek=(0, 10), sağ_bilek=(0, 20))
    bent = joints(sağ_omuz=(0, 0), sağ_dirsek=(0, 10), sağ_bilek=(10, 10))
    dets = [{"score": 0.5, "keypoints": straight}, {"score": 0.9, "keypoints": bent}]
    assert n.process({"detections": dets}) == {"angle": 90.0, "found": True}     # the surest person
    assert n.process({"detections": dets[:1]})["angle"] == 180.0
    hidden = joints(sağ_omuz=(0, 0), sağ_dirsek=(0, 10))
    assert n.process({"detections": [{"score": 1, "keypoints": hidden}]})["found"] is False
    assert n.process({"detections": None}) == {"angle": 0.0, "found": False}
    assert n.process({"detections": [{"score": 1, "box": (0, 0, 1, 1)}]})["found"] is False


# --- OBB -----------------------------------------------------------------------------------------------------------

def obb_tensor():
    raw = np.zeros((1, 4 + 2 + 1, 50), np.float32)
    raw[0, :4, 0], raw[0, 4, 0], raw[0, 6, 0] = (200, 200, 120, 40), 0.9, math.radians(30)
    raw[0, :4, 1], raw[0, 4, 1], raw[0, 6, 1] = (202, 201, 118, 40), 0.7, math.radians(31)   # duplicate
    raw[0, :4, 2], raw[0, 5, 2], raw[0, 6, 2] = (200, 200, 120, 40), 0.8, math.radians(30)   # same place, class 1
    raw[0, :4, 3], raw[0, 4, 3], raw[0, 6, 3] = (400, 400, 50, 50), 0.1, 0.0                 # too weak
    return raw


def test_obb_decoding():
    dets = obb_postprocess(obb_tensor(), 0.25, 0.45, 1.0, (0, 0), ["vida", "somun"])
    assert sorted(d["label"] for d in dets) == ["somun", "vida"]      # other classes are never suppressed
    d = next(d for d in dets if d["label"] == "vida")
    assert d["angle"] == pytest.approx(30.0, abs=0.01) and d["center"] == (200.0, 200.0)
    poly = np.array(d["poly"])
    assert poly.shape == (4, 2)
    assert np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[1] - poly[2]) == pytest.approx(160, abs=0.5)
    x, y, w, h = d["box"]
    assert x <= poly[:, 0].min() + 1 and x + w >= poly[:, 0].max() - 1     # the axis box holds the rotated one
    with pytest.raises(ValueError, match="OBB"):
        obb_postprocess(np.zeros((1, 5, 10), np.float32), 0.25, 0.45, 1.0, (0, 0), [])


def test_obb_block_and_drawing(tmp_path):
    path = const_model(tmp_path / "obb.onnx", {"output0": obb_tensor()}, {"names": "{0: 'vida', 1: 'somun'}"})
    n = create("YoloOBB")
    n.values.update(model=path)
    assert n.safe_setup(), n.setup_error
    out = n.process({"image": np.zeros((640, 640, 3), np.uint8)})
    assert out["count"] == 2 and out["angle"] == pytest.approx(30.0, abs=0.01)
    img = create("DrawDetections").process({"image": np.zeros((640, 640, 3), np.uint8),
                                            "detections": out["detections"]})["image"]
    p = np.round(np.array(out["detections"][0]["poly"][0])).astype(int)
    assert img[p[1] - 2:p[1] + 3, p[0] - 2:p[0] + 3].any()             # the rotated outline is drawn
    x, y, w, h = out["detections"][0]["box"]
    assert not img[y + h // 2, x].any()                                # not the axis-aligned box
    empty = n.process({"image": np.zeros((10, 10), np.uint8)})        # grey picture works too
    assert empty["count"] == 2


# --- PatchCore -----------------------------------------------------------------------------------------------------

def part(rng, defect=None):
    img = np.full((240, 240, 3), (60, 90, 120), np.uint8)
    for y in range(0, 240, 12):
        cv2.line(img, (0, y), (239, y), (80, 110, 140), 2)
    cv2.circle(img, (120, 120), 50, (180, 180, 180), -1)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if defect == "scratch":
        cv2.line(img, (80, 50), (170, 95), (220, 220, 220), 2)
    if defect == "stain":
        cv2.circle(img, (200, 200), 9, (20, 40, 60), -1)
    return img


@pytest.fixture
def good(tmp_path):
    rng = np.random.default_rng(3)
    d = tmp_path / "saglam"
    d.mkdir()
    for i in range(8):
        cv2.imwrite(str(d / f"{i}.png"), part(rng))
    return d


def test_patchcore_finds_defects(good, capsys):
    n = create("PatchCore")
    n.values.update(folder=str(good), size=192)
    assert n.safe_setup(), n.setup_error
    rng = np.random.default_rng(9)
    goods = [n.process({"image": part(rng)}) for _ in range(4)]
    assert all(g["ok"] for g in goods) and max(g["score"] for g in goods) < 1.0
    for kind, (cx, cy) in (("scratch", (125, 72)), ("stain", (200, 200))):
        out = n.process({"image": part(rng, kind)})
        assert not out["ok"] and out["score"] > 1.5, kind
        assert any(x - 5 <= cx <= x + w + 5 and y - 5 <= cy <= y + h + 5
                   for x, y, w, h in (d["box"] for d in out["detections"])), kind     # found where it is
        assert out["mask"].shape == (240, 240) and out["heatmap"].shape == (240, 240, 3)
    assert "eğitildi" in capsys.readouterr().out


def test_patchcore_reuses_training_until_images_change(good, capsys):
    n = create("PatchCore")
    n.values.update(folder=str(good), size=128)
    assert n.safe_setup()
    capsys.readouterr()
    n2 = create("PatchCore")
    n2.values.update(folder=str(good), size=128)
    assert n2.safe_setup()
    assert "kayıtlı eğitim" in capsys.readouterr().out
    assert len(list((good / "_patchcore").glob("*.npz"))) == 1
    cv2.imwrite(str(good / "yeni.png"), part(np.random.default_rng(5)))
    n3 = create("PatchCore")
    n3.values.update(folder=str(good), size=128)
    assert n3.safe_setup()
    assert "eğitiliyor" in capsys.readouterr().out and n3.state["model"].n == 9
    for size in (160, 192, 224, 256):                                   # other settings: kept side by side…
        m = create("PatchCore")
        m.values.update(folder=str(good), size=size)
        assert m.safe_setup()
    assert len(list((good / "_patchcore").glob("*.npz"))) == 4          # …but only the last few
    capsys.readouterr()
    m = create("PatchCore")
    m.values.update(folder=str(good), size=256)
    assert m.safe_setup() and "kayıtlı eğitim" in capsys.readouterr().out


def test_patchcore_sensitivity_and_too_few_images(good, tmp_path):
    n = create("PatchCore")
    n.values.update(folder=str(good), size=128)
    assert n.safe_setup()
    stain = part(np.random.default_rng(1), "stain")
    out = n.process({"image": stain})
    assert not out["ok"]
    n.values["sensitivity"] = out["score"] + 0.1                       # a looser limit lets the same part pass
    assert n.process({"image": stain})["ok"]
    one = tmp_path / "tek"
    one.mkdir()
    cv2.imwrite(str(one / "a.png"), part(np.random.default_rng(1)))
    n = create("PatchCore")
    n.values.update(folder=str(one))
    assert not n.safe_setup() and "en az 2" in n.setup_error


def feature_model(path):
    """A tiny CNN in ONNX with two feature maps (strides 4 and 8), like a backbone's layer2/layer3."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.3, (8, 3, 3, 3)).astype(np.float32)
    w2 = rng.normal(0, 0.3, (16, 8, 3, 3)).astype(np.float32)
    inits = [helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist()),
             helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist())]
    nodes = [helper.make_node("Conv", ["image", "w1"], ["c1"], strides=[4, 4], pads=[1, 1, 1, 1]),
             helper.make_node("Relu", ["c1"], ["f1"]),
             helper.make_node("Conv", ["f1", "w2"], ["c2"], strides=[2, 2], pads=[1, 1, 1, 1]),
             helper.make_node("Relu", ["c2"], ["f2"])]
    graph = helper.make_graph(nodes, "feat", [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                                                            [1, 3, "h", "w"])],
                              [helper.make_tensor_value_info("f1", TensorProto.FLOAT, None),
                               helper.make_tensor_value_info("f2", TensorProto.FLOAT, None)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return str(path)


def test_patchcore_with_own_onnx_backbone(good, tmp_path):
    n = create("PatchCore")
    n.values.update(folder=str(good), backbone=pc.BACKBONES[2], onnx=feature_model(tmp_path / "f.onnx"), size=128)
    assert n.safe_setup(), n.setup_error
    assert n.state["model"].bank.shape[1] == 24                          # 8 + 16 channels stacked
    rng = np.random.default_rng(4)
    assert n.process({"image": part(rng, "stain")})["score"] > n.process({"image": part(rng)})["score"]
    n = create("PatchCore")
    n.values.update(folder=str(good), backbone=pc.BACKBONES[2])
    assert not n.safe_setup() and "seçilmedi" in n.setup_error


def test_resnet_needs_torch_once(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="pip install torch torchvision"):
        pc.ensure_resnet()
    (tmp_path / "modeller").mkdir()
    (tmp_path / "modeller" / pc.RESNET_FILE).write_bytes(b"x")
    assert pc.ensure_resnet().endswith(pc.RESNET_FILE)                    # made once: torch not needed again


def test_resnet_export(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))
    real = torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, "resnet18", lambda weights=None, **k: real(weights=None, **k))  # offline
    path = pc.ensure_resnet()
    f = pc.OnnxFeatures(path, 224)(np.zeros((100, 150, 3), np.uint8))
    assert f.shape == (28, 28, 128 + 256)
    assert pc.OnnxFeatures(path, 256)(np.zeros((50, 50), np.uint8)).shape == (32, 32, 384)


def test_coreset_covers_the_data():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.1, (200, 8)) for c in (0, 5, 10)]).astype(np.float32)
    idx = pc.coreset(x, 3)
    assert sorted(set(np.round(x[idx].mean(1) / 5).astype(int))) == [0, 1, 2]   # one from each cluster
    assert len(pc.coreset(x, 10_000)) == len(x)
    d = pc.nearest(x[:5], x, np.zeros(5, int), np.array([0] * 5 + [1] * (len(x) - 5)))
    assert (d > 0).all()                                                  # own rows are skipped


# --- fixes from the pre-release bug hunt ------------------------------------------------------------------------

def test_obb_end_to_end_yolo26():
    raw = np.zeros((1, 300, 7), np.float32)
    raw[0, 0] = (200, 200, 120, 40, 0.9, 1, math.radians(30))
    raw[0, 1] = (100, 100, 20, 20, 0.1, 0, 0)                             # below the limit
    dets = obb_postprocess(raw, 0.25, 0.45, 1.0, (0, 0), ["vida", "somun"])
    assert len(dets) == 1 and dets[0]["label"] == "somun" and dets[0]["angle"] == pytest.approx(30, abs=0.01)


def test_rotated_nms_matches_ultralytics_rules():
    from visionflow.nodes.ai_more import probiou
    b = np.array([[0, 0, 100, 20, 0], [0, 0, 100, 20, 0], [0, 0, 100, 20, math.pi / 2], [500, 500, 10, 10, 0]],
                 np.float64)
    iou = probiou(b, b)
    assert iou[0, 1] == pytest.approx(1, abs=1e-3) and iou[0, 3] < 1e-3 and 0.05 < iou[0, 2] < 0.6
    raw = np.zeros((1, 6, 3), np.float32)
    raw[0, :, 0] = (200, 200, 100, 20, 0.9, 0)
    raw[0, :, 1] = (200, 200, 100, 20, 0.8, math.pi / 2)                # a cross: kept at iou 0.7
    raw[0, :, 2] = (201, 200, 100, 20, 0.7, 0)                          # a copy: removed
    assert len(obb_postprocess(raw, 0.25, 0.7, 1.0, (0, 0), ["x"])) == 2


@pytest.mark.parametrize("img", [np.zeros((240, 240, 4), np.uint8), np.zeros((240, 240), np.uint16) + 3000,
                                 np.random.default_rng(0).random((240, 240, 3)).astype(np.float32),
                                 np.zeros((240, 240, 1), np.uint8)])
def test_odd_pictures_do_not_crash(good, img, tmp_path):
    n = create("PatchCore")
    n.values.update(folder=str(good), size=128)
    assert n.safe_setup()
    out = n.process({"image": img})
    assert out["heatmap"].shape == (240, 240, 3)
    path = const_model(tmp_path / "pose.onnx", {"output0": pose_tensor()}, {"kpt_shape": "[17, 3]"})
    y = create("YoloPose")
    y.values.update(model=path)
    assert y.safe_setup()
    assert y.process({"image": img})["count"] == 1


def test_broken_or_changed_cache_retrains(good, tmp_path, capsys):
    import os
    import time
    n = create("PatchCore")
    n.values.update(folder=str(good), size=128)
    assert n.safe_setup()
    cache = next((good / "_patchcore").glob("*.npz"))
    cache.write_bytes(cache.read_bytes()[:50])                            # cut off by a power cut
    capsys.readouterr()
    n = create("PatchCore")
    n.values.update(folder=str(good), size=128)
    assert n.safe_setup(), n.setup_error
    assert "eğitiliyor" in capsys.readouterr().out
    model = feature_model(tmp_path / "f.onnx")
    m = create("PatchCore")
    m.values.update(folder=str(good), backbone=pc.BACKBONES[2], onnx=model, size=128)
    assert m.safe_setup()
    later = time.time() + 5
    os.utime(model, (later, later))                                     # retrained model, same name and size
    capsys.readouterr()
    m = create("PatchCore")
    m.values.update(folder=str(good), backbone=pc.BACKBONES[2], onnx=model, size=128)
    assert m.safe_setup() and "eğitiliyor" in capsys.readouterr().out


def test_fit_samples_each_image():
    feats = pc.BuiltinFeatures(128)
    model = pc.PatchCoreModel(feats, max_patches=300)
    rng = np.random.default_rng(2)
    model.fit([part(rng) for _ in range(6)])
    assert len(model.bank) <= 300 and set(model.bank_src) <= set(range(6))
    assert len(model.good_scores) == 6


def test_resnet_download_failure_is_friendly(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))

    def offline(**k):
        raise OSError("Tunnel connection failed")
    monkeypatch.setattr(torchvision.models, "resnet18", offline)
    with pytest.raises(RuntimeError, match="indirilemedi"):
        pc.ensure_resnet()
    assert not list((tmp_path / "modeller").glob("*")) if (tmp_path / "modeller").exists() else True
