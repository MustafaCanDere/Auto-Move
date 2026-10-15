import json
import pathlib

import numpy as np
import pytest

from visionflow.core import NODE_REGISTRY, Graph, GraphError, load_builtin
from visionflow.nodes.ai import letterbox, yolo_postprocess

load_builtin()
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def run(g: Graph):
    out = {}
    g.run(on_frame=lambda n, r: out.update(r))
    return out


def red_pipeline():
    g = Graph()
    s = g.add_new("TestPattern", animate=False)
    c = g.add_new("ColorRange")
    f = g.add_new("FindContours")
    g.connect(s.id, "image", c.id, "image")
    g.connect(c.id, "mask", f.id, "mask")
    return g, f


def test_every_block_has_metadata():
    assert len(NODE_REGISTRY) >= 30
    for cls in NODE_REGISTRY.values():
        assert cls.title and cls.category
        node = cls()
        assert set(node.values) == {p.name for p in cls.params}


def test_counts_red_ball():
    g, f = red_pipeline()
    res = run(g)
    assert res[f.id].error is None
    assert res[f.id].outputs["count"] == 1


def test_type_mismatch_and_cycle_rejected():
    g, f = red_pipeline()
    cmp_ = g.add_new("Compare")
    with pytest.raises(GraphError):
        g.connect(cmp_.id, "result", f.id, "mask")       # bool -> image
    blur1, blur2 = g.add_new("Blur"), g.add_new("Blur")
    g.connect(blur1.id, "image", blur2.id, "image")
    with pytest.raises(GraphError):
        g.connect(blur2.id, "image", blur1.id, "image")  # cycle
    assert len(g.connections) == 3


def test_error_isolated_to_branch():
    g = Graph()
    img = g.add_new("ImageFile", path="yok.png")
    gray = g.add_new("Grayscale")
    num = g.add_new("Number", value=3)
    g.connect(img.id, "image", gray.id, "image")
    res = run(g)
    assert res[img.id].error and res[gray.id].error
    assert res[num.id].outputs["value"] == 3


def test_roundtrip(tmp_path):
    g, _ = red_pipeline()
    p = tmp_path / "a.json"
    g.save(p)
    g2 = Graph.load(p)
    assert g2.to_dict() == g.to_dict()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_examples_load(path):
    g = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert g.order()
    assert not g.warnings, g.warnings


def test_edge_counter_and_python_script():
    g = Graph()
    py = g.add_new("PythonScript", code="state['n'] = state.get('n', 0) + 1\nout = state['n'] % 2 == 0")
    cnt = g.add_new("EdgeCounter")
    g.connect(py.id, "out", cnt.id, "trigger")
    g.setup()
    for _ in range(6):
        res = g.run_once()
    assert res[cnt.id].outputs["count"] == 3


def test_yolo_postprocess_v8_layout():
    img = np.zeros((480, 640, 3), np.uint8)
    _, scale, pad = letterbox(img, 640)
    out = np.zeros((1, 84, 8400), np.float32)
    # one "person" at original (100,50) size 200x100 -> letterbox coords
    cx, cy, w, h = (200 * scale + pad[0], 100 * scale + pad[1], 200 * scale, 100 * scale)
    out[0, :4, 0] = (cx, cy, w, h)
    out[0, 4, 0] = 0.9
    dets = yolo_postprocess(out, 0.5, 0.45, scale, pad, ["person"])
    assert len(dets) == 1 and dets[0]["label"] == "person"
    assert dets[0]["box"] == (100, 50, 200, 100)


def test_yolo_block_with_onnx_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    raw = np.zeros((1, 84, 8400), np.float32)
    raw[0, :4, 0] = (320, 320, 100, 100)
    raw[0, 4 + 2, 0] = 0.95                                 # class 2 = car
    const = helper.make_node("Constant", [], ["output0"], value=helper.make_tensor(
        "v", TensorProto.FLOAT, raw.shape, raw.flatten().tolist()))
    graph = helper.make_graph([const], "fake_yolo",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "fake.onnx"
    onnx.save(model, path)

    g = Graph()
    s = g.add_new("TestPattern", animate=False, width=640, height=640)
    y = g.add_new("YoloDetector", model=str(path))
    g.connect(s.id, "image", y.id, "image")
    res = run(g)
    assert res[y.id].error is None, res[y.id].error
    assert res[y.id].outputs["count"] == 1
    assert res[y.id].outputs["detections"][0]["label"] == "car"
