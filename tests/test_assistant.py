"""AI flow builder: validation, self-correction round, editor integration (fake model, no network)."""
import json
import os
from types import SimpleNamespace

import pytest

from visionflow import assistant
from visionflow.core import Graph, load_builtin

load_builtin()


class FakeClient:
    """Stands in for anthropic.Anthropic(); returns canned answers and records the requests."""

    def __init__(self, *answers, stop="end_turn"):
        self.answers, self.calls, self.stop = list(answers), [], stop
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        text = self.answers.pop(0)
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="thinking", thinking=""),
                                                               SimpleNamespace(type="text", text=text)])


def answer(nodes, conns, text="Kırmızı nesneleri sayan akış kuruldu."):
    return f"{text}\n```json\n{json.dumps({'nodes': nodes, 'connections': conns})}\n```"


GOOD = answer(
    [{"id": "n1", "type": "TestPattern"}, {"id": "n2", "type": "ColorRange", "values": {"h_max": "12"}},
     {"id": "n3", "type": "FindContours", "values": {"min_area": 300}},
     {"id": "n4", "type": "Compare", "values": {}}, {"id": "n5", "type": "ModbusWrite", "values": {"host": "192.168.1.10"}},
     {"id": "n6", "type": "Preview"}],
    [{"src": "n1", "src_port": "image", "dst": "n2", "dst_port": "image"},
     {"src": "n2", "src_port": "mask", "dst": "n3", "dst_port": "mask"},
     {"src": "n3", "src_port": "count", "dst": "n4", "dst_port": "a"},
     {"src": "n4", "src_port": "result", "dst": "n5", "dst_port": "value"},
     {"src": "n1", "src_port": "image", "dst": "n6", "dst_port": "image"}])


def test_catalog_lists_every_block():
    from visionflow.core import NODE_REGISTRY
    cat = assistant.catalog_text()
    assert all(f"- {t}" in cat for t in NODE_REGISTRY)
    assert "in: image:image" in cat and "[" in cat     # ports and choices are spelled out


def test_valid_proposal_builds_runnable_flow():
    fake = FakeClient(GOOD)
    prop = assistant.propose("kırmızıları say, 10'u geçince PLC", client=fake)
    assert prop.warnings == [] and len(fake.calls) == 1
    g = prop.graph
    assert len(g.nodes) == 6 and len(g.connections) == 5
    cr = next(n for n in g.nodes.values() if n.type_name == "ColorRange")
    assert cr.values["h_max"] == 12                       # "12" coerced to the param's int type
    xs = sorted({n.pos[0] for n in g.nodes.values()})
    assert len(xs) >= 4                                    # laid out in columns
    assert prop.explanation.startswith("Kırmızı nesneleri")
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5" and "ColorRange" in req["system"]
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert len(g2.connections) == 5


def test_errors_are_sent_back_and_fixed():
    bad = answer([{"id": "a", "type": "TestPattern"}, {"id": "b", "type": "Bulanik"},
                  {"id": "c", "type": "Blur", "values": {"boyut": 5}}, {"id": "d", "type": "Compare"}],
                 [{"src": "a", "src_port": "image", "dst": "d", "dst_port": "a"}])
    fake = FakeClient(bad, GOOD)
    prop = assistant.propose("bir şey", client=fake)
    assert prop.warnings == [] and len(fake.calls) == 2
    feedback = fake.calls[1]["messages"][-1]["content"]
    assert "Bulanik" in feedback and "boyut" in feedback and "a.image → d.a" in feedback
    assert fake.calls[1]["messages"][-2]["role"] == "assistant"


def test_gives_up_with_partial_flow_and_warnings():
    bad = answer([{"id": "a", "type": "TestPattern"}, {"id": "b", "type": "Yok"}], [])
    fake = FakeClient(bad, bad, bad)
    prop = assistant.propose("x", client=fake)
    assert len(fake.calls) == assistant.MAX_FIX_ROUNDS + 1
    assert len(prop.graph.nodes) == 1 and any("Yok" in w for w in prop.warnings)


def test_no_json_and_refusal_raise():
    with pytest.raises(assistant.AssistantError):
        assistant.propose("x", client=FakeClient("bilmiyorum", "yok", "hayır"))
    with pytest.raises(assistant.AssistantError, match="reddedildi"):
        assistant.propose("x", client=FakeClient(GOOD, stop="refusal"))
    with pytest.raises(assistant.AssistantError):
        assistant.propose("   ", client=FakeClient(GOOD))


def test_modify_sends_current_flow():
    g = Graph()
    g.add_new("Camera")
    g.add_new("Note")
    fake = FakeClient(GOOD)
    assistant.propose("bir de önizleme ekle", current=g, client=fake)
    msg = fake.calls[0]["messages"][0]["content"]
    assert "Camera" in msg and "Note" not in msg and "İstek: bir de önizleme ekle" in msg


def test_bad_choice_and_range_are_reported():
    data = {"nodes": [{"id": "s", "type": "TestPattern", "values": {"motion": "zıpla", "width": 99999}}],
            "connections": []}
    g, errs = assistant.build_graph(data)
    assert len(errs) == 2 and "seçenekler" in errs[0] and "aralık" in errs[1]


def test_cli_writes_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant, "make_client", lambda key=None: FakeClient(GOOD))
    out = tmp_path / "a.json"
    assert assistant.main(["kırmızı say", "-o", str(out)]) == 0
    assert len(json.loads(out.read_text())["nodes"]) == 6


# --- editor ---------------------------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def win(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w.recovery = tmp_path / "k.json"
    w.show()
    yield w, app
    w.stop()
    w.saved_snap = w._snap()
    w.close()


def test_dialog_builds_and_applies_undoably(win):
    w, app = win
    from PySide6.QtCore import QPointF
    w.scene.add_node("Blur", QPointF(0, 0))
    w._snapshot()
    dlg = w.open_assistant(client=FakeClient(GOOD))
    dlg.request.setPlainText("kırmızıları say")
    dlg.generate()
    dlg.job.wait(5000)
    app.processEvents()
    assert dlg.apply_btn.isEnabled() and "Bloklar (6)" in dlg.result.toPlainText()
    dlg.apply()
    assert len(w.scene.graph.nodes) == 6 and len(w.scene.items_by_id) == 6
    w.undo()
    assert [n.type_name for n in w.scene.graph.nodes.values()] == ["Blur"]


def test_dialog_reports_errors(win):
    w, app = win
    dlg = w.open_assistant(client=FakeClient("cevap yok", "yine yok", "hâlâ yok"))
    dlg.request.setPlainText("x")
    dlg.generate()
    dlg.job.wait(5000)
    app.processEvents()
    assert dlg.result.toPlainText().startswith("Olmadı") and not dlg.apply_btn.isEnabled()
    dlg.close()
