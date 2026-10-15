"""2.2 (b): part traceability archive — engine (visionflow/trace.py), 'Parçayı Arşivle' / 'Arşivde Ara' blocks,
browser window and the 'arsiv' command line."""
import csv
import datetime as dt
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import zipfile

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow import trace  # noqa: E402
from visionflow.core import Graph, create, load_builtin  # noqa: E402

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]


def part_image(i=0, h=120, w=160, noise=False):
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30 + i % 10, (40, 200, 220), -1)
    if noise:
        img = np.random.default_rng(i).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return img


@pytest.fixture
def writers():
    made = []

    def make(root, **kw):
        kw.setdefault("log", lambda *_: None)
        w = trace.ArchiveWriter(root, **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close(5)


# --- engine ---------------------------------------------------------------------------------------------------------

def test_thousands_of_records_are_written_without_blocking(tmp_path, writers):
    w = writers(tmp_path / "a", queue_size=10000)
    gate = threading.Event()
    real = w._write_batch

    def slow(con, batch):          # a very slow disk: the writer is stuck until the gate opens
        gate.wait(10)
        real(con, batch)
    w._write_batch = slow
    img = part_image()
    t = time.perf_counter()
    ids = [w.submit(serial=f"SN{i:05d}", ok=i % 4 != 0, values={"çap": 10 + i / 1000}, image=img if i % 50 == 0
                    else None) for i in range(5000)]
    elapsed = time.perf_counter() - t
    assert elapsed < 3.0, elapsed               # submit never waited for the stuck disk
    assert all(ids) and len(set(ids)) == 5000
    assert w.pending > 4000
    gate.set()
    assert w.flush(60)
    a = trace.Archive(tmp_path / "a")
    assert a.count() == 5000 and w.written == 5000 and w.dropped == 0
    s = a.summary()
    assert (s["good"], s["bad"]) == (3750, 1250)
    assert len(list((tmp_path / "a" / "resimler").rglob("*.jpg"))) == 100


def test_queue_overflow_drops_counts_and_warns_once(tmp_path, writers):
    logs = []
    w = writers(tmp_path / "q", queue_size=10, log=logs.append)
    gate = threading.Event()
    real = w._write_batch
    w._write_batch = lambda con, batch: (gate.wait(10), real(con, batch))
    ids = [w.submit(serial=str(i), ok=True) for i in range(60)]
    dropped = ids.count("")
    assert dropped >= 39 and w.dropped == dropped
    assert sum("kuyruğu dolu" in m for m in logs) == 1      # rate-limited warning
    gate.set()
    assert w.flush(20)
    assert trace.Archive(tmp_path / "q").count() == 60 - dropped == w.written


def test_queue_memory_limit_drops_big_images(tmp_path, writers):
    w = writers(tmp_path / "m", queue_size=1000, queue_mb=1)
    gate = threading.Event()
    real = w._write_batch
    w._write_batch = lambda con, batch: (gate.wait(10), real(con, batch))
    big = np.zeros((600, 800, 3), np.uint8)               # 1.4 MB each
    ids = [w.submit(serial=str(i), ok=False, image=big) for i in range(5)]
    assert ids[0] and ids.count("") >= 3                    # the first always fits, the rest exceed 1 MB
    gate.set()
    w.flush(20)


def test_retention_by_days(tmp_path, writers):
    now = time.time()
    w = writers(tmp_path / "r", keep_days=7, clock=lambda: now)
    for days in (30, 10, 8, 6, 1, 0):
        w.submit(serial=f"G{days}", ok=False, image=part_image(days), ts=now - days * 86400)
    w.flush(10)
    old_day = dt.datetime.fromtimestamp(now - 30 * 86400).strftime("%Y-%m-%d")
    assert (tmp_path / "r" / "resimler" / old_day).is_dir()
    assert w.apply_retention() == 3
    a = trace.Archive(tmp_path / "r")
    assert sorted(r["serial"] for r in a.search()) == ["G0", "G1", "G6"]
    assert not (tmp_path / "r" / "resimler" / old_day).exists()
    assert len(list((tmp_path / "r" / "resimler").rglob("*.jpg"))) == 3


def test_retention_by_size_keeps_newest(tmp_path, writers):
    w = writers(tmp_path / "s", jpeg_quality=95)
    t0 = time.time() - 1000
    for i in range(30):
        w.submit(serial=f"S{i:02d}", ok=False, image=part_image(i, 200, 200, noise=True), ts=t0 + i)
    w.flush(20)
    per = sum(f.stat().st_size for f in (tmp_path / "s").rglob("*.jpg")) / 30
    w.max_gb = (per * 10 + 200_000) / 1024 ** 3                # ~10 pictures + the database
    removed = w.apply_retention()
    assert 5 < removed < 30
    left = [r["serial"] for r in trace.Archive(tmp_path / "s").search(limit=100, newest_first=False)]
    assert left == [f"S{i:02d}" for i in range(30 - len(left), 30)]      # the oldest went first
    on_disk = sum(f.stat().st_size for f in (tmp_path / "s").rglob("*.jpg"))
    assert on_disk + w._db_bytes(w._connect()) <= w.max_gb * 1024 ** 3
    assert len(list((tmp_path / "s").rglob("*.jpg"))) == len(left)


def test_writer_thread_prunes_by_itself(tmp_path, writers, monkeypatch):
    monkeypatch.setattr(trace, "RETENTION_EVERY_S", 0.0)
    now = time.time()
    w = writers(tmp_path / "auto", keep_days=1)
    w.submit(serial="eski", ok=True, ts=now - 3 * 86400)
    w.submit(serial="yeni", ok=True, ts=now)
    w.flush(10)
    deadline = time.time() + 5
    a = trace.Archive(tmp_path / "auto")
    while time.time() < deadline and a.count() != 1:
        time.sleep(0.05)
    assert [r["serial"] for r in a.search()] == ["yeni"]


def test_images_only_for_failed_parts(tmp_path, writers):
    w = writers(tmp_path / "f", images="Yalnız hatalı parçalar")
    img, ann = part_image(1), part_image(2)
    w.submit(serial="OK1", ok=True, image=img, annotated=ann)
    w.submit(serial="NOK1", ok=False, image=img, annotated=ann)
    w.submit(serial="BELIRSIZ", ok=None, image=img)
    w.submit(serial="HİÇ", ok=False, image=img, images="Hiçbiri")
    w.flush(10)
    a = trace.Archive(tmp_path / "f")
    recs = {r["serial"]: r for r in a.search()}
    assert recs["OK1"]["img"] == "" and recs["OK1"]["ann"] == ""
    assert recs["NOK1"]["img"].endswith("_o.jpg") and recs["NOK1"]["ann"].endswith("_a.jpg")
    assert recs["BELIRSIZ"]["img"]                    # unknown verdict keeps its picture
    assert recs["HİÇ"]["img"] == ""
    back = trace.read_image(recs["NOK1"]["img_path"])
    assert back.shape == img.shape and np.abs(back.astype(int) - img).mean() < 4
    assert recs["NOK1"]["bytes"] == sum(os.path.getsize(recs["NOK1"][k]) for k in ("img_path", "ann_path"))


@pytest.mark.parametrize("img", [
    np.full((50, 60), 200, np.uint8),
    np.full((50, 60), 60000, np.uint16),
    np.full((50, 60, 3), 0.5, np.float32),
    np.full((50, 60, 4), 77, np.uint8),
    np.full((50, 60, 1), 9, np.uint8),
    np.ones((50, 60), bool),
    np.full((50, 60, 6), 5, np.uint8),
])
def test_any_image_kind_is_stored_as_jpeg(tmp_path, writers, img):
    w = writers(tmp_path / "k")
    w.submit(serial="x", ok=False, image=img)
    w.flush(10)
    r = trace.Archive(tmp_path / "k").search()[0]
    back = trace.read_image(r["img_path"])
    assert back is not None and back.shape[:2] == (50, 60)


def test_bad_inputs_give_turkish_errors(tmp_path, writers):
    w = writers(tmp_path / "e")
    with pytest.raises(ValueError, match="Görüntü bekleniyor"):
        w.submit(serial="x", image="resim.png")
    with pytest.raises(ValueError, match="boyut"):
        w.submit(serial="x", image=np.zeros((2, 2, 2, 2), np.uint8))
    with pytest.raises(ValueError, match="ölçüm değeri"):
        trace.values_dict(np.zeros((100, 100), np.uint8))
    assert w.submit(serial=None, ok=True, image=np.zeros((0, 0, 3), np.uint8))      # empty picture: no picture
    (tmp_path / "dosya").write_text("x")
    with pytest.raises(ValueError, match="Arşiv"):
        trace.ArchiveWriter(tmp_path / "dosya" / "alt")
    with pytest.raises(ValueError, match="Görüntü saklama"):
        trace.ArchiveWriter(tmp_path / "e2", images="bazen")


def test_values_and_text_fields_are_normalised(tmp_path, writers):
    assert trace.values_dict(3.5) == {"değer": 3.5}
    assert trace.values_dict(None) == {}
    assert trace.values_dict(np.float32(2.0)) == {"değer": 2.0}
    assert trace.values_dict({"a": np.int64(3), 5: [1, 2]}) == {"a": 3, "5": [1, 2]}
    assert trace.values_dict(float("nan")) == {"değer": "nan"}
    assert trace.clean_text(123.0) == "123" and trace.clean_text("a\nb\x00") == "a b"
    w = writers(tmp_path / "v")
    w.submit(serial=4711.0, ok=np.bool_(True), values={"n": np.float64(1.25), "liste": np.arange(3)})
    w.flush(10)
    r = trace.Archive(tmp_path / "v").search()[0]
    assert r["serial"] == "4711" and r["ok"] is True and r["values"] == {"n": 1.25, "liste": [0, 1, 2]}


def _fill(w, now):
    rows = [("ABC123", True, "", "Hat-A", "İstasyon 1", "Ürün A", 3),
            ("abc124", False, "Çizik", "Hat-A", "İstasyon 1", "Ürün A", 2),
            ("XYZ-ığüşöç", False, "Çapak", "Hat-B", "İstasyon 2", "Ürün B", 1),
            ("İZMİR-7", False, "Çizik", "Hat-B", "İstasyon 2", "Ürün B", 0),
            ("ABC123", False, "Ölçü dışı", "Hat-A", "Tekrar kontrol", "Ürün A", 0)]
    for i, (s, ok, reason, flow, st, rec, days) in enumerate(rows):
        w.submit(serial=s, ok=ok, reason=reason, flow=flow, station=st, recipe=rec, lot=f"LOT-{i}",
                 values={"çap": 10 + i, "durum": reason or "iyi"}, image=part_image(i), annotated=part_image(i + 1),
                 ts=now - days * 86400 + i)
    assert w.flush(10)


def test_search_filters(tmp_path, writers):
    now = time.time()
    w = writers(tmp_path / "ara")
    _fill(w, now)
    a = trace.Archive(tmp_path / "ara")
    F = trace.Filter
    assert a.count() == 5
    assert {r["serial"] for r in a.search(F(serial="abc"))} == {"ABC123", "abc124"}
    assert [r["serial"] for r in a.search(F(serial="izmir"))] == ["İZMİR-7"]       # Turkish İ folds to i
    assert [r["serial"] for r in a.search(F(serial="IĞÜŞ"))] == ["XYZ-ığüşöç"]
    assert [r["serial"] for r in a.search(F(serial="lot-2"))] == ["XYZ-ığüşöç"]     # the lot is searched too
    assert a.count(F(serial="%")) == 0 and a.count(F(serial="_")) == 0               # LIKE wildcards are literal
    assert a.count(F(verdict=True)) == 1 and a.count(F(verdict=False)) == 4
    assert a.count(F(flow="Hat-B")) == 2 and a.count(F(station="İstasyon 1")) == 2
    assert a.count(F(recipe="Ürün A", verdict=False)) == 2
    assert a.count(F(reason="çiz")) == 2
    assert a.count(F(start=now - 1.5 * 86400)) == 3
    assert a.count(F(start=now - 2.5 * 86400, end=now - 0.5 * 86400)) == 2
    newest = a.search(limit=2)
    assert [r["serial"] for r in newest] == ["ABC123", "İZMİR-7"]
    assert [r["serial"] for r in a.search(limit=2, offset=4)] == ["ABC123"]
    s = a.summary()
    assert s["count"] == 5 and s["bad"] == 4 and s["fail_pct"] == 80.0
    assert s["top_reasons"][0] == ("Çizik", 2)
    assert a.distinct("station") == ["Tekrar kontrol", "İstasyon 1", "İstasyon 2"]
    last, n = a.last_by_serial("ABC123")
    assert n == 2 and last["station"] == "Tekrar kontrol" and last["ok"] is False
    assert a.last_by_serial("ABC123", station="İstasyon 1")[0]["ok"] is True
    assert a.last_by_serial("yok") == (None, 0)
    assert a.get(last["rid"])["serial"] == "ABC123"
    assert trace.Archive(tmp_path / "bos").search() == [] and trace.Archive(tmp_path / "bos").count() == 0
    assert not (tmp_path / "bos").exists()                  # reading does not create an archive


def test_export_csv_xlsx_zip(tmp_path, writers):
    w = writers(tmp_path / "ex")
    _fill(w, time.time())
    a = trace.Archive(tmp_path / "ex")
    out = tmp_path / "çıktı klasörü" / "kayıtlar.csv"
    msg = a.export(out, flt=trace.Filter(verdict=False))
    assert "4 kayıt" in msg
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))
    assert rows[0][:7] == ["Kayıt no", "Zaman", "Akış", "İstasyon", "Seri no", "Lot", "Sonuç"]
    assert rows[0][-2:] == ["çap", "durum"]
    assert len(rows) == 5 and {r[6] for r in rows[1:]} == {"Hatalı"}
    assert "XYZ-ığüşöç" in [r[4] for r in rows]

    pytest.importorskip("openpyxl")
    import openpyxl
    xl = tmp_path / "a.xlsx"
    a.export(xl)
    ws = openpyxl.load_workbook(xl).active
    vals = list(ws.values)
    assert len(vals) == 6 and vals[0][4] == "Seri no" and "İZMİR-7" in [v[4] for v in vals]

    z = tmp_path / "a.zip"
    msg = a.export(z, records=a.search(trace.Filter(serial="abc")))
    assert "3 kayıt ve 6 görüntü" in msg
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert "kayitlar.csv" in names
        imgs = [n for n in names if n.startswith("resimler/")]
        assert len(imgs) == 6 and any(n.endswith("_ABC123_orijinal.jpg") for n in imgs)
        assert any(n.endswith("_isaretli.jpg") for n in imgs)
        data = np.frombuffer(zf.read(imgs[0]), np.uint8)
        assert cv2.imdecode(data, cv2.IMREAD_COLOR).shape == (120, 160, 3)
        assert zf.read("kayitlar.csv").decode("utf-8-sig").count("\n") == 4

    with pytest.raises(ValueError, match="Desteklenmeyen"):
        a.export(tmp_path / "a.pdf")


def test_turkish_folder_and_serial(tmp_path, writers):
    root = tmp_path / "Üretim Şubesi" / "İzlenebilirlik ğüşıöç"
    w = writers(root)
    w.submit(serial="ŞĞÜ/İ 12:34*?", ok=False, image=part_image(), reason="Çatlak")
    w.flush(10)
    r = trace.Archive(root).search()[0]
    assert r["serial"] == "ŞĞÜ/İ 12:34*?"
    name = pathlib.Path(r["img"]).name
    assert "/" not in name and "*" not in name and "?" not in name and ":" not in name and "ŞĞÜ" in name
    assert trace.read_image(r["img_path"]).shape == (120, 160, 3)


def test_concurrent_reads_while_writing(tmp_path, writers):
    w = writers(tmp_path / "c", queue_size=100000)
    a = trace.Archive(tmp_path / "c")
    w.submit(serial="ilk", ok=True)
    w.flush(5)
    stop = threading.Event()
    errors = []

    def produce():
        i = 0
        while not stop.is_set():
            w.submit(serial=f"P{i}", ok=i % 2 == 0, values={"i": i})
            i += 1
            if i % 200 == 0:
                time.sleep(0.005)
    t = threading.Thread(target=produce)
    t.start()
    counts = []
    try:
        end = time.time() + 1.5
        while time.time() < end:
            try:
                counts.append(a.count())
                a.search(trace.Filter(serial="P1"), limit=20)
                a.summary()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
    finally:
        stop.set()
        t.join()
    assert w.flush(30)
    assert not errors
    assert counts == sorted(counts) and counts[-1] > counts[0]
    assert a.count() == w.written


def test_shared_writer_per_folder(tmp_path):
    a = trace.open_writer(tmp_path / "ortak", log=lambda *_: None)
    b = trace.open_writer(tmp_path / "ortak", keep_days=3)
    assert a is b and a.keep_days == 3
    a.submit(serial="1", ok=True)
    assert trace.pending_writer(tmp_path / "ortak") is a
    assert trace.release_writer(a)
    assert a._thread is not None and a._thread.is_alive() or a.pending == 0
    b.submit(serial="2", ok=True)
    assert trace.release_writer(b)
    assert trace.pending_writer(tmp_path / "ortak") is None
    assert b._thread is None
    assert trace.Archive(tmp_path / "ortak").count() == 2


def test_parse_date():
    assert trace.parse_date("2026-10-01") == dt.datetime(2026, 10, 1).timestamp()
    assert trace.parse_date("01.10.2026 08:30") == dt.datetime(2026, 10, 1, 8, 30).timestamp()
    assert trace.parse_date("2026-10-01", end=True) == dt.datetime(2026, 10, 2).timestamp()
    with pytest.raises(ValueError, match="Tarih"):
        trace.parse_date("dün")


# --- blocks ---------------------------------------------------------------------------------------------------------

def test_archive_block_edges_outputs_and_errors(tmp_path):
    g = Graph()
    n = g.add(create("ArchivePart"))
    n.values.update(folder=str(tmp_path / "blok"), station="Hat 3", flow_name="Kapak hattı", images="Hepsi")
    assert not g.setup().get(n.id)
    try:
        img = part_image()
        out = n.process({"image": img, "annotated": None, "serial": "K-1", "ok": True, "reason": "", "values": 5.0,
                         "lot": "L1", "trigger": True})
        assert out["saved"] and out["record_id"]
        again = n.process({"image": img, "annotated": None, "serial": "K-1", "ok": True, "reason": "",
                           "values": None, "lot": None, "trigger": True})
        assert not again["saved"] and again["record_id"] == out["record_id"]      # held trigger: one part
        n.process({"image": None, "annotated": None, "serial": "K-2", "ok": False, "reason": "x", "values": None,
                   "lot": None, "trigger": False})
        out2 = n.process({"image": img, "annotated": img, "serial": "K-2", "ok": 0, "reason": "Leke",
                          "values": {"alan": 12}, "lot": None, "trigger": True})
        assert out2["saved"]
        with pytest.raises(ValueError, match="görüntü"):
            n.process({"image": "a.png", "annotated": None, "serial": "", "ok": True, "reason": "", "values": None,
                       "lot": None, "trigger": None})
        with pytest.raises(ValueError, match="doğru/yanlış"):
            n.process({"image": None, "annotated": None, "serial": "", "ok": "evet", "reason": "", "values": None,
                       "lot": None, "trigger": None})
    finally:
        g.teardown()
    recs = trace.Archive(tmp_path / "blok").search()
    assert [r["serial"] for r in recs] == ["K-2", "K-1"]
    assert recs[0]["ok"] is False and recs[0]["reason"] == "Leke" and recs[0]["values"] == {"alan": 12}
    assert recs[1]["values"] == {"değer": 5.0} and recs[1]["lot"] == "L1"
    assert recs[1]["station"] == "Hat 3" and recs[1]["flow"] == "Kapak hattı"


def test_archive_block_reads_recipe_and_bad_folder(tmp_path):
    g = Graph()
    n = g.add(create("ArchivePart"))
    sel = g.add(create("RecipeSelect"))
    n.values["folder"] = str(tmp_path / "r")
    g.setup()
    try:
        sel.state["name"] = "Ürün C"
        n.process({"image": None, "annotated": None, "serial": "R1", "ok": True, "reason": "", "values": None,
                   "lot": None, "trigger": None})
    finally:
        g.teardown()
    assert trace.Archive(tmp_path / "r").search()[0]["recipe"] == "Ürün C"
    (tmp_path / "f").write_text("dosya")
    g2 = Graph()
    m = g2.add(create("ArchivePart"))
    m.values["folder"] = str(tmp_path / "f" / "alt")
    errs = g2.setup()
    g2.teardown()
    assert "Arşiv" in errs[m.id]


def test_blocks_in_a_headless_run_and_lookup(tmp_path):
    from visionflow import headless
    folder = tmp_path / "hat arşivi"
    g = Graph()
    sim = g.add(create("SignalSimulator"))
    sim.values.update(rate=60000, bad_ratio=30, seed=3)
    mk = g.add(create("PythonScript"))
    mk.values["code"] = ("out = f'SN-{int(a)}'\nout_img = np.full((80, 100, 3), 40, np.uint8)\n"
                         "cv2.putText(out_img, out, (2, 40), 0, 0.5, (255, 255, 255), 1)")
    inv = g.add(create("PythonScript"))
    inv.values["code"] = "out = not a"
    arc = g.add(create("ArchivePart"))
    arc.values.update(folder=str(folder), images="Yalnız hatalı parçalar", station="Son kontrol")
    g.connect(sim.id, "parts", mk.id, "a")
    g.connect(sim.id, "bad", inv.id, "a")
    g.connect(mk.id, "out_img", arc.id, "image")
    g.connect(mk.id, "out", arc.id, "serial")
    g.connect(inv.id, "out", arc.id, "ok")
    g.connect(sim.id, "measurement", arc.id, "values")
    g.connect(sim.id, "trigger", arc.id, "trigger")
    flow = tmp_path / "akış.json"
    g.save(str(flow))
    assert headless.run_flow(str(flow), frames=60, fps=0, stats=999) == 0
    a = trace.Archive(folder)
    recs = a.search(limit=1000)
    assert len(recs) >= 5                                   # one record per simulated part
    assert len({r["serial"] for r in recs}) == len(recs)
    assert all(r["station"] == "Son kontrol" and "değer" in r["values"] and r["flow"] == "akış" for r in recs)
    assert all(bool(r["img"]) == (r["ok"] is False) for r in recs)
    assert any(r["ok"] is False for r in recs) and any(r["ok"] for r in recs)
    assert trace.pending_writer(folder) is None             # the run's teardown stopped the writer

    bad = next(r for r in recs if r["ok"] is False)
    look = create("ArchiveLookup")
    look.values["folder"] = str(folder)
    look.safe_setup()
    out = look.process({"serial": bad["serial"], "trigger": None})
    assert out["found"] and out["ok"] is False and out["record_id"] == bad["rid"] and out["count"] == 1
    assert out["image"].shape == (80, 100, 3) and out["values"] == bad["values"]
    assert look.process({"serial": "YOK-1", "trigger": None})["found"] is False
    assert look.process({"serial": None, "trigger": None})["found"] is False
    look.values["folder"] = str(tmp_path / "olmayan")
    with pytest.raises(ValueError, match="Arşiv bulunamadı"):
        look.process({"serial": "SN-1", "trigger": True})


# --- command line ---------------------------------------------------------------------------------------------------

def test_cli_search_and_export(tmp_path, writers, capsys):
    w = writers(tmp_path / "cli")
    _fill(w, time.time())
    root = str(tmp_path / "cli")
    assert trace.cli(["--seri", "ABC123", "--klasor", root]) == 0
    out = capsys.readouterr().out
    assert out.count("ABC123") >= 2 and "HATALI" in out and "SAĞLAM" in out
    assert trace.cli(["--seri", "yok", "--klasor", root]) == 1
    capsys.readouterr()
    start = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    target = tmp_path / "out.csv"
    assert trace.cli(["--disa-aktar", str(target), "--baslangic", start, "--klasor", root]) == 0
    rows = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig")), delimiter=";"))
    assert len(rows) - 1 == trace.Archive(root).count(trace.Filter(start=trace.parse_date(start)))
    assert trace.cli(["--ozet", "--sonuc", "hatalı", "--klasor", root]) == 0
    assert "Hatalı: 4" in capsys.readouterr().out
    assert trace.cli(["--baslangic", "dün", "--klasor", root]) == 2
    assert trace.cli(["--bilinmeyen", "--klasor", root]) == 2
    assert trace.cli(["--seri", "x", "--klasor", str(tmp_path / "yok")]) == 1


def test_cli_through_main_module(tmp_path, writers):
    w = writers(tmp_path / "m")
    w.submit(serial="ABC123", ok=False, reason="Çizik")
    w.flush(5)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "visionflow", "arsiv", "--seri", "ABC123", "--klasor",
                        str(tmp_path / "m")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert "ABC123" in r.stdout and "Çizik" in r.stdout


# --- browser --------------------------------------------------------------------------------------------------------

@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_browser_filter_view_and_export(tmp_path, writers, qapp, monkeypatch):
    from visionflow.ui import trace_browser as tb
    w = writers(tmp_path / "ui")
    _fill(w, time.time())
    for i in range(150):
        w.submit(serial=f"SERI-{i}", ok=True, ts=time.time() - 10 * 86400)
    w.flush(10)
    dlg = tb.TraceBrowser(None, tmp_path / "ui")
    try:
        assert dlg.total == 155 and dlg.table.rowCount() == tb.PAGE
        assert "Sayfa 1/2" in dlg.page_label.text()
        dlg.go(1)
        assert dlg.table.rowCount() == 55 and not dlg.next_btn.isEnabled()
        dlg.serial_edit.setText("abc")
        dlg.search()
        assert dlg.total == 3 and dlg.table.rowCount() == 3
        assert {dlg.table.item(r, 2).text() for r in range(3)} == {"ABC123", "abc124"}
        dlg.verdict_combo.setCurrentText("Hatalı")
        dlg.search()
        assert dlg.total == 2
        assert "hata oranı" in dlg.summary_label.text() and "%100.0" in dlg.summary_label.text()
        dlg.table.selectRow(0)
        assert dlg._image is not None and not dlg.table.item(0, 0).icon().isNull()
        assert "Ölçülen değerler" in dlg.values_view.toHtml() and "çap" in dlg.values_view.toPlainText()
        dlg.annotated_check.setChecked(False)
        rec = dlg.selected_records()[0]
        assert np.array_equal(dlg._image, trace.read_image(rec["img_path"]))
        dlg.serial_edit.clear()
        dlg.verdict_combo.setCurrentText("Hepsi")
        dlg.combos["station"].setCurrentText("İstasyon 2")
        dlg.search()
        assert dlg.total == 2
        out = tmp_path / "filtre.csv"
        assert "2 kayıt" in dlg.export_to(out, selected=False)
        dlg.table.selectRow(1)
        z = tmp_path / "secili.zip"
        assert "1 kayıt ve 2 görüntü" in dlg.export_to(z, selected=True)
        dlg.table.clearSelection()
        with pytest.raises(ValueError, match="seçin"):
            dlg.export_to(tmp_path / "x.csv", selected=True)
        dlg.date_check.setChecked(True)
        assert dlg.start_edit.isEnabled()
        dlg.start_edit.setDateTime(tb.QDateTime.currentDateTime().addSecs(-36 * 3600))
        dlg.combos["station"].setCurrentText("Hepsi")
        dlg.search()
        assert dlg.total == 3                                    # the last 36 hours: the old SERI parts are out
        assert all(not dlg.table.item(r, 2).text().startswith("SERI") for r in range(dlg.table.rowCount()))
        warned = []
        monkeypatch.setattr(tb.QMessageBox, "warning", lambda *a: warned.append(a[2]))
        monkeypatch.setattr(tb.QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "y.pdf"), ""))
        dlg.ask_export(selected=False)
        assert warned and "Desteklenmeyen" in warned[0]
        dlg.set_root(tmp_path / "bos")
        assert dlg.total == 0 and "henüz arşiv yok" in dlg.summary_label.text()
    finally:
        dlg.close()


def test_menu_and_operator_open_the_browser(tmp_path, qapp, monkeypatch):
    monkeypatch.setattr(trace, "default_root", lambda: tmp_path / "varsayilan")
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    win._confirm_discard = lambda: True
    try:
        acts = [a for m in win.menuBar().actions() if m.menu() for a in m.menu().actions()]
        act = next(a for a in acts if "İzlenebilirlik arşivi" in a.text())
        act.trigger()
        assert win._trace_browser.isVisible() and win._trace_browser.archive.root == tmp_path / "varsayilan"
        win._trace_browser.close()
    finally:
        win.close()
    from visionflow.ui.operator import OperatorPanel
    panel = OperatorPanel(Graph().to_dict())
    try:
        panel.trace_btn.click()
        assert panel._trace_browser.isVisible()
        panel._trace_browser.close()
    finally:
        panel.close()
