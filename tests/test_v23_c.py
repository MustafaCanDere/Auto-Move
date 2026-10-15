"""2.3 · Veri Seti Yöneticisi: statistics and problems, class edits with backup, stratified split, COCO / VOC / YOLO
conversion, box-aware augmentation, perceptual-hash duplicates, model-assisted labelling with review, CLI, window."""
import json
import os
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from visionflow import datasets as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 480, 360


# --- synthetic data -------------------------------------------------------------------------------------------------

def part(cls: int) -> np.ndarray:
    if cls == 0:                # red plate with a white cross and a dark dot
        p = np.full((40, 60, 3), (40, 40, 200), np.uint8)
        cv2.line(p, (8, 20), (52, 20), (255, 255, 255), 3)
        cv2.line(p, (30, 6), (30, 34), (255, 255, 255), 3)
        cv2.circle(p, (48, 30), 4, (20, 20, 20), -1)
    else:                       # green square with white stripes
        p = np.full((44, 44, 3), (60, 170, 60), np.uint8)
        for x in range(6, 40, 9):
            cv2.rectangle(p, (x, 4), (x + 3, 39), (240, 240, 240), -1)
    return p


def make(seed: int, n: int = 3, classes=(0, 1)):
    """(BGR picture, [(x, y, w, h, class)]) — parts pasted upright on a noisy grey background."""
    rng = np.random.default_rng(seed)
    img = (np.full((H, W, 3), 110, np.float32) + rng.normal(0, 6, (H, W, 3))).clip(0, 255).astype(np.uint8)
    boxes, occ, tries = [], np.zeros((H, W), bool), 0
    while len(boxes) < n and tries < 300:
        tries += 1
        c = int(classes[int(rng.integers(len(classes)))])
        p = part(c)
        h, w = p.shape[:2]
        x, y = int(rng.integers(5, W - w - 5)), int(rng.integers(5, H - h - 5))
        if occ[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8].any():
            continue
        img[y:y + h, x:x + w] = p
        occ[y:y + h, x:x + w] = True
        boxes.append((x, y, w, h, c))
    return img, boxes


def yolo_lines(boxes, w=W, h=H):
    return "\n".join(D.format_line(D.to_yolo([c, x, y, x + bw, y + bh], w, h)) for x, y, bw, bh, c in boxes)


def build(root: pathlib.Path, n: int = 10, labelled: int | None = None, classes=("plaka", "kare"), seed0: int = 0):
    """YOLO dataset: n pictures, the first ``labelled`` with labels. Returns ({stem: gt boxes}, Dataset)."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    gts = {}
    for i in range(n):
        img, gt = make(seed0 + i)
        D.imwrite(root / "images" / f"r{i:02d}.png", img)
        gts[f"r{i:02d}"] = gt
        if labelled is None or i < labelled:
            (root / "labels" / f"r{i:02d}.txt").write_text(yolo_lines(gt), encoding="utf-8")
    (root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    return gts, D.Dataset(root)


def px_boxes(ds, item):
    w, h = ds.size(item)
    return sorted([round(v, 3) for v in D.to_xyxy(b, w, h)] for b in ds.boxes(item))


def best_iou(gt, boxes):
    g = (gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])
    return max([D.box_iou(g, b[1:]) for b in boxes if b[0] == gt[4]] or [0.0])


# --- statistics and problems ----------------------------------------------------------------------------------------

def test_stats_counts_and_problems(tmp_path):
    gts, ds = build(tmp_path, 6, labelled=4)
    lbl = tmp_path / "labels"
    # r01: a tiny box, a box half outside, a broken line, an unknown class
    (lbl / "r01.txt").write_text(yolo_lines(gts["r01"]) + "\n0 0.5 0.5 0.004 0.004\n1 0.98 0.5 0.1 0.1\n"
                                 "x y z\n7 0.5 0.5 0.1 0.1\n0 0.1 0.1 0.2 0.2 0.3 0.3 0.1 0.4", encoding="utf-8")
    (lbl / "r03.txt").write_text("", encoding="utf-8")            # labelled, no objects
    (tmp_path / "images" / "bozuk.png").write_bytes(b"not a picture")
    dup = cv2.GaussianBlur(D.imread(tmp_path / "images" / "r00.png"), (3, 3), 0)
    D.imwrite(tmp_path / "images" / "r00_kopya.jpg", dup)
    ds.reload()
    st = ds.stats()
    assert st["images"] == 8 and st["labelled"] == 4 and st["unlabelled"] == 4 and st["empty"] == 1
    n0 = sum(1 for k in ("r00", "r01", "r02") for b in gts[k] if b[4] == 0) + 1      # + the tiny box
    n1 = sum(1 for k in ("r00", "r01", "r02") for b in gts[k] if b[4] == 1) + 1      # + the outside box
    assert st["per_class"] == [n0, n1] and st["unknown_classes"] == {7: 1}
    assert st["boxes"] == n0 + n1 + 1
    assert st["images_per_class"][0] == len({k for k in ("r00", "r01", "r02") if any(b[4] == 0 for b in gts[k])})
    kinds = {(p["image"], p["kind"]) for p in st["problems"]}
    assert ("images/r01.png", "kucuk") in kinds and ("images/r01.png", "disarida") in kinds
    assert ("images/r01.png", "bozuk") in kinds and ("images/r01.png", "sinif") in kinds
    assert ("images/bozuk.png", "okunamadi") in kinds
    assert st["duplicates"] == [["images/r00.png", "images/r00_kopya.jpg"]]
    assert ("images/r00_kopya.jpg", "kopya") in kinds
    # no false alarms on clean pictures
    assert not any(p["image"] in ("images/r00.png", "images/r02.png") for p in st["problems"])
    assert st["sizes"] == {f"{W}x{H}": 7} and st["min_size"] == (W, H)
    # filters
    assert {it.rel for it in ds.filter("unlabelled")} == {"images/r04.png", "images/r05.png", "images/bozuk.png",
                                                          "images/r00_kopya.jpg"}
    assert {it.rel for it in ds.filter("empty")} == {"images/r03.png"}
    assert {it.rel for it in ds.filter("problems", st)} == {"images/r01.png", "images/bozuk.png",
                                                             "images/r00_kopya.jpg"}
    assert all(any(b[0] == 1 for b in ds.boxes(it)) for it in ds.filter("class:1"))
    # the polygon line survives a stats run and is counted as "other", not broken
    assert "0 0.1 0.1 0.2 0.2 0.3 0.3 0.1 0.4" in (lbl / "r01.txt").read_text()
    # cache: second run reads no picture
    calls = []
    orig = D.imread
    D.imread = lambda p: calls.append(p) or orig(p)
    try:
        assert D.Dataset(tmp_path).stats()["boxes"] == st["boxes"]
    finally:
        D.imread = orig
    assert calls == [tmp_path / "images" / "bozuk.png"] or calls == []


def test_tiny_threshold_and_empty_dataset(tmp_path):
    (tmp_path / "e").mkdir()
    st = D.Dataset(tmp_path / "e").stats()
    assert st["images"] == 0 and st["balance"] is None and st["problems"] == [] and st["min_size"] is None
    with pytest.raises(ValueError, match="yok"):
        D.Dataset(tmp_path / "yok")
    gts, ds = build(tmp_path / "d", 2)
    smallest = min(min(b[2], b[3]) for g in gts.values() for b in g)
    assert not any(p["kind"] == "kucuk" for p in ds.stats()["problems"])
    assert any(p["kind"] == "kucuk" for p in ds.stats(tiny_px=smallest + 1)["problems"])


def test_layouts_flat_roboflow_and_split_folders(tmp_path):
    img, gt = make(1)
    flat = tmp_path / "flat"
    flat.mkdir()
    D.imwrite(flat / "a.png", img)
    (flat / "a.txt").write_text(yolo_lines(gt))
    ds = D.Dataset(flat)
    assert ds.layout == "flat" and len(ds.items) == 1 and len(ds.boxes(ds.items[0])) == len(gt)
    rf = tmp_path / "rf"
    for s in ("train", "valid"):
        (rf / s / "images").mkdir(parents=True)
        (rf / s / "labels").mkdir(parents=True)
        D.imwrite(rf / s / "images" / f"{s}.jpg", img)
        (rf / s / "labels" / f"{s}.txt").write_text(yolo_lines(gt))
    (rf / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnames: ['plaka', 'kare']\n")
    ds = D.Dataset(rf)
    assert ds.layout == "roboflow" and ds.classes == ["plaka", "kare"]
    assert {it.split for it in ds.items} == {"train", "val"} and all(ds.boxes(it) for it in ds.items)
    ys = tmp_path / "ys"
    for s in ("train", "val"):
        (ys / "images" / s).mkdir(parents=True)
        (ys / "labels" / s).mkdir(parents=True)
        D.imwrite(ys / "images" / s / "x.png", img)
        (ys / "labels" / s / "x.txt").write_text(yolo_lines(gt[:1]))
    ds = D.Dataset(ys)
    assert ds.layout == "yolo" and sorted(it.split for it in ds.items) == ["train", "val"]
    assert all(len(ds.boxes(it)) == 1 for it in ds.items)
    assert ds.find("images/val/x.png").split == "val" and ds.find("images/train/x.png").split == "train"
    assert ds.find(str(ys / "images" / "val" / "x.png")).split == "val" and ds.find("x.png") is not None


# --- class edits ----------------------------------------------------------------------------------------------------

def test_rename_merge_delete_with_backup_and_restore(tmp_path):
    gts, ds = build(tmp_path, 5, classes=("plaka", "kare", "vida"))
    lp = tmp_path / "labels" / "r00.txt"
    lp.write_text(lp.read_text() + "\n2 0.5 0.5 0.1 0.1\n2 0.1 0.1 0.2 0.2 0.3 0.3 0.1 0.4")
    (tmp_path / D.SUGGEST_DIR / "images").mkdir(parents=True)
    (tmp_path / D.SUGGEST_DIR / "images" / "r01.txt").write_text("2 0.5 0.5 0.2 0.2 0.70")
    ds.reload()
    before = {p.name: p.read_text() for p in (tmp_path / "labels").glob("*.txt")}
    count = lambda c: sum(1 for it in ds.items for b in ds.boxes(it) if b[0] == c)  # noqa: E731
    n0, n1 = count(0), count(1)

    with pytest.raises(ValueError, match="zaten var"):
        ds.rename_class(0, "KARE")
    with pytest.raises(ValueError, match="boş"):
        ds.rename_class(0, "  ")
    bk = ds.rename_class(0, "levha  \n kırmızı")
    assert ds.classes == ["levha kırmızı", "kare", "vida"]
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8").splitlines()[0] == "levha kırmızı"
    assert "levha kırmızı" in (tmp_path / "data.yaml").read_text(encoding="utf-8") and (bk / "classes.txt").exists()

    bk = ds.merge_classes([1, 2], 1)                # vida → kare
    assert ds.classes == ["levha kırmızı", "kare"]
    assert count(0) == n0 and count(1) == n1 + 1
    assert lp.read_text().splitlines()[-1].startswith("1 0.1 0.1")          # polygon remapped too
    assert ds.suggestions(ds.find("images/r01.png"))[0][0] == 1             # suggestion remapped too
    assert {p.name: p.read_text() for p in (bk / "labels").glob("*.txt")} == before   # backup = untouched files

    ds.delete_class(0)
    assert ds.classes == ["kare"] and count(0) == n1 + 1
    assert all(b[0] == 0 for it in ds.items for b in ds.boxes(it))
    with pytest.raises(ValueError):
        ds.delete_class(5)
    with pytest.raises(ValueError, match="başka"):
        ds.merge_classes([0], 0)
    # restore the first backup: everything as it was
    first = sorted(ds.backups())[0]
    ds.restore(first)
    assert {p.name: p.read_text() for p in (tmp_path / "labels").glob("*.txt")} == before
    assert ds.classes == ["plaka", "kare", "vida"]


# --- split ----------------------------------------------------------------------------------------------------------

def _split_set(tmp_path, n_a=60, n_b=20, n_empty=10, n_unl=10):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img = np.full((20, 20, 3), 128, np.uint8)
    k = 0
    for kind, n in (("a", n_a), ("b", n_b), ("e", n_empty), ("u", n_unl)):
        for _ in range(n):
            name = f"{kind}{k:03d}"
            D.imwrite(tmp_path / "images" / f"{name}.png", img)
            if kind == "a":
                (tmp_path / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2")
            elif kind == "b":
                (tmp_path / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1")
            elif kind == "e":
                (tmp_path / "labels" / f"{name}.txt").write_text("")
            k += 1
    (tmp_path / "classes.txt").write_text("a\nb")
    return D.Dataset(tmp_path)


def test_split_stratified_ratios_and_determinism(tmp_path):
    ds = _split_set(tmp_path)
    out = ds.split((0.7, 0.2, 0.1), seed=3)
    assert [len(out[s]) for s in D.SPLITS] == [70, 20, 10]
    for kind, n in (("a", 60), ("b", 20), ("e", 10), ("u", 10)):         # every group in the same ratio
        got = [sum(1 for it in out[s] if it.image.name.startswith(kind)) for s in D.SPLITS]
        assert sum(got) == n
        for g, r in zip(got, (0.7, 0.2, 0.1)):
            assert abs(g - r * n) <= 1, (kind, got)
    assert sorted(it.rel for s in D.SPLITS for it in out[s]) == sorted(it.rel for it in ds.items)
    text = {s: (tmp_path / f"{s}.txt").read_text() for s in D.SPLITS}
    again = D.Dataset(tmp_path).split((0.7, 0.2, 0.1), seed=3)
    assert {s: [i.rel for i in v] for s, v in again.items()} == {s: [i.rel for i in v] for s, v in out.items()}
    assert text == {s: (tmp_path / f"{s}.txt").read_text() for s in D.SPLITS}
    other = D.Dataset(tmp_path).split((0.7, 0.2, 0.1), seed=4)
    assert [i.rel for i in other["val"]] != [i.rel for i in out["val"]]
    y = (tmp_path / "data.yaml").read_text()
    assert 'train: "train.txt"' in y and 'val: "val.txt"' in y and 'test: "test.txt"' in y
    assert text["val"].splitlines()[0].startswith("./images/")
    reread = D.Dataset(tmp_path)                    # the split is read back from the lists
    assert sum(1 for it in reread.items if it.split == "test") == 10
    # the labelling tool keeps the split when it saves
    from visionflow.datasets import write_data_yaml
    write_data_yaml(tmp_path, ["a", "b", "c"], "images")
    y = (tmp_path / "data.yaml").read_text()
    assert 'val: "val.txt"' in y and '2: "c"' in y
    # labelled only, ratios that do not add up to 1, no test part
    out = D.Dataset(tmp_path).split((3, 1, 0), seed=0, labelled_only=True)
    assert [len(out[s]) for s in D.SPLITS] == [68, 22, 0] or [len(out[s]) for s in D.SPLITS] == [67, 23, 0]
    assert "test:" not in (tmp_path / "data.yaml").read_text()
    for bad in ((0, 1, 0), (-1, 1, 1), ("x", 1, 1), (0, 0, 0)):
        with pytest.raises(ValueError):
            D.Dataset(tmp_path).split(bad)


# --- conversion round trips -----------------------------------------------------------------------------------------

def test_coco_voc_yolo_round_trips_keep_boxes(tmp_path):
    gts, ds = build(tmp_path / "src", 5)
    # an odd, non-integer box too
    lp = tmp_path / "src" / "labels" / "r00.txt"
    lp.write_text(lp.read_text() + "\n1 0.333333 0.711111 0.123457 0.098765")
    ds.reload()
    ref = {it.image.name: px_boxes(ds, it) for it in ds.items}

    def same(ds2, tol):
        got = {it.image.name: px_boxes(ds2, it) for it in ds2.items}
        assert set(got) == set(ref)
        for k in ref:
            assert len(got[k]) == len(ref[k])
            for a, b in zip(got[k], ref[k]):
                assert a[0] == b[0] and max(abs(u - v) for u, v in zip(a[1:], b[1:])) <= tol, (k, a, b)
        assert ds2.classes == ds.classes

    js = D.export_coco(ds, tmp_path / "coco")
    doc = json.loads(js.read_text())
    assert [c["name"] for c in doc["categories"]] == ["plaka", "kare"] and len(doc["images"]) == 5
    a = next(a for a in doc["annotations"] if a["image_id"] == 1)
    assert a["category_id"] in (1, 2) and len(a["bbox"]) == 4
    same(D.import_coco(js, tmp_path / "from_coco"), 0.05)

    voc = D.export_voc(ds, tmp_path / "voc")
    x = ET.parse(next((voc / "Annotations").glob("r01.xml"))).getroot()
    assert x.findtext("size/width") == str(W) and x.find("object/bndbox/xmin") is not None
    same(D.import_voc(voc, tmp_path / "from_voc"), 1.0)

    same(D.Dataset(D.export_yolo(ds, tmp_path / "yolo")), 1e-3)
    # convert(): auto-detected formats, COCO → VOC through a temporary YOLO set
    same(D.Dataset(D.convert(tmp_path / "voc", tmp_path / "c1", "yolo")), 1.0)
    D.convert(tmp_path / "coco", tmp_path / "c2", "voc")
    same(D.import_voc(tmp_path / "c2", tmp_path / "c3"), 1.0)
    assert D.detect_format(tmp_path / "coco") == "coco" and D.detect_format(tmp_path / "voc") == "voc"
    assert D.detect_format(tmp_path / "src") == "yolo"
    with pytest.raises(ValueError, match="boş değil"):
        D.export_coco(ds, tmp_path / "coco")
    with pytest.raises(ValueError, match="aynı"):
        D.export_yolo(ds, tmp_path / "src")
    with pytest.raises(ValueError, match="biçim|Biçim"):
        D.convert(tmp_path / "src", tmp_path / "x", "pdf")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="COCO"):
        D.import_coco(tmp_path / "bad.json", tmp_path / "x2")


def test_export_yolo_with_split_and_voc_image_sets(tmp_path):
    gts, ds = build(tmp_path / "src", 10)
    ds.split((0.6, 0.2, 0.2), seed=1)
    out = D.export_yolo(D.Dataset(tmp_path / "src"), tmp_path / "y")
    assert len(list((out / "images" / "train").glob("*.png"))) == 6
    assert len(list((out / "labels" / "val").glob("*.txt"))) == 2
    y = (out / "data.yaml").read_text()
    assert 'train: "images/train"' in y and 'test: "images/test"' in y
    ds2 = D.Dataset(out)
    assert sorted(it.split for it in ds2.items).count("test") == 2
    voc = D.export_voc(D.Dataset(tmp_path / "src"), tmp_path / "v")
    assert len((voc / "ImageSets" / "Main" / "train.txt").read_text().split()) == 6


# --- augmentation ---------------------------------------------------------------------------------------------------

def _object_scene():
    img = np.full((120, 200, 3), 90, np.uint8)
    cv2.rectangle(img, (30, 20), (79, 49), (0, 0, 255), -1)            # pixels 30..79 x 20..49 → edges 30..80, 20..50
    cv2.rectangle(img, (120, 70), (169, 109), (0, 255, 0), -1)
    return img, [[0, 30, 20, 80, 50], [1, 120, 70, 170, 110]]


def _mask_box(img, colour):
    m = np.all(np.abs(img.astype(int) - colour) < 60, axis=2).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(m)
    return [x, y, x + w, y + h]


@pytest.mark.parametrize("ops", [{"flip_h": True}, {"flip_v": True}, {"rot90": 1}, {"rot90": 2}, {"rot90": 3},
                                 {"flip_h": True, "rot90": 1}, {"flip_v": True, "flip_h": True, "rot90": 3}])
def test_augment_flips_and_quarter_turns_are_exact(ops):
    img, boxes = _object_scene()
    out, bx = D.augment(img, boxes, ops)
    assert len(bx) == 2
    for b, col in zip(bx, ((0, 0, 255), (0, 255, 0))):
        assert b[1:] == pytest.approx(_mask_box(out, col), abs=1e-9)


@pytest.mark.parametrize("angle", [-12.0, -5.0, 3.0, 8.0])
def test_augment_small_rotation_boxes(angle):
    img, boxes = _object_scene()
    out, bx = D.augment(img, boxes, {"rotate": angle})
    assert out.shape == img.shape
    for b, col in zip(bx, ((0, 0, 255), (0, 255, 0))):
        assert max(abs(u - v) for u, v in zip(b[1:], _mask_box(out, col))) <= 1.5


def test_augment_photometric_keeps_boxes_and_rejects_bad_input():
    img, boxes = _object_scene()
    rng = np.random.default_rng(0)
    out, bx = D.augment(img, boxes, {"brightness": 40, "blur": 4, "noise": 8}, rng)
    assert [b[1:] for b in bx] == [b[1:] for b in boxes] and out.dtype == np.uint8
    assert abs(float(out.mean()) - float(img.mean()) - 40) < 3
    out16, _ = D.augment(img.astype(np.uint16) * 200, boxes, {"brightness": 500, "noise": 30}, rng)
    assert out16.dtype == np.uint16
    gray, gb = D.augment(img[:, :, 0], boxes, {"rot90": 1, "noise": 5}, rng)
    assert gray.shape == (200, 120) and len(gb) == 2
    # a box rotated out of the picture is dropped
    _, bx = D.augment(img, [[0, 0, 0, 4, 4]], {"rotate": 45})
    assert bx == []
    with pytest.raises(ValueError):
        D.augment(None, [], {})
    with pytest.raises(ValueError, match="işlem"):
        D.random_ops(rng, [])
    for _ in range(20):
        ops = D.random_ops(rng, ["rotate", "blur"], max_angle=5)
        assert ops and set(ops) <= {"rotate", "blur"} and abs(ops.get("rotate", 0)) <= 5


def test_augment_and_save_copies_with_correct_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img, boxes = _object_scene()
    D.imwrite(tmp_path / "images" / "s.png", img)
    (tmp_path / "labels" / "s.txt").write_text("\n".join(D.format_line(D.to_yolo(b, 200, 120)) for b in boxes))
    D.imwrite(tmp_path / "images" / "u.png", img)                       # unlabelled: not augmented
    (tmp_path / "classes.txt").write_text("k\ny")
    ds = D.Dataset(tmp_path)
    ds.split((1, 0, 0))
    n = D.augment_dataset(ds, ds.items, 4, ["flip_h", "flip_v", "rot90"], seed=5)
    assert n == 4 and len(ds.items) == 6
    for k in range(1, 5):
        it = ds.find(f"images/s_aug{k}.png")
        out = D.imread(it.image)
        h, w = out.shape[:2]
        got = sorted(D.to_xyxy(b, w, h) for b in ds.boxes(it))
        assert got[0][1:] == pytest.approx(_mask_box(out, (0, 0, 255)), abs=1e-3)
        assert got[1][1:] == pytest.approx(_mask_box(out, (0, 255, 0)), abs=1e-3)
        assert it.split == "train"                                      # added to train.txt
    with pytest.raises(ValueError):
        D.augment_dataset(ds, [ds.find("images/u.png")], 2, ["flip_h"])
    with pytest.raises(ValueError):
        D.augment_dataset(ds, ds.items, 0, ["flip_h"])


# --- duplicates -----------------------------------------------------------------------------------------------------

def test_perceptual_hash_finds_near_duplicates_only():
    a, _ = make(1)
    b, _ = make(2)
    near = [cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_COLOR),
            cv2.convertScaleAbs(a, alpha=1.0, beta=12), cv2.resize(a, (240, 180), interpolation=cv2.INTER_AREA)]
    ha = D.phash(a)
    for n in near:
        assert bin(ha ^ D.phash(n)).count("1") <= 4
    assert bin(ha ^ D.phash(b)).count("1") > 10
    assert bin(ha ^ D.phash(cv2.flip(a, 1))).count("1") > 10
    hashes = {"a": ha, "a2": D.phash(near[0]), "b": D.phash(b), "c": D.phash(make(3)[0]), "a3": D.phash(near[2])}
    assert D.duplicate_groups(hashes) == [["a", "a2", "a3"]]
    assert D.duplicate_groups({}) == [] and D.duplicate_groups({"x": 1}) == []
    assert D.phash(a[:, :, 0]) == D.phash(cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR))


# --- suggestions: accept / reject -----------------------------------------------------------------------------------

def test_suggestions_accept_reject_and_review_order(tmp_path):
    gts, ds = build(tmp_path, 5, labelled=1)
    it1, it2, it3, it4 = (ds.find(f"images/r{i:02d}.png") for i in (1, 2, 3, 4))
    ds.write_suggestions(it1, [(0, 0.5, 0.5, 0.2, 0.2, 0.9), (1, 0.2, 0.2, 0.1, 0.1, 0.4)])
    ds.write_suggestions(it2, [(0, 0.3, 0.3, 0.1, 0.1, 0.75)])
    ds.write_suggestions(it3, [])
    ds.write_suggestions(it4, [(1, 0.6, 0.6, 0.1, 0.1, 0.95)])
    assert not it1.label.exists()                                     # suggestions are not labels
    q = ds.review_queue()
    assert [t[0].rel for t in q] == ["images/r03.png", "images/r01.png", "images/r02.png", "images/r04.png"]
    assert [round(t[2], 2) for t in q] == [0.0, 0.4, 0.75, 0.95]
    assert ds.stats()["suggested"] == 4 and len(ds.filter("suggested")) == 4
    # accept one picture, only the sure box
    assert ds.accept(it1, min_conf=0.5) == 1
    assert not ds.suggest_path(it1).exists() and ds.boxes(it1) == [(0, 0.5, 0.5, 0.2, 0.2)]
    # accept a suggestion that repeats an existing box: not added twice
    ds.write_suggestions(it1, [(0, 0.501, 0.5, 0.2, 0.2, 0.9), (1, 0.8, 0.8, 0.1, 0.1, 0.9)])
    assert ds.accept(it1) == 1 and len(ds.boxes(it1)) == 2
    ds.reject(it2)
    assert not ds.suggest_path(it2).exists() and not it2.label.exists()
    # "no objects" accepted → an empty label file (a labelled negative picture)
    assert ds.accept(it3) == 0 and it3.label.exists() and ds.boxes(it3) == []
    assert ds.accept_all() == (1, 1) and len(ds.boxes(it4)) == 1
    ds.write_suggestions(it2, [(0, 0.3, 0.3, 0.1, 0.1, 0.7)])
    assert ds.reject_all() == 1 and ds.review_queue() == []
    # an existing polygon line survives accepting
    it0 = ds.find("images/r00.png")
    it0.label.write_text(it0.label.read_text() + "\n0 0.1 0.1 0.2 0.2 0.3 0.3")
    ds.write_suggestions(it0, [(1, 0.9, 0.9, 0.05, 0.05, 0.8)])
    ds.accept(it0)
    assert "0 0.1 0.1 0.2 0.2 0.3 0.3" in it0.label.read_text()


def test_autolabel_with_fewshot_model_matches_ground_truth(tmp_path):
    gts, ds = build(tmp_path, 9, labelled=3)
    model = D.train_fewshot(ds, angle_range=10, scale_range=(0.9, 1.1))
    assert model.exists() and model.suffix == ".vfdet"
    res = D.autolabel(ds, model, conf=0.5)
    assert res["images"] == 6 and res["device"] == "CPU"
    ious = []
    for it in ds.items:
        stem = it.image.stem
        if int(stem[1:]) < 3:
            assert not ds.suggest_path(it).exists()                     # labelled pictures are never touched
            continue
        sug = [D.to_xyxy(b, W, H) for b in ds.suggestions(it)]
        assert len(sug) == len(gts[stem])                               # no extra, no missing objects
        ious += [best_iou(g, sug) for g in gts[stem]]
    assert min(ious) >= 0.8 and np.mean(ious) >= 0.85, ious
    # the queue has every unlabelled picture, least sure first; accepting all turns them into labels
    q = ds.review_queue()
    assert len(q) == 6 and [t[2] for t in q] == sorted(t[2] for t in q)
    ds.accept_all()
    assert ds.stats()["labelled"] == 9
    # a second run has nothing left to do
    assert D.autolabel(ds, model)["images"] == 0


def test_fewshot_training_and_predictor_errors(tmp_path):
    _, ds = build(tmp_path, 3, labelled=0)
    with pytest.raises(ValueError, match="etiketleyin"):
        D.train_fewshot(ds)
    with pytest.raises(ValueError, match="yok"):
        D.Predictor(tmp_path / "yok.vfdet")
    (tmp_path / "m.txt").write_text("x")
    with pytest.raises(ValueError, match=".vfdet"):
        D.Predictor(tmp_path / "m.txt")
    (tmp_path / "m.vfdet").write_bytes(b"junk")
    with pytest.raises(ValueError):
        D.Predictor(tmp_path / "m.vfdet")
    (tmp_path / "m.onnx").write_bytes(b"junk")
    with pytest.raises(ValueError, match="YOLO"):
        D.Predictor(tmp_path / "m.onnx")


def test_autolabel_with_yolo_onnx_uses_the_yolo_block(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    raw = np.zeros((1, 84, 8400), np.float32)
    raw[0, :4, 0] = (320, 320, 100, 100)
    raw[0, 4 + 2, 0] = 0.95                                     # class 2 = car
    raw[0, :4, 1] = (100, 100, 40, 40)
    raw[0, 4 + 0, 1] = 0.3                                      # under the threshold
    const = helper.make_node("Constant", [], ["output0"], value=helper.make_tensor(
        "v", TensorProto.FLOAT, raw.shape, raw.flatten().tolist()))
    graph = helper.make_graph([const], "fake_yolo",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "fake.onnx"
    onnx.save(model, path)
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    D.imwrite(root / "images" / "a.png", np.full((640, 640, 3), 100, np.uint8))
    ds = D.Dataset(root)
    from visionflow import accel
    pred = D.Predictor(path, 0.5)
    assert pred.kind == "yolo" and pred.node.values["device"] == accel.GLOBAL and pred.device
    pred.close()
    res = D.autolabel(ds, path, 0.5)
    assert res["boxes"] == 1 and ds.classes == ["car"]
    (b,) = [D.to_xyxy(s, 640, 640) for s in ds.suggestions(ds.items[0])]
    assert b[1:] == pytest.approx([270, 270, 370, 370], abs=1)


# --- command line ---------------------------------------------------------------------------------------------------

def test_cli_commands(tmp_path, capsys):
    gts, ds = build(tmp_path / "d", 8, labelled=3)
    assert D.cli(["istatistik", str(tmp_path / "d")]) == 0
    out = capsys.readouterr().out
    assert "Resim: 8" in out and "plaka" in out and "Sorun bulunmadı" in out
    assert D.cli(["istatistik", str(tmp_path / "d"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["labelled"] == 3
    assert D.cli(["bol", str(tmp_path / "d"), "--oran", "0.5,0.25,0.25", "--tohum", "2"]) == 0
    assert "train=4" in capsys.readouterr().out
    assert D.cli(["donustur", str(tmp_path / "d"), str(tmp_path / "coco"), "--bicim", "coco"]) == 0
    assert (tmp_path / "coco" / "annotations.json").exists()
    assert D.cli(["etiketle", str(tmp_path / "d"), "--egit", "--esik", "0.5", "--kabul"]) == 0
    out = capsys.readouterr().out
    assert "Kabul edildi: 5 resim" in out
    assert D.Dataset(tmp_path / "d").stats()["labelled"] == 8
    # errors: Turkish message, exit code, no traceback
    assert D.cli(["istatistik", str(tmp_path / "yok")]) == 1
    assert "HATA" in capsys.readouterr().err
    assert D.cli(["etiketle", str(tmp_path / "d")]) == 1
    assert D.cli(["bol", str(tmp_path / "d"), "--oran", "a,b"]) == 1
    assert D.cli(["donustur", str(tmp_path / "d"), str(tmp_path / "coco"), "--bicim", "coco"]) == 1
    assert D.cli(["uc"]) == 2
    assert D.cli(["istatistik"]) == 2                       # argparse: missing folder
    assert D.cli([]) == 0 and "veriseti" in capsys.readouterr().out


def test_cli_through_python_m_visionflow(tmp_path):
    build(tmp_path / "d", 3)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "visionflow", "veriseti", "istatistik", str(tmp_path / "d")],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert r.returncode == 0 and "Resim: 3" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, "-m", "visionflow", "--help"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert "veriseti" in r.stdout


# --- labelling tool integration -------------------------------------------------------------------------------------

@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_labeling_tool_opens_split_folders_and_keeps_the_split(tmp_path, app):
    from visionflow.ui.tools import Dataset as ToolDataset, LabelingDialog
    gts, ds = build(tmp_path, 6)
    ds.split((0.5, 0.5, 0), seed=0)
    tds = ToolDataset(tmp_path)
    tds.save_meta()
    assert 'val: "val.txt"' in (tmp_path / "data.yaml").read_text()
    # images/train layout opened at the split folder
    out = D.export_yolo(D.Dataset(tmp_path), tmp_path / "y")
    tds = ToolDataset(out / "images" / "val")
    assert tds.root == out and tds.lbl_dir == out / "labels" / "val" and tds.classes == ["plaka", "kare"]
    img = tds.images[0]
    assert len(tds.load(img, W, H)) == len(gts[img.stem])
    dlg = LabelingDialog(folder=str(out / "images" / "val"), image=str(img))
    assert dlg.ds.images[dlg.index] == img and len(dlg.canvas.boxes) == len(gts[img.stem])
    dlg.close()
    assert 'val: "images/val"' in (out / "data.yaml").read_text()


# --- the window -----------------------------------------------------------------------------------------------------

def test_window_end_to_end(tmp_path, app):
    from visionflow.ui.dataset_manager import DatasetManager
    gts, ds = build(tmp_path / "d", 8, labelled=3)
    D.imwrite(tmp_path / "d" / "images" / "r00_kopya.png", D.imread(tmp_path / "d" / "images" / "r00.png"))
    dlg = DatasetManager()
    dlg.quiet = True
    assert dlg.open_dataset(str(tmp_path / "d"))
    assert "9</b> resim" in dlg.summary.text() and "Kopya resim 1" in dlg.summary.text()
    assert dlg.chart.names == ["plaka", "kare"] and sum(dlg.chart.counts) == sum(len(gts[f"r{i:02d}"])
                                                                                   for i in range(3))
    assert dlg.problems.count() == 1
    dlg.chart.resize(400, 200)
    dlg.chart.grab()                                                    # QPainter paint runs
    assert dlg.grid.count() == 9
    dlg.filter_combo.setCurrentIndex(dlg.filter_combo.findData("unlabelled"))
    assert dlg.grid.count() == 6
    dlg.filter_combo.setCurrentIndex(dlg.filter_combo.findData("problems"))
    assert dlg.grid.count() == 1
    dlg.filter_combo.setCurrentIndex(dlg.filter_combo.findData("class:1"))
    assert 0 < dlg.grid.count() <= 3
    for _ in range(5):
        dlg._load_some_thumbs()
    assert not dlg.grid.item(0).icon().isNull()
    # quick open in the labelling tool
    lab = dlg.open_in_labeler("images/r02.png")
    assert lab.ds.images[lab.index].name == "r02.png"
    lab.close()
    # classes
    dlg.class_list.setCurrentRow(1)
    assert dlg.rename_class(name="kutu") is not None and dlg.ds.classes == ["plaka", "kutu"]
    assert dlg.class_list.count() == 2 and "kutu" in dlg.class_list.item(1).text()
    assert dlg.rename_class(0, "kutu") is None and "zaten var" in dlg.last_message
    dlg.add_class("vida")
    assert dlg.ds.classes == ["plaka", "kutu", "vida"]
    dlg.delete_class(2)
    assert dlg.ds.classes == ["plaka", "kutu"]
    # split, export
    out = dlg.do_split((0.5, 0.25, 0.25), 1)
    assert sum(len(v) for v in out.values()) == 9 and "Bölündü" in dlg.split_info.text()
    assert dlg.export("coco", str(tmp_path / "coco")) is not None
    assert dlg.export("coco", str(tmp_path / "coco")) is None and "boş değil" in dlg.last_message
    # augmentation preview
    ops, boxes = dlg.preview_augment("images/r01.png")
    assert ops and dlg.aug_after.pixmap is not None and dlg.aug_before.pixmap is not None
    for cb in dlg.aug_checks.values():
        cb.setChecked(False)
    assert dlg.preview_augment("images/r01.png") is None and "işlem" in dlg.aug_info.text()
    # auto-label: train few-shot from the 3 labelled pictures, review, accept / reject
    dlg.fs_angle.setValue(10)
    res = dlg.train_and_label(0.5)
    assert res["images"] == 6 and dlg.model_edit.text().endswith(".vfdet")
    assert dlg.review.count() == 6 and dlg.review_view.pixmap is not None
    first = dlg._review_rel()
    assert dlg.accept_current() >= 1 and dlg.ds.find(first).label.exists()
    assert dlg.review.count() == 5
    second = dlg._review_rel()
    dlg.reject_current()
    assert dlg.review.count() == 4 and not dlg.ds.find(second).label.exists()
    dlg.accept_all(0.0)
    assert dlg.review.count() == 0 and dlg.st["labelled"] == 8
    assert dlg.run_autolabel(str(tmp_path / "yok.onnx")) is None and "yok" in dlg.last_message
    # augment and save from the window (train part only)
    dlg.aug_checks["flip_h"].setChecked(True)
    n = dlg.augment_and_save(copies=1, seed=0)
    assert n == sum(1 for it in dlg.ds.items if it.split == "train" and "_aug" not in it.image.stem
                    and it.label.exists()) and n > 0
    # restore the first backup (before the rename)
    assert dlg.restore_backup(str(sorted(dlg.ds.backups())[0])) > 0 and dlg.ds.classes == ["plaka", "kare"]
    dlg.close()


def test_window_without_dataset_and_new_dataset(tmp_path, app):
    from visionflow.ui.dataset_manager import DatasetManager
    dlg = DatasetManager()
    dlg.quiet = True
    assert dlg.do_split() is None and "Önce" in dlg.last_message
    assert dlg.open_dataset(str(tmp_path / "yok")) is False and "yok" in dlg.last_message
    assert dlg.new_dataset(str(tmp_path / "yeni"), ["a", "b"])
    assert (tmp_path / "yeni" / "images").is_dir() and dlg.ds.classes == ["a", "b"]
    src = tmp_path / "x.png"
    D.imwrite(src, make(0)[0])
    assert dlg.add_images([str(src), str(src)]) == 2 and dlg.grid.count() == 2
    # import COCO into a new dataset
    gts, ds = build(tmp_path / "s", 2)
    js = D.export_coco(ds, tmp_path / "coco")
    assert dlg.import_dataset("coco", str(js), str(tmp_path / "imp"))
    assert dlg.ds.root == tmp_path / "imp" and dlg.st["boxes"] == sum(len(g) for g in gts.values())
    dlg.close()


def test_menu_entry(app):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    try:
        acts = [a.text() for m in w.menuBar().actions() if m.menu() for a in m.menu().actions()]
        assert any("Veri seti yöneticisi" in t for t in acts)
        dlg = w.open_dataset_manager()
        assert dlg.isVisible()
        dlg.close()
    finally:
        w.stop()
        w.close()
        app.processEvents()
