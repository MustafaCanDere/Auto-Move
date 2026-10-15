"""v2.2 (d): template gallery ('Yeni akış' wizard), new templates and practical inspection blocks."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

import cv2
import numpy as np
import pytest

from visionflow.core import registry
from visionflow.core.graph import Graph
import visionflow.nodes  # noqa: F401
from visionflow import templates as T
from visionflow.nodes.inspection_more import _ocr_fix, find_dates, parse_color_classes, parse_count_spec

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW_TEMPLATES = ["etiket_var_yok_kontrolu", "kapak_conta_kontrolu", "renk_ile_ayiklama_plc", "tarih_lot_kodu_okuma",
                 "kit_eksik_parca_alarmi", "boyut_olcum_tolerans_rapor", "yuzey_cizik_leke_kontrolu",
                 "plaka_hizalama_kontrolu"]


def mk(type_name, **values):
    n = registry.create(type_name)
    n.values.update(values)
    assert n.safe_setup(), n.setup_error
    return n


def run(n, **inputs):
    full = {name: None for name, _ in n.inputs}
    full.update(inputs)
    return n.process(full)


# --- Etiket Var mı --------------------------------------------------------------------------------------------------

def label_img(dx=0, angle=0.0, present=True, colour=(245, 245, 245), box=(95, 130, 165)):
    img = np.full((300, 400, 3), 50, np.uint8)
    cv2.rectangle(img, (60, 40), (340, 260), box, -1)
    if present:
        pts = cv2.boxPoints(((200 + dx, 150), (160, 80), -angle)).astype(np.int32)
        cv2.fillPoly(img, [pts], colour)
    return img


def test_label_presence_by_colour():
    n = mk("LabelPresence", roi="60,40,280,220", threshold=215, max_offset=15, max_angle=4)
    o = run(n, image=label_img())
    assert o["found"] and o["ok"] and o["offset"] < 2 and abs(o["angle"]) < 1
    assert o["image"].shape == (300, 400, 3)
    o = run(n, image=label_img(present=False))
    assert not o["found"] and not o["ok"]
    o = run(n, image=label_img(dx=40))
    assert o["found"] and not o["ok"] and abs(o["dx"] - 40) < 2 and abs(o["dy"]) < 2
    o = run(n, image=label_img(angle=10))          # turned 10° counter-clockwise on screen
    assert o["found"] and not o["ok"] and abs(o["angle"] - 10) < 1.5
    o = run(n, image=label_img(angle=-10))
    assert abs(o["angle"] + 10) < 1.5
    n.values.update(expected_x=240, expected_y=150)     # a fixed expected centre
    o = run(n, image=label_img(dx=40))
    assert o["ok"] and o["offset"] < 2


def test_label_presence_dark_hsv_and_inputs():
    n = mk("LabelPresence", method="koyu etiket", threshold=60, roi="60,40,280,220")
    assert run(n, image=label_img(colour=(20, 20, 20)))["ok"]
    assert not run(n, image=label_img(present=False))["found"]
    n = mk("LabelPresence", method="renk aralığı (HSV)", h_min=100, h_max=130, s_min=100, roi="60,40,280,220")
    assert run(n, image=label_img(colour=(200, 60, 20)))["ok"]          # blue label
    assert not run(n, image=label_img(colour=(20, 60, 200)))["found"]   # red one is not it
    n = mk("LabelPresence", threshold=215, roi="60,40,280,220")
    img = label_img()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert run(n, image=gray)["ok"]
    assert run(n, image=gray.astype(np.uint16) * 257)["ok"]
    assert run(n, image=gray.astype(np.float32) / 255.0)["ok"]
    assert run(n, image=cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))["ok"]
    with pytest.raises(ValueError, match="bağlı değil"):
        run(n, image=None)
    with pytest.raises(ValueError, match="resim"):
        run(n, image="metin")
    n.values["roi"] = "1000,1000,10,10"
    with pytest.raises(ValueError, match="dışında"):
        run(n, image=img)


def test_label_presence_by_template():
    ref = np.full((60, 120, 3), 255, np.uint8)
    cv2.putText(ref, "LOT42", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.rectangle(ref, (0, 0), (119, 59), (0, 0, 0), 2)
    for angle, shift in ((0, 0), (6, 0), (-6, 25)):
        scene = np.full((300, 400, 3), 90, np.uint8)
        M = cv2.getRotationMatrix2D((60, 30), angle, 1.0)
        M[:, 2] += (200 + shift - 60, 150 - 30)
        scene = cv2.warpAffine(ref, M, (400, 300), dst=scene, borderMode=cv2.BORDER_TRANSPARENT)
        n = mk("LabelPresence", method="şablon (referans resim)", search_angle=10, max_angle=3, max_offset=10)
        o = run(n, image=scene, reference=ref)
        assert o["found"] and o["score"] > 0.6, o
        assert abs(o["angle"] - angle) <= 1.0 and abs(o["dx"] - shift) <= 2 and abs(o["dy"]) <= 2, o
        assert o["ok"] == (angle == 0 and shift == 0)
    blank = np.full((300, 400, 3), 90, np.uint8)
    assert not run(n, image=blank, reference=ref)["found"]
    with pytest.raises(ValueError, match="büyük"):
        run(n, image=blank[:40, :40], reference=ref)
    with pytest.raises(ValueError, match="Referans resim yok"):
        run(n, image=blank)


# --- Tarih Kodu Doğrula ---------------------------------------------------------------------------------------------

def date_node(now=dt.datetime(2026, 10, 14, 10, 0), **values):
    n = mk("DateCodeCheck", **values)
    n.now = lambda: now
    return n


@pytest.mark.parametrize("text,iso", [("12.10.2026", "2026-10-12"), ("2026-10-12", "2026-10-12"),
                                      ("12OCT26", "2026-10-12"), ("12 EKİ 2026", "2026-10-12"),
                                      ("12/10/26", "2026-10-12"), ("12-oca-2027", "2027-01-12"),
                                      ("TETT 05 MAYIS 2027", "2027-05-05"), ("12 Ekim 2026", "2026-10-12"), ("10/2026", "2026-10-31"),
                                      ("2026.02.03", "2026-02-03")])
def test_date_formats(text, iso):
    assert [d.isoformat() for _, d, _ in find_dates(text) if d] == [iso]


def test_date_code_checks():
    n = date_node()
    o = run(n, text="ÜRT: 14.10.2026 SKT: 14.10.2027 LOT A12")
    assert o["ok"] and o["date"] == "2027-10-14" and o["days_left"] == 365      # the latest date by default
    n.values["pick"] = "ilk tarih"
    assert run(n, text="ÜRT: 14.10.2026 SKT: 14.10.2027")["date"] == "2026-10-14"
    o = run(date_node(), text="SKT 01.03.2025")
    assert not o["ok"] and o["found"] and o["reason"] == "süresi geçmiş" and o["days_left"] < 0
    o = run(date_node(min_days=30), text="SKT 01.11.2026")
    assert not o["ok"] and "18 gün" in o["reason"]
    o = run(date_node(keyword="SKT", pick="en geç tarih"), text="SKT: 01.11.2026 ÜRT 14.10.2030")
    assert o["date"] == "2026-11-01"                    # the date after the keyword, not the latest one
    o = run(date_node(keyword="EXP"), text="SKT: 01.11.2026")
    assert not o["found"] and "EXP" in o["reason"]


def test_date_code_today_shift_and_offset():
    n = date_node(check="bugün olmalı (üretim)")
    assert run(n, text="14.10.2026")["ok"]
    assert not run(n, text="13.10.2026")["ok"]
    night = date_node(now=dt.datetime(2026, 10, 14, 3, 0), check="bugün olmalı (üretim)", shift_start="06:00")
    assert run(night, text="13.10.2026")["ok"]          # 03:00 still belongs to the night shift of the 13th
    assert not run(night, text="14.10.2026")["ok"]
    n = date_node(check="bugün + N gün olmalı", offset_days=365, tolerance_days=1)
    assert run(n, text="SKT 15.10.2027")["ok"]
    o = run(n, text="SKT 20.10.2027")
    assert not o["ok"] and "olmalı" in o["reason"]
    assert run(date_node(check="sadece okunsun"), text="01.01.2000")["ok"]
    with pytest.raises(ValueError, match="SS:DD"):
        run(date_node(shift_start="25:00"), text="14.10.2026")


def test_date_code_bad_text_and_ocr_mistakes():
    n = date_node()
    for t in (None, "", "   "):
        o = run(n, text=t)
        assert not o["found"] and "metin yok" in o["reason"]
    assert run(n, text="LOT 1234")["reason"] == "tarih bulunamadı"
    assert not run(n, text=12345)["found"]                      # not text: no crash
    o = run(n, text="31.02.2027")
    assert not o["found"] and "geçersiz" in o["reason"]
    o = run(n, text="SKT 1O.l2.2O27")
    assert o["date"] == "2027-12-10"
    assert run(n, text="SKT: 140CT27")["date"] == "2027-10-14"
    assert _ocr_fix("12 0CAK 2027") == "12 OCAK 2027"
    n.values["fix_ocr"] = False
    assert not run(n, text="SKT 1O.l2.2O27")["found"]
    n.values["format"] = "YYYY-AA-GG"
    assert not run(n, text="12.10.2027")["found"]
    assert run(n, text="2027-10-12")["found"]


# --- Renk ile Ayıkla ------------------------------------------------------------------------------------------------

def disc(col, bg=20, r=60):
    img = np.full((200, 200, 3), bg, np.uint8)
    cv2.circle(img, (100, 100), r, col, -1)
    return img


def test_color_sort_classes():
    n = mk("ColorSort", roi="70,70,60,60")
    for col, name, idx in (((30, 30, 210), "kırmızı", 1), ((60, 165, 35), "yeşil", 2), ((210, 85, 30), "mavi", 3),
                           ((30, 210, 225), "sarı", 4)):
        o = run(n, image=disc(col))
        assert (o["class"], o["index"], o["ok"]) == (name, idx, True), (col, o)
        assert o["distance"] < 30 and len(o["lab"]) == 3
    o = run(n, image=disc((128, 128, 128)))
    assert o["class"] == "bilinmiyor" and o["index"] == 0 and not o["ok"]
    n.values.update(roi="", background="koyu zemini yok say", bg_level=60)     # small part, dark belt ignored
    assert run(n, image=disc((30, 30, 210), r=30))["class"] == "kırmızı"
    n.values["background"] = "yok sayma"
    assert run(n, image=disc((30, 30, 210), r=30))["class"] == "bilinmiyor"     # the belt is the dominant colour
    n.values.update(background="koyu zemini yok say", bg_level=255)
    assert run(n, image=disc((30, 30, 210)))["index"] == 0                       # everything is "background"
    gray = mk("ColorSort", classes="beyaz=100,0,0; siyah=0,0,0")
    assert run(gray, image=np.full((50, 50), 255, np.uint8))["class"] == "beyaz"
    assert run(gray, image=np.zeros((50, 50), np.uint16))["class"] == "siyah"
    for stat in ("ortalama", "medyan"):
        gray.values["stat"] = stat
        assert run(gray, image=np.full((50, 50, 4), 250, np.uint8))["class"] == "beyaz"


def test_color_sort_bad_spec():
    assert [n for n, _ in parse_color_classes("a=#ff0000; b=50,0,0\n c = 0,0,0")] == ["a", "b", "c"]
    for bad, msg in (("", "tanımlı değil"), ("kırmızı", "ad=#rrggbb"), ("=#ff0000", "boş"), ("a=1,2", "3 sayı"),
                     ("a=#ggg000", "3 sayı")):
        with pytest.raises(ValueError, match=msg):
            parse_color_classes(bad)
    n = mk("ColorSort", classes="xx")
    with pytest.raises(ValueError):
        run(n, image=disc((0, 0, 255)))


# --- Kenar Kusuru Bul -----------------------------------------------------------------------------------------------

def part(chip=False, burr=False, circle=False, dark_on_light=False):
    bg, fg = (230, 30) if dark_on_light else (30, 220)
    img = np.full((300, 300, 3), bg, np.uint8)
    if circle:
        cv2.circle(img, (150, 150), 90, (fg,) * 3, -1)
    else:
        cv2.rectangle(img, (60, 60), (240, 240), (fg,) * 3, -1)
    if chip:
        cv2.circle(img, (150, 60), 12, (bg,) * 3, -1)
    if burr:
        cv2.rectangle(img, (240, 140), (250, 150), (fg,) * 3, -1)
    return img


@pytest.mark.parametrize("mode", ["dışbükey zarf (çentik, kırık)", "çentik + çapak (dışa taşan)"])
def test_edge_defect_chips(mode):
    n = mk("EdgeDefect", mode=mode)
    for kw in ({}, {"circle": True}, {"dark_on_light": True}):
        o = run(n, image=part(**kw))
        assert o["found"] and o["ok"] and o["count"] == 0, (kw, o["detections"])
    for kw in ({"chip": True}, {"chip": True, "circle": True}, {"chip": True, "dark_on_light": True}):
        o = run(n, image=part(**kw))
        assert not o["ok"] and o["count"] == 1 and 8 <= o["max_depth"] <= 13, (kw, o["detections"])
        cx, cy = o["detections"][0]["center"]
        assert abs(cx - 150) < 10 and 60 <= cy <= 75
        assert o["image"].shape == (300, 300, 3)


def test_edge_defect_burrs_only_in_burr_mode():
    hull = mk("EdgeDefect")
    burr = mk("EdgeDefect", mode="çentik + çapak (dışa taşan)")
    o = run(burr, image=part(burr=True))
    assert o["count"] == 1 and o["detections"][0]["kind"] == "çapak" and o["max_depth"] >= 6
    o = run(burr, image=part(burr=True, chip=True))
    assert sorted(d["kind"] for d in o["detections"]) == ["çapak", "çentik"]
    o = run(burr, image=part(burr=True, circle=True))
    assert [d["kind"] for d in o["detections"]] == ["çapak"]
    assert all(d["kind"] == "çentik" for d in run(hull, image=part(burr=True))["detections"])
    burr.values["smooth"] = 31                       # bigger smoothing: corners still are no burrs
    assert run(burr, image=part())["count"] == 0
    burr.values["min_depth"] = 20
    assert run(burr, image=part(burr=True, chip=True))["count"] == 0


def test_edge_defect_no_part_and_inputs():
    n = mk("EdgeDefect")
    o = run(n, image=np.full((100, 100, 3), 40, np.uint8))
    assert not o["found"] and not o["ok"] and o["count"] == 0
    g = cv2.cvtColor(part(chip=True), cv2.COLOR_BGR2GRAY)
    assert run(n, image=g)["count"] == 1
    assert run(n, image=g.astype(np.uint16) * 257)["count"] == 1
    n.values.update(roi="40,40,220,220", polarity="açık parça / koyu zemin", threshold=100)
    assert run(n, image=part(chip=True))["count"] == 1
    with pytest.raises(ValueError):
        run(n, image=None)


# --- Doku / Yüzey Kusuru --------------------------------------------------------------------------------------------

def surface(scratch=False, stain=False, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    g = 140 + rng.normal(0, 5, (300, 400)) + np.linspace(-30, 30, 400)[None, :]
    img = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if scratch:
        cv2.line(img, (60, 60), (260, 220), (60, 60, 60), 1)
    if stain:
        cv2.circle(img, (320, 230), 18, (105, 105, 105), -1)
    if dtype == np.uint16:
        img = img.astype(np.uint16) * 257
    return img


def test_surface_defect_finds_scratch_and_stain():
    n = mk("SurfaceDefect")
    for seed in range(3):
        assert run(n, image=surface(seed=seed))["ok"]
    o = run(n, image=surface(scratch=True))
    assert not o["ok"] and o["count"] == 1 and o["detections"][0]["kind"] == "çizik"
    o = run(n, image=surface(stain=True))
    assert o["count"] == 1 and o["detections"][0]["kind"] == "leke" and o["area"] > 500
    o = run(n, image=surface(scratch=True, stain=True))
    assert o["count"] == 2 and o["mask"].shape == (300, 400) and o["mask"].max() == 255
    assert run(n, image=surface(stain=True, dtype=np.uint16))["count"] == 1
    n.values["roi"] = "0,0,200,300"                  # the stain is outside the region
    assert run(n, image=surface(stain=True))["ok"]
    n.values.update(roi="", method="çizik (Laplace)")
    assert run(n, image=surface(scratch=True))["count"] >= 1
    n.values["method"] = "leke (arka plan farkı)"
    assert run(n, image=surface(stain=True))["count"] == 1
    n.values.update(method="hepsi", min_area=5000)
    assert run(n, image=surface(stain=True))["ok"]


def test_surface_defect_sensitivity_and_edge_cases():
    weak = surface()
    cv2.circle(weak, (200, 150), 15, (125, 125, 125), -1)       # a faint stain
    low, high = mk("SurfaceDefect", sensitivity=1), mk("SurfaceDefect", sensitivity=90)
    assert run(low, image=weak)["ok"] and not run(high, image=weak)["ok"]
    flat = np.full((60, 60, 3), 128, np.uint8)
    assert run(high, image=flat)["ok"]                          # perfectly flat: no noise taken for defects
    assert run(mk("SurfaceDefect", window=255), image=surface()[:20, :20])["ok"]   # window bigger than the region
    with pytest.raises(ValueError):
        run(low, image=None)


# --- Hizalama Kontrolü ----------------------------------------------------------------------------------------------

def holes(p1, p2):
    return [{"box": (p1[0] - 5, p1[1] - 5, 10, 10), "score": 0.9, "label": "delik"},
            {"box": (p2[0] - 5, p2[1] - 5, 10, 10), "score": 0.8, "label": "delik"}]


def test_alignment_two_features():
    n = mk("AlignmentCheck", nominal_x=200, nominal_y=150, angle_tol=1, offset_tol=3)
    o = run(n, detections=holes((100, 150), (300, 150)))
    assert o["ok"] and o["found"] and o["angle"] == 0 and o["offset"] == 0
    a = np.radians(3)
    o = run(n, detections=holes((200 - 100 * np.cos(a), 150 + 100 * np.sin(a)), (200 + 100 * np.cos(a),
                                                                               150 - 100 * np.sin(a))))
    assert not o["ok"] and abs(o["angle"] - 3) < 0.01 and abs(o["angle_error"] - 3) < 0.01
    o = run(n, detections=holes((300, 160), (100, 160)), px_per_mm=2.0)     # order of the list does not matter
    assert o["angle"] == 0 and o["dy"] == 5 and o["offset"] == 5 and not o["ok"]
    n.values["offset_tol"] = 6
    assert run(n, detections=holes((300, 160), (100, 160)), px_per_mm=2.0)["ok"]
    n.values.update(nominal_x=-1, nominal_y=-1)
    assert run(n, detections=holes((10, 10), (110, 10)))["offset"] == 0          # position not checked
    n.values["nominal_angle"] = 90
    n.values["pick"] = "yukarıdan ilk iki"
    o = run(n, detections=holes((100, 200), (100, 100)))
    assert abs(o["angle_error"]) < 0.01                       # a vertical part, 90° expected
    dets = holes((100, 150), (300, 150)) + [{"box": (0, 0, 2, 2), "label": "vida"}]
    n.values.update(label="delik", nominal_angle=0, pick="soldan ilk iki")
    assert run(n, detections=dets)["angle"] == 0


def test_alignment_line_and_bad_input():
    n = mk("AlignmentCheck", mode="en uzun çizgi", angle_tol=1)
    lines = [{"box": (0, 0, 1, 1), "line": (10, 100, 50, 60)}, {"box": (0, 0, 1, 1), "line": (0, 100, 300, 95)}]
    o = run(n, detections=lines, image=np.zeros((200, 320, 3), np.uint8))
    assert abs(o["angle"] - 0.955) < 0.01 and o["ok"] and o["image"].shape == (200, 320, 3)
    o = run(n, detections=[{"box": (0, 0, 1, 1), "line": (0, 100, 300, 120)}])
    assert not o["ok"] and o["angle"] < -3
    assert not run(n, detections=holes((0, 0), (5, 5)))["found"]                 # no lines
    n.values["mode"] = "iki tespitin merkezleri"
    for dets in (None, [], holes((0, 0), (5, 5))[:1]):
        o = run(n, detections=dets, image=np.zeros((50, 50), np.uint8))
        assert not o["found"] and not o["ok"]
    with pytest.raises(ValueError, match="aynı yerde"):
        run(n, detections=holes((5, 5), (5, 5)))
    with pytest.raises(ValueError, match="tespit listesi"):
        run(n, detections="iki delik")


# --- Sayaç Karşılaştır ----------------------------------------------------------------------------------------------

def labels(*names):
    return [{"box": (0, 0, 1, 1), "label": n, "score": 1.0} for n in names]


def test_count_compare_spec_and_detections():
    assert parse_count_spec("3 vida, 2 somun") == {"vida": 3, "somun": 2}
    assert parse_count_spec("Vida=3; somun: 2\npul x4; 2 adet conta; 5x kapak") == \
        {"vida": 3, "somun": 2, "pul": 4, "conta": 2, "kapak": 5}
    assert parse_count_spec("1 vida, 2 vida") == {"vida": 3}
    for bad in ("", "vida", "üç vida", ",,"):
        with pytest.raises(ValueError):
            parse_count_spec(bad)
    n = mk("CountCompare", spec="3 vida, 2 somun")
    o = run(n, detections=labels("vida", "Vida #3", "vida 0.91", "somun", "somun"))
    assert o["ok"] and o["missing"] == "tamam" and o["summary"] == "3/3 vida, 2/2 somun"
    o = run(n, detections=labels("vida", "somun"))
    assert not o["ok"] and o["missing"] == "2 vida eksik, 1 somun eksik" and o["missing_count"] == 3
    o = run(n, detections=labels("vida", "vida", "vida", "vida", "somun", "somun", "pul"))
    assert not o["ok"] and o["missing"] == "1 vida fazla" and o["extra_count"] == 1
    n.values["mode"] = "en az bu kadar"
    assert run(n, detections=labels("vida", "vida", "vida", "vida", "somun", "somun", "pul"))["ok"]
    n.values["others"] = True
    o = run(n, detections=labels("vida", "vida", "vida", "somun", "somun", "pul"))
    assert not o["ok"] and o["missing"] == "1 pul fazla"
    o = run(n, detections=None)
    assert not o["ok"] and o["missing_count"] == 5


def test_count_compare_counts_input():
    n = mk("CountCompare", spec="2 daire, 1 altıgen")
    assert run(n, counts={"daire": 2, "altıgen": 1})["ok"]
    assert run(n, counts="daire=2, altıgen=1")["ok"]
    assert run(n, counts={"daire": 1}, detections=labels("daire", "altıgen"))["ok"]     # both added up
    assert not run(n, counts={"daire": "çok", "altıgen": None})["ok"]                   # odd values ignored
    with pytest.raises(ValueError, match="sözlüğü"):
        run(n, counts=[1, 2])
    with pytest.raises(ValueError, match="tespit listesi"):
        run(n, detections="vida")
    n.values["spec"] = "bir sürü vida"
    with pytest.raises(ValueError, match="anlaşılmadı"):
        run(n, detections=[])


# --- templates: metadata, answers, headless runs --------------------------------------------------------------------

def by_name():
    return {t["name"]: t for t in T.list_templates()}


def test_template_listing_and_filter():
    items = T.list_templates()
    names = [t["name"] for t in items]
    assert set(NEW_TEMPLATES) <= set(names)
    # the gallery templates first, by title (later releases add their own ones to this group)
    new = [t["name"] for t in items if t["new"]]
    assert set(NEW_TEMPLATES) <= set(new)
    assert names[:len(new)] == sorted(new, key=lambda n: by_name()[n]["title"].casefold())
    for t in items:
        assert t["title"] and all(c in T.CATEGORIES for c in t["categories"]), t["name"]
        if t["name"] in NEW_TEMPLATES:
            assert t["new"] and t["description"] and t["questions"]
            assert t["questions"][0]["kind"] == "source"
    assert {c for t in items for c in t["categories"]} == set(T.CATEGORIES)     # every category has something
    plc = [t["name"] for t in items if T.matches(t, "PLC")]
    assert "renk_ile_ayiklama_plc" in plc and "kenar_algilama" not in plc
    assert [t["name"] for t in items if T.matches(t, "", "ETİKET yerinde")] == ["etiket_var_yok_kontrolu"]
    assert not [t for t in items if T.matches(t, "", "olmayanbirşey")]
    assert "boyut_olcum_tolerans_rapor" in [t["name"] for t in items if T.matches(t, "", "olcum TOLERANS")]
    # the older examples got a source question when they have one picture source
    assert by_name()["tepsi_kontrol"]["questions"][0]["kind"] == "source"
    assert not by_name()["telegram_asistani"]["questions"]


def test_template_key_is_ignored_when_opening(tmp_path):
    p = ROOT / "examples" / "etiket_var_yok_kontrolu.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "template" in data
    g = Graph.load(str(p))
    assert not g.warnings and "template" not in g.to_dict()
    assert T.load_template(tmp_path / "yok.json") is None
    (tmp_path / "bozuk.json").write_text("{", encoding="utf-8")
    (tmp_path / "liste.json").write_text("[]", encoding="utf-8")
    (tmp_path / "x.json").write_text(json.dumps({"nodes": [], "template": {"title": "X", "categories": "Sayma",
                                                                         "questions": [{"id": "a", "kind": "?"},
                                                                                       "bozuk"]}}),
                                     encoding="utf-8")
    items = T.list_templates(tmp_path)
    assert [(t["title"], t["categories"], t["questions"]) for t in items] == [("X", ["Sayma"], [])]


def test_apply_answers_fills_settings():
    t = by_name()["boyut_olcum_tolerans_rapor"]
    g = T.apply_answers(t, {"nominal": 30, "tol": 0.2, "ppm": 8, "csv": "kayit/a.csv"})
    tol = g.nodes["tolerans"]
    assert (tol.values["nominal"], tol.values["low"], tol.values["high"]) == (30.0, 29.8, 30.2)
    assert g.nodes["olc"].values["px_per_mm"] == 8 and g.nodes["csv"].values["path"] == "kayit/a.csv"
    g = T.apply_answers(t)                                      # defaults
    assert g.nodes["tolerans"].values["low"] == 49.5 and g.nodes["kaynak"].type_name == "ImageFolder"
    t = by_name()["renk_ile_ayiklama_plc"]
    g = T.apply_answers(t)
    assert not g.nodes["plc"].enabled                           # no PLC unless asked
    g = T.apply_answers(t, {"plc": True, "plc_ip": "10.0.0.5", "plc_adres": "12", "tolerans": "20,5",
                            "renkler": "a=#ff0000; b=#00ff00"})
    plc = g.nodes["plc"]
    assert plc.enabled and plc.values["host"] == "10.0.0.5" and plc.values["address"] == 12
    assert g.nodes["renk"].values["tolerance"] == 20.5 and g.nodes["renk"].values["classes"].startswith("a=")
    for bad, msg in (({"tolerans": "çok"}, "Renk toleransı"), ({"tolerans": 1000}, "arasında"),
                     ({"plc_adres": 1.5}, "tam sayı"), ({"plc": "belki"}, "evet/hayır")):
        with pytest.raises(ValueError, match=msg):
            T.apply_answers(t, bad)


def test_apply_answers_swaps_the_source():
    t = by_name()["etiket_var_yok_kontrolu"]
    g = T.apply_answers(t, {"kaynak": {"type": "USB kamera", "address": "1"}})
    cam = g.nodes["kaynak"]
    assert cam.type_name == "Camera" and cam.values["source"] == "1"
    assert any(c.src == "kaynak" and c.dst == "etiket" and c.dst_port == "image" for c in g.connections)
    g = T.apply_answers(t, {"kaynak": "IP kamera (RTSP/HTTP)"})
    assert g.nodes["kaynak"].values["source"].startswith("rtsp://")
    g = T.apply_answers(t, {"kaynak": ("Video dosyası", "hat1.mp4")})
    assert g.nodes["kaynak"].type_name == "VideoFile" and g.nodes["kaynak"].values["path"] == "hat1.mp4"
    g = T.apply_answers(t, {"kaynak": {"type": "Resim klasörü", "address": "/veri/resimler"}})
    assert g.nodes["kaynak"].type_name == "ImageFolder" and g.nodes["kaynak"].values["folder"] == "/veri/resimler"
    with pytest.raises(ValueError, match="yolunu"):
        T.apply_answers(t, {"kaynak": {"type": "Video dosyası", "address": " "}})
    with pytest.raises(ValueError, match="seçenekler"):
        T.apply_answers(t, {"kaynak": "Uydu"})
    # an older example: its only source (Resim Dosyası) is swapped, its other links stay
    tep = by_name()["tepsi_kontrol"]
    g = T.apply_answers(tep, {"kaynak": "USB kamera"})
    src = next(n for n in g.nodes.values() if n.type_name == "Camera")
    assert any(c.src == src.id for c in g.connections) and not g.warnings


def test_expression_evaluator():
    assert T._eval("a - b * 2 + (-1) / 4", {"a": 10, "b": 2}) == 5.75
    for bad in ("__import__('os')", "a ** 2", "x", "1/0", "a +", "True + 1"):
        with pytest.raises(ValueError):
            T._eval(bad, {"a": 1})


EXPECTED = {   # per sample picture, in order: what the decision output shows
    "etiket_var_yok_kontrolu": ("karar", ["Etiket tamam: True", "Etiket tamam: False", "Etiket tamam: False",
                                          "Etiket tamam: False"]),
    "kapak_conta_kontrolu": ("karar", ["Kapak sağlam: True", "Kapak sağlam: False", "Kapak sağlam: False",
                                       "Kapak sağlam: False"]),
    "renk_ile_ayiklama_plc": ("sinif", ["Renk: kırmızı", "Renk: yeşil", "Renk: mavi", "Renk: sarı"]),
    "kit_eksik_parca_alarmi": ("eksik", ["Kit: tamam", "Kit: 1 altıgen eksik", "Kit: 1 daire eksik"]),
    "boyut_olcum_tolerans_rapor": ("sonuc_goster", ["SAĞLAM", "SAĞLAM", "ÜST SINIR ÜSTÜNDE", "ALT SINIR ALTINDA"]),
    "yuzey_cizik_leke_kontrolu": ("karar", ["Yüzey temiz: True", "Yüzey temiz: False", "Yüzey temiz: False"]),
    "plaka_hizalama_kontrolu": ("karar", ["Hizalı: True", "Hizalı: False", "Hizalı: False"]),
}


def _ocr_ready() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    import shutil
    return bool(shutil.which("tesseract"))


@pytest.mark.parametrize("name", NEW_TEMPLATES)
def test_every_template_runs_headless(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                 # files the flow writes (CSV) land here
    g = T.apply_answers(by_name()[name])
    src = g.nodes["kaynak"]
    src.values.update(delay=0.0, loop=False)
    seen, errors = [], []

    def on_frame(n, results):
        for nid, r in results.items():
            if r.error:
                errors.append((nid, r.error))
        if name in EXPECTED:
            out = results[EXPECTED[name][0]].outputs
            seen.append(out.get("text"))

    frames = g.run(max_frames=10, on_frame=on_frame)
    assert frames == len(list(pathlib.Path(src.v("folder")).glob("*.png")))
    ocr_missing = name == "tarih_lot_kodu_okuma" and not _ocr_ready()
    if ocr_missing:      # without Tesseract only the OCR block (and what follows it) reports, nothing crashes
        assert {nid for nid, _ in errors} <= {"ocr", "tarih", "metin", "karar", "okunan"}
    else:
        assert not errors, errors
    if name in EXPECTED:
        want = EXPECTED[name][1]
        assert [any(w in (s or "") for w in [x]) for s, x in zip(seen, want)] == [True] * len(want), seen
    if name == "boyut_olcum_tolerans_rapor":
        rows = (tmp_path / "olcumler.csv").read_text(encoding="utf-8-sig").splitlines()
        assert len(rows) == 5 and "SAĞLAM" in rows[1]


def test_date_template_reads_dates(tmp_path, monkeypatch):
    if not _ocr_ready():
        pytest.skip("Tesseract kurulu değil")
    g = T.apply_answers(by_name()["tarih_lot_kodu_okuma"], {"gun": 30})
    g.nodes["kaynak"].values.update(delay=0.0, loop=False)
    g.nodes["tarih"].now = lambda: dt.datetime(2026, 10, 14, 12, 0)
    got = []
    g.run(on_frame=lambda n, res: got.append((res["tarih"].outputs["date"], res["tarih"].outputs["ok"])))
    assert got == [("2027-10-14", True), ("2025-03-01", False), ("2027-10-14", True)]


def test_templates_pass_the_checker():
    from visionflow.check import check
    for name in NEW_TEMPLATES:
        errors = [i for i in check(T.apply_answers(by_name()[name])) if i.level == "hata"]
        assert not errors, (name, errors)


# --- gallery UI (offscreen) -----------------------------------------------------------------------------------------

@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_thumbnail_renders(qapp):
    from visionflow.ui.template_gallery import render_thumbnail
    for t in T.list_templates():
        pm = render_thumbnail(t["data"])
        assert not pm.isNull() and pm.width() == 240
    img = render_thumbnail({"nodes": []}).toImage()
    assert not img.isNull()
    assert not render_thumbnail({"nodes": [{"type": "Yok"}, "x", {"type": "Blur", "pos": "?"}],
                                 "connections": [None, {"src": 1}]}).isNull()


def test_gallery_lists_filters_and_uses(qapp):
    from PySide6.QtCore import QSettings
    from visionflow.ui.template_gallery import HIDE_KEY, TemplateGallery
    settings = QSettings(os.path.join(os.environ["VISIONFLOW_HOME"], "galeri_test.ini"), QSettings.IniFormat)
    settings.clear()
    gal = TemplateGallery(settings=settings)
    assert gal.list.count() == len(T.list_templates()) and gal.selected() is not None
    gal.category.setCurrentIndex(gal.category.findData("Ölçüm"))
    vis = gal.visible()
    assert vis and all("Ölçüm" in t["categories"] for t in vis)
    assert "boyut_olcum_tolerans_rapor" in [t["name"] for t in vis]
    gal.search.setText("hizalama")
    assert [t["name"] for t in gal.visible()] == ["plaka_hizalama_kontrolu"]
    gal.search.setText("yokböyleşey")
    assert gal.list.count() == 0 and not gal.use_btn.isEnabled() and not gal.use()
    gal.search.clear()
    gal.category.setCurrentIndex(0)
    assert gal.select("kapak_conta_kontrolu") and "Conta" in gal.asks.text()
    assert not gal.use({"conta": "çok"}) and "Hata" in gal.asks.text()
    assert gal.use({"conta": 35, "plc": True, "plc_ip": "10.1.1.1"})
    assert gal.result() == 1 and gal.chosen["name"] == "kapak_conta_kontrolu"
    assert gal.graph.nodes["conta"].values["min_pct"] == 35 and gal.graph.nodes["plc"].enabled
    gal.hide_on_start.setChecked(True)
    assert settings.value(HIDE_KEY) in (True, "true")
    gal2 = TemplateGallery(settings=settings)
    assert gal2.hide_on_start.isChecked()
    gal2.blank()
    assert gal2.graph is not None and not gal2.graph.nodes and gal2.chosen is None


def test_questions_dialog(qapp):
    from visionflow.ui.template_gallery import QuestionsDialog
    t = by_name()["boyut_olcum_tolerans_rapor"]
    dlg = QuestionsDialog(t)
    kind, combo, addr = dlg.widgets["kaynak"]
    assert combo.currentText() == T.SAMPLE and not addr.isEnabled()
    dlg.widgets["nominal"][1].setValue(20)
    dlg.widgets["tol"][1].setValue(0.1)
    dlg.widgets["csv"][1].setText("b.csv")
    combo.setCurrentText("USB kamera")
    assert addr.isEnabled() and addr.text() == "0"
    addr.setText("2")
    dlg.accept()
    assert dlg.result() == 1
    g = dlg.graph
    assert g.nodes["tolerans"].values["low"] == pytest.approx(19.9) and g.nodes["csv"].values["path"] == "b.csv"
    assert g.nodes["kaynak"].type_name == "Camera" and g.nodes["kaynak"].values["source"] == "2"
    dlg = QuestionsDialog(t)
    dlg.widgets["kaynak"][1].setCurrentText("Video dosyası")
    dlg.widgets["kaynak"][2].setText("")
    dlg.accept()                                  # a missing video path: stays open with the reason
    assert dlg.result() != 1 and "yolunu" in dlg.error.text()
    choice = {"title": "S", "questions": T._clean_questions([{"id": "c", "kind": "choice", "choices": ["a", "b"],
                                                               "default": "b", "targets": []}]),
              "data": {"nodes": []}}
    d = QuestionsDialog(choice)
    assert d.answers() == {"c": "b"}


@pytest.fixture
def win(qapp):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    qapp.processEvents()


def test_new_flow_from_gallery_in_editor(win):
    from visionflow.ui import template_gallery as G

    def pick(dlg):
        dlg.select("renk_ile_ayiklama_plc")
        return dlg.use({"plc_ip": "10.2.2.2"})

    assert G.new_from_gallery(win, exec_dialog=pick)
    assert "renk" in win.root.nodes and win.root.nodes["plc"].values["host"] == "10.2.2.2"
    assert win.path is None and win.dirty and win.windowTitle().startswith("Yeni akış*")
    assert not G.new_from_gallery(win, exec_dialog=lambda dlg: False)       # cancelled: the flow stays
    assert "renk" in win.root.nodes
    assert G.new_from_gallery(win, exec_dialog=lambda dlg: dlg.blank() or True)
    assert not win.root.nodes
    win._confirm_discard = lambda: False                                      # "Vazgeç" in the save question
    assert not G.new_from_gallery(win, exec_dialog=pick) and not win.root.nodes


def test_menu_new_opens_gallery_and_first_start(win, monkeypatch):
    from visionflow.ui import template_gallery as G
    calls = []
    monkeypatch.setattr(G, "new_from_gallery", lambda w: calls.append(w) or True)
    act = next(a for a in win.findChildren(type(win.undo_act)) if a.text() == "Yeni…")
    act.trigger()
    assert calls == [win]
    blank = next(a for a in win.findChildren(type(win.undo_act)) if a.text() == "Boş akış")
    win.scene.add_node("Blur", __import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(0, 0))
    blank.trigger()
    assert not win.root.nodes
    win.settings.setValue("recent", [])
    win.settings.setValue(G.HIDE_KEY, False)
    assert G.should_show_on_start(win)
    win.settings.setValue(G.HIDE_KEY, True)
    assert not G.should_show_on_start(win)
    win.settings.setValue(G.HIDE_KEY, False)
    win.settings.setValue("recent", ["/x/a.json"])
    assert not G.should_show_on_start(win) and not G.maybe_show_on_start(win)
    win.settings.setValue("recent", [])
