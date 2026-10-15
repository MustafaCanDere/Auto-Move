"""2.0 (c): factory-floor blocks — reject delay, SPC, OEE, lot tracking, alarm history, multi-camera verdict and
the signal simulator. Time comes from injected clocks, numbers are checked against hand-computed values."""
import csv
import datetime as dt
import json
import pathlib

import pytest

import visionflow.nodes  # noqa: F401
from visionflow.core import Graph, registry
from visionflow.nodes import production

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "ret_kapisi_simulasyon.json"
T0 = dt.datetime(2026, 10, 13, 10, 0, 0)


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def make(type_name, clock=None, **values):
    n = registry.create(type_name)
    n.values.update(values)
    if clock is not None:
        n.now = clock
    n.setup()
    return n


def step(n, **inputs):
    full = {name: None for name, _ in n.inputs}
    full.update(inputs)
    return n.process(full)


# ---------------------------------------------------------------- reject delay
def part(n, clock, bad, dt_=0.01, **extra):
    """One part: trigger high for one frame, then low. Returns both outputs."""
    a = step(n, trigger=True, bad=bad, **extra)
    clock.t += dt_
    b = step(n, trigger=False, bad=False, **extra)
    clock.t += dt_
    return a, b


def test_reject_by_part_count_close_parts():
    c = Clock()
    n = make("RejectDelay", c, mode="parça sayısı", distance=3, pulse_ms=0)
    pattern = [True, False, True, True, False, False, False, False]
    fired = []
    for i, bad in enumerate(pattern):
        a, b = part(n, c, bad)
        if a["reject"]:
            fired.append(i)
        assert not b["reject"]                         # pulse length 0: only the frame the part arrives
    # part 0 is rejected when part 3 is seen, parts 2 and 3 (back to back) at parts 5 and 6
    assert fired == [3, 5, 6]
    assert a["rejected"] == 3 and a["pending"] == 0 and a["parts"] == 8


def test_reject_pending_count_and_distance_zero():
    c = Clock()
    n = make("RejectDelay", c, distance=2, pulse_ms=0)
    a, _ = part(n, c, True)
    assert a["pending"] == 1 and not a["reject"]
    a, _ = part(n, c, True)
    assert a["pending"] == 2
    a, _ = part(n, c, False)
    assert a["reject"] and a["pending"] == 1
    n0 = make("RejectDelay", c, distance=0, pulse_ms=0)
    a, _ = part(n0, c, True)
    assert a["reject"] and a["pending"] == 0           # blower right at the camera


def test_reject_pulse_length():
    c = Clock(100.0)
    n = make("RejectDelay", c, distance=1, pulse_ms=100)
    part(n, c, True, dt_=0.0)
    c.t = 200.0
    assert step(n, trigger=True, bad=False)["reject"]
    c.t = 200.05
    assert step(n, trigger=False)["reject"]
    c.t = 200.1
    assert not step(n, trigger=False)["reject"]


def test_reject_trigger_held_high_counts_once():
    c = Clock()
    n = make("RejectDelay", c, distance=1, pulse_ms=0)
    for _ in range(5):
        r = step(n, trigger=True, bad=True)
    assert r["parts"] == 1 and r["pending"] == 1


def test_reject_by_encoder_close_parts():
    c = Clock()
    n = make("RejectDelay", c, mode="enkoder", distance=250, pulse_ms=0)
    assert step(n, trigger=True, bad=True, encoder=0)["pending"] == 1
    step(n, trigger=False, encoder=50)
    assert step(n, trigger=True, bad=True, encoder=100)["pending"] == 2     # closer than the distance
    step(n, trigger=False, encoder=120)
    step(n, trigger=True, bad=False, encoder=150)
    assert not step(n, trigger=False, encoder=249)["reject"]
    r = step(n, trigger=False, encoder=250)
    assert r["reject"] and r["pending"] == 1
    assert not step(n, trigger=False, encoder=349)["reject"]
    r = step(n, trigger=False, encoder=360)
    assert r["reject"] and r["pending"] == 0 and r["rejected"] == 2


def test_reject_encoder_two_in_one_frame_and_wrap():
    c = Clock()
    n = make("RejectDelay", c, mode="enkoder", distance=100, pulse_ms=0)
    step(n, trigger=True, bad=True, encoder=0)
    step(n, trigger=False, encoder=10)
    step(n, trigger=True, bad=True, encoder=20)
    r = step(n, trigger=False, encoder=500)              # slow frame: both reach the blower at once
    assert r["reject"] and r["rejected"] == 2 and r["pending"] == 0
    w = make("RejectDelay", c, mode="enkoder", distance=250, pulse_ms=0, encoder_max=1000)
    step(w, trigger=True, bad=True, encoder=900)
    assert not step(w, trigger=False, encoder=950)["reject"]
    assert not step(w, trigger=False, encoder=100)["reject"]   # rolled over: 1100 < 1150
    assert step(w, trigger=False, encoder=160)["reject"]       # 1160


def test_reject_by_time():
    c = Clock(5.0)
    n = make("RejectDelay", c, mode="süre (ms)", distance=500, pulse_ms=0)
    step(n, trigger=True, bad=True)
    c.t = 5.2
    step(n, trigger=False)
    step(n, trigger=True, bad=True)                       # second bad part 200 ms later
    c.t = 5.49
    assert not step(n, trigger=False)["reject"]
    c.t = 5.5
    assert step(n, trigger=False)["pending"] == 1
    c.t = 5.6
    assert not step(n, trigger=False)["reject"]
    c.t = 5.7
    r = step(n, trigger=False)
    assert r["reject"] and r["pending"] == 0


def test_reject_errors_and_reset():
    c = Clock()
    n = make("RejectDelay", c, mode="enkoder")
    with pytest.raises(ValueError, match="encoder"):
        step(n, trigger=True, bad=True)
    with pytest.raises(ValueError, match="sayı değil"):
        step(n, trigger=True, bad=True, encoder="abc")
    with pytest.raises(ValueError, match="trigger"):
        step(make("RejectDelay", c), bad=True)
    m = make("RejectDelay", c, distance=5)
    part(m, c, True)
    assert step(m, trigger=False, reset=True)["pending"] == 0


# ---------------------------------------------------------------- SPC
def feed(n, values):
    out = None
    for v in values:
        out = step(n, value=v)
    return out


def test_spc_xbar_limits_and_capability_hand_computed():
    n = make("SPCChart", subgroup=5, baseline=4, lsl=8.0, usl=18.0)
    out = feed(n, [10, 11, 12, 13, 14, 11, 12, 13, 14, 15, 9, 10, 11, 12])
    assert out["learning"] and out["ucl"] is None and out["points"] == 2
    out = feed(n, [13] + [12] * 5)
    assert not out["learning"] and out["points"] == 4
    # means 12, 13, 11, 12 → CL 12; ranges 4, 4, 4, 0 → R̄ 3, σ = 3 / 2.326; limits = CL ± 3σ/√5
    assert out["center"] == pytest.approx(12.0)
    assert out["sigma"] == pytest.approx(1.2897678, abs=1e-6)
    assert out["ucl"] == pytest.approx(13.730405, abs=1e-6)
    assert out["lcl"] == pytest.approx(10.269595, abs=1e-6)
    assert out["cp"] == pytest.approx(1.292222, abs=1e-6)        # 10 / 6σ
    assert out["cpk"] == pytest.approx(1.033778, abs=1e-6)       # (12 - 8) / 3σ
    assert not out["alarm"] and out["alarm_text"] == ""
    out = feed(n, [14] * 5)                                       # mean 14 > UCL 13.73
    assert out["alarm"] and out["alarm_text"].startswith("Kural 1") and out["point"] == 14
    out = feed(n, [12] * 4)                                       # half a subgroup: nothing new
    assert out["alarm"] and out["points"] == 5
    assert not feed(n, [12])["alarm"]


def test_spc_individuals_rule2_nine_points_one_side():
    n = make("SPCChart", subgroup=1, baseline=5)
    out = feed(n, [10, 12, 11, 13, 9])
    # mean 11; moving ranges 2, 1, 2, 4 → MR̄ 2.25, σ = 2.25 / 1.128
    assert out["center"] == pytest.approx(11.0)
    assert out["ucl"] == pytest.approx(16.984043, abs=1e-6)
    assert out["lcl"] == pytest.approx(5.015957, abs=1e-6)
    alarms = [step(n, value=v)["alarm"] for v in [11.5, 12] * 4]
    assert alarms == [False] * 8
    out = step(n, value=11.5)                                     # 9th point above the center
    assert out["alarm"] and out["alarm_text"] == "Kural 2: 9 nokta ortalamanın üstünde"


def test_spc_rule3_trend_and_rule1_low():
    n = make("SPCChart", subgroup=1, baseline=5)
    feed(n, [10, 12, 11, 13, 9])
    outs = [step(n, value=v) for v in [10, 10.5, 11, 11.5, 12]]
    assert [o["alarm"] for o in outs] == [False, False, False, False, True]
    assert outs[-1]["alarm_text"] == "Kural 3: 6 nokta sürekli artıyor"
    out = step(n, value=4)
    assert out["alarm_text"].startswith("Kural 1: nokta 3σ dışında (4 < LCL")


def test_spc_extra_rules_and_rolling_mode():
    n = make("SPCChart", subgroup=1, baseline=5, extra_rules=True)
    feed(n, [10, 12, 11, 13, 9])                                  # σ ≈ 1.9947, CL 11 → 2σ = 14.99
    step(n, value=15.5)
    out = step(n, value=15.2)
    assert "Kural 4" in out["alarm_text"]
    r = make("SPCChart", subgroup=1, baseline=0)
    outs = [step(r, value=v) for v in [10, 12, 11, 13, 9]]
    assert all(o["ucl"] is None and not o["alarm"] for o in outs)   # 5 earlier points needed
    out = step(r, value=30)
    assert out["ucl"] == pytest.approx(16.984043, abs=1e-6) and out["alarm"]
    assert not out["learning"]


def test_spc_trigger_bad_input_and_reset():
    n = make("SPCChart", subgroup=1, baseline=2)
    for v in [1, None, "x", float("nan"), True]:
        out = step(n, value=v)
    assert out["points"] == 1                                    # only the 1 counted
    t = make("SPCChart", subgroup=1, baseline=3)
    step(t, value=5, trigger=True)
    step(t, value=6, trigger=True)                               # held high: not a new sample
    step(t, value=7, trigger=False)
    assert step(t, value=8, trigger=True)["points"] == 2
    assert step(t, value=8, reset=True)["points"] == 0
    with pytest.raises(ValueError, match="LSL"):
        step(make("SPCChart", lsl=5, usl=1), value=1)
    bad = registry.create("SPCChart")
    bad.values.update(subgroup=1, baseline=1)
    assert not bad.safe_setup() and "en az 2" in bad.setup_error


def test_spc_stats_function():
    assert production.spc_stats([(1.0, 0.0)], 1) is None
    assert production.spc_stats([], 4) is None
    cl, s = production.spc_stats([(10.0, 2.059), (12.0, 2.059)], 4)
    assert cl == 11.0 and s == pytest.approx(1.0)


# ---------------------------------------------------------------- OEE
class DateClock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def at(self, seconds):
        self.t = T0 + dt.timedelta(seconds=seconds)


def test_oee_hand_computed():
    c = DateClock(T0)
    n = make("OEECalc", c, ideal_cycle=2.0)
    step(n, running=True, good=0, bad=0)
    c.at(3000)
    step(n, running=False, good=1200, bad=50)          # ran 50 min, then stopped
    c.at(3600)
    out = step(n, running=False, good=1200, bad=50)
    # A = 3000/3600, P = 2 s × 1250 / 3000 s, Q = 1200/1250 → OEE = 0.8333 × 0.8333 × 0.96
    assert out["availability"] == 83.33
    assert out["performance"] == 83.33
    assert out["quality"] == 96.0
    assert out["oee"] == 66.67
    assert out["downtime_min"] == 10.0 and out["run_min"] == 50.0
    assert out["good_count"] == 1200 and out["bad_count"] == 50


def test_oee_pulse_mode_counter_reset_and_cap():
    c = DateClock(T0)
    n = make("OEECalc", c, ideal_cycle=10.0, count_mode="darbe (yükselen kenar)")
    for i, g in enumerate([True, True, False, True, False, True]):
        c.at(i)
        out = step(n, running=True, good=g, bad=False)
    assert out["good_count"] == 3 and out["performance"] == 100.0    # 30 s ideal in 5 s: capped
    m = make("OEECalc", c, ideal_cycle=1.0)
    for v in [5, 8, 2, 4]:                                           # counter reset 8 → 2: still +2
        out = step(m, running=True, good=v, bad=None)
    assert out["good_count"] == 7 and out["quality"] == 100.0


def test_oee_shift_reset_and_bad_times():
    c = DateClock(T0)                                                 # 10:00
    n = make("OEECalc", c, ideal_cycle=1.0, shift_times="10:30")
    step(n, running=True, good=0, bad=0)
    c.at(1200)
    out = step(n, running=True, good=100, bad=0)
    assert out["run_min"] == 20.0 and out["good_count"] == 100
    c.at(2100)                                                        # 10:35: new shift since 10:30
    out = step(n, running=True, good=130, bad=0)
    assert out["run_min"] == 5.0 and out["availability"] == 100.0 and out["good_count"] == 30
    bad = registry.create("OEECalc")
    bad.values["shift_times"] = "25:00"
    assert not bad.safe_setup() and "Saat" in bad.setup_error
    first = make("OEECalc", c)
    out = step(first, running=True)
    assert out["oee"] == 0.0 and out["quality"] == 100.0


def test_oee_reset_input():
    c = DateClock(T0)
    n = make("OEECalc", c, ideal_cycle=1.0)
    step(n, running=True, good=0)
    c.at(60)
    step(n, running=False, good=60)
    c.at(120)
    out = step(n, running=True, good=60, reset=True)
    assert out["good_count"] == 0 and out["run_min"] == 0.0
    c.at(180)
    out = step(n, running=True, good=90)
    assert out["good_count"] == 30 and out["availability"] == 100.0 and out["performance"] == 50.0


# ---------------------------------------------------------------- lot tracking
@pytest.fixture
def vf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))
    return tmp_path


def pulse(n, **kw):
    a = step(n, trigger=True, **kw)
    step(n, trigger=False, **kw)
    return a


def test_lot_auto_ids_size_and_daily_sequence(vf_home):
    c = DateClock(T0)
    n = make("LotTracker", c, size=3, prefix="LOT", name="t1")
    outs = [pulse(n) for _ in range(4)]
    assert [o["lot_id"] for o in outs] == ["LOT-20261013-001"] * 3 + ["LOT-20261013-002"]
    assert [o["count"] for o in outs] == [1, 2, 3, 1]
    assert [o["done"] for o in outs] == [False, False, True, False]
    assert outs[2]["remaining"] == 0 and outs[3]["remaining"] == 2 and outs[3]["last_lot"] == "LOT-20261013-001"
    assert not step(n, trigger=False)["done"]                         # one-frame pulse
    c.t = T0 + dt.timedelta(days=1)
    step(n, new_lot=True)
    out = pulse(n)
    assert out["lot_id"] == "LOT-20261014-001" and out["count"] == 1 and out["lots"] == 2


def test_lot_manual_number_and_new_lot(vf_home):
    c = DateClock(T0)
    n = make("LotTracker", c, size=2, name="t2")
    assert pulse(n, lot="A7")["lot_id"] == "A7"
    assert pulse(n, lot="A7")["done"]
    assert pulse(n, lot="A7")["lot_id"] == "A7-2"                      # same barcode, lot already full
    out = step(n, lot="B1")                                          # new barcode closes the open lot early
    assert out["done"] and out["last_lot"] == "A7-2" and out["lot_id"] == "B1" and out["count"] == 0
    out = step(n, lot="B1", new_lot=True)
    assert not out["done"] and out["lot_id"] == "B1-2"                 # empty lot: nothing to close


def test_lot_persists_and_can_be_off(vf_home):
    c = DateClock(T0)
    n = make("LotTracker", c, size=10, name="kalici")
    for _ in range(4):
        pulse(n)
    again = make("LotTracker", c, size=10, name="kalici")
    out = pulse(again)
    assert out["lot_id"] == "LOT-20261013-001" and out["count"] == 5
    assert json.loads((vf_home / "lotlar" / "kalici.json").read_text(encoding="utf-8"))["count"] == 5
    off = make("LotTracker", c, size=10, name="gecici", persist=False)
    pulse(off)
    assert not (vf_home / "lotlar" / "gecici.json").exists()
    (vf_home / "lotlar" / "bozuk.json").write_text("[1, 2", encoding="utf-8")
    assert pulse(make("LotTracker", c, name="bozuk"))["count"] == 1   # broken file: start fresh


# ---------------------------------------------------------------- alarm history
def test_alarm_history_flapping_csv_and_today(vf_home, tmp_path):
    path = tmp_path / "log" / "alarm.csv"
    c = DateClock(T0)
    n = make("AlarmHistory", c, path=str(path), min_seconds=2.0, keep=5)

    def at(s, cond, text=""):
        c.at(s)
        return step(n, condition=cond, text=text)

    at(0, True, "titreşim")
    assert not at(1, False)["active"]                                   # 1 s: too short, ignored
    at(10, True, "Malzeme yok")
    assert not at(11, True)["active"]
    out = at(12, True)
    assert out["active"] and out["count_today"] == 1 and out["last_text"] == "Malzeme yok" and out["duration"] == 2
    at(20, False)
    at(21, True)                                                        # 1 s gap: same alarm
    at(25, False)
    assert at(26, False)["active"]
    out = at(27, False)
    assert not out["active"] and out["duration"] == 15.0 and out["count_today"] == 1
    rows = list(csv.reader(path.open(encoding="utf-8-sig"), delimiter=";"))
    assert rows == [["Başlangıç", "Bitiş", "Süre (sn)", "Sebep"],
                    ["2026-10-13 10:00:10", "2026-10-13 10:00:25", "15,0", "Malzeme yok"]]
    assert "Malzeme yok" in out["history"]
    c.t = T0 + dt.timedelta(days=1)
    assert step(n, condition=False)["count_today"] == 0


def test_alarm_history_keeps_last_n_and_teardown(tmp_path):
    c = DateClock(T0)
    n = make("AlarmHistory", c, path="", keep=2)
    for i in range(3):
        c.at(i * 10)
        step(n, condition=True, text=f"A{i}")
        c.at(i * 10 + 5)
        out = step(n, condition=False)
    assert out["count_today"] == 3 and len(out["history"].splitlines()) == 2
    assert out["history"].splitlines()[0].endswith("A2")
    p = tmp_path / "x.csv"
    m = make("AlarmHistory", c, path=str(p))
    step(m, condition=True, text=None)
    c.t += dt.timedelta(seconds=3)
    assert step(m, condition=True, text="sonradan")["last_text"] == "sonradan"
    m.teardown()
    assert "(akış durdu)" in p.read_text(encoding="utf-8-sig")


def test_alarm_history_unwritable_csv_keeps_rows(tmp_path):
    c = DateClock(T0)
    blocker = tmp_path / "dosya"
    blocker.write_text("x")
    n = make("AlarmHistory", c, path=str(blocker / "a.csv"))
    step(n, condition=True, text="x")
    c.at(1)
    out = step(n, condition=False)
    assert "CSV yazılamadı" in out["history"] and len(n.state["pending"]) == 1


# ---------------------------------------------------------------- multi-station verdict
def test_multi_station_matches_by_order():
    c = Clock()
    n = make("MultiStationVerdict", c, stations=3, names="Üst, Yan, Alt")
    idle = {"t1": False, "t2": False, "t3": False, "ok1": True, "ok2": True, "ok3": True}
    step(n, **idle)
    # station 1 reports two parts before the others report the first one
    step(n, **{**idle, "t1": True, "ok1": True})
    step(n, **idle)
    step(n, **{**idle, "t1": True, "ok1": False})
    out = step(n, **{**idle, "t2": True, "t3": True})
    assert out["done"] and out["ok"] and out["failed"] == ""
    out = step(n, **idle)
    assert not out["done"]
    out = step(n, **{**idle, "t2": True, "t3": True, "ok3": False})
    assert out["done"] and not out["ok"] and out["bad"] and out["failed"] == "Üst, Alt" and out["parts"] == 2


def test_multi_station_timeout():
    c = Clock(10.0)
    n = make("MultiStationVerdict", c, stations=2, timeout=0.5)
    step(n, t1=True, ok1=True, t2=False, ok2=True)
    c.t = 10.49
    assert not step(n, t1=False, ok1=True, t2=False, ok2=True)["done"]
    c.t = 10.5
    out = step(n, t1=False, ok1=True, t2=False, ok2=True)
    assert out["done"] and out["timeout"] and not out["ok"] and out["failed"] == "İstasyon 2 bildirmedi"
    assert out["timeouts"] == 1
    c.t = 10.6
    out = step(n, t1=False, ok1=True, t2=False, ok2=True)
    assert not out["timeout"] and not out["done"]
    with pytest.raises(ValueError, match="t2"):
        step(n, t1=True, ok1=True)
    with pytest.raises(ValueError, match="ok1"):
        step(make("MultiStationVerdict", c, stations=1), t1=True)


# ---------------------------------------------------------------- simulator
def test_simulator_rate_encoder_and_seed():
    c = Clock(0.0)
    n = make("SignalSimulator", c, rate=60, pulses_per_part=100, bad_ratio=50, seed=7)
    outs = []
    for i in range(41):                                  # 4 s at 10 frames/s
        c.t = i * 0.1
        outs.append(step(n))
    trig = [i for i, o in enumerate(outs) if o["trigger"]]
    assert trig == [10, 20, 30, 40]
    assert outs[15]["encoder"] == 150 and outs[40]["parts"] == 4
    assert outs[40]["good_count"] + outs[40]["bad_count"] == 4
    c2 = Clock(0.0)
    m = make("SignalSimulator", c2, rate=60, pulses_per_part=100, bad_ratio=50, seed=7)
    bads = []
    for i in range(41):
        c2.t = i * 0.1
        o = step(m)
        if o["trigger"]:
            bads.append(o["bad"])
    assert bads == [outs[i]["bad"] for i in trig]        # same seed → same parts


def test_simulator_bad_ratio_extremes_and_backlog():
    c = Clock(0.0)
    allbad = make("SignalSimulator", c, rate=600, bad_ratio=100)
    c.t = 0.1
    assert step(allbad)["bad"]
    fast = make("SignalSimulator", c, rate=6000, bad_ratio=0)         # 100 parts/s at 4 frames: one per 2 frames
    c.t = 0.0
    fast.setup()
    trigs = []
    for i in range(1, 5):
        c.t = i * 0.01
        o = step(fast)
        trigs.append(o["trigger"])
    assert trigs == [True, False, True, False] and not o["bad"]
    c.t = 10.0
    o = step(fast)
    assert o["trigger"] and o["parts"] == 3                          # backlog dropped, not replayed for ages


def test_simulator_stops_and_measurement():
    c = Clock(0.0)
    n = make("SignalSimulator", c, rate=60, stop_every=10, stop_seconds=5, sd=0.0, mean=5.0, drift=0.5)
    c.t = 9.99
    assert step(n)["running"]
    c.t = 12.0
    o = step(n)
    assert not o["running"] and o["encoder"] == 1000                # encoder stands still while stopped
    c.t = 16.0
    o = step(n)
    assert o["running"] and o["encoder"] == 1100
    m = make("SignalSimulator", c, rate=60, sd=0.0, mean=5.0, drift=0.5)
    c.t = 0.0
    m.setup()
    vals = []
    for i in range(1, 7):
        c.t = i * 0.5
        o = step(m)
        if o["trigger"]:
            vals.append(o["measurement"])
    assert vals == [5.0, 5.5, 6.0]


# ---------------------------------------------------------------- simulator → blocks, example flow
def test_simulator_feeds_reject_delay_in_a_graph():
    g = Graph()
    sim = g.add_new("SignalSimulator", rate=600, bad_ratio=30, seed=3)
    rej = g.add_new("RejectDelay", mode="enkoder", distance=300, pulse_ms=0)
    cnt = g.add_new("RejectDelay", mode="parça sayısı", distance=3, pulse_ms=0)
    for a, b in (("trigger", "trigger"), ("bad", "bad"), ("encoder", "encoder")):
        g.connect(sim.id, a, rej.id, b)
        g.connect(sim.id, a, cnt.id, b)
    c = Clock(0.0)
    sim.now = c
    g.setup()
    enc_hits, cnt_hits = [], []
    for i in range(1, 400):
        c.t = i * 0.025                                  # 40 frames/s, 10 parts/s
        r = g.run_once()
        assert not any(x.error for x in r.values())
        enc_hits.append(r[rej.id].outputs["reject"])
        cnt_hits.append(r[cnt.id].outputs["reject"])
    # parts are exactly 100 pulses apart, so 300 pulses = 3 parts: both modes reject at the same frames
    assert enc_hits == cnt_hits and sum(enc_hits) == r[rej.id].outputs["rejected"] > 0
    assert r[sim.id].outputs["bad_count"] - r[rej.id].outputs["rejected"] == r[rej.id].outputs["pending"]


def test_example_flow_runs():
    g = Graph.load(str(EXAMPLE))
    assert not g.warnings
    types = {n.type_name for n in g.nodes.values()}
    assert {"SignalSimulator", "RejectDelay", "OEECalc", "SPCChart", "ShowValue"} <= types
    sim = next(n for n in g.nodes.values() if n.type_name == "SignalSimulator")
    rej = next(n for n in g.nodes.values() if n.type_name == "RejectDelay")
    c = Clock(0.0)
    sim.now = c
    rej.now = c
    g.setup()
    rejects = 0
    for i in range(1, 300):
        c.t = i / 30
        r = g.run_once()
        assert not any(x.error for x in r.values()), {k: x.error for k, x in r.items() if x.error}
        rejects += r[rej.id].outputs["rejected"] > 0
    out = r[rej.id].outputs
    assert out["parts"] == r[sim.id].outputs["parts"] == 9
    assert out["rejected"] + out["pending"] == r[sim.id].outputs["bad_count"]
    shows = [r[n.id].outputs["text"] for n in g.nodes.values() if n.type_name == "ShowValue"]
    assert shows and all(isinstance(s, str) for s in shows)
    g.teardown()


def test_blocks_are_registered_with_turkish_titles():
    for name in ("RejectDelay", "SPCChart", "OEECalc", "LotTracker", "AlarmHistory", "MultiStationVerdict",
                 "SignalSimulator"):
        cls = registry.NODE_REGISTRY[name]
        assert cls.title and cls.description and cls.category
    assert registry.NODE_REGISTRY["SignalSimulator"].is_source


def test_oee_bool_in_counter_mode_counts_pulses():
    c = DateClock(T0)
    n = make("OEECalc", c)
    for g in [False, True, False, True, True]:
        out = step(n, running=True, good=g, bad=False)
    assert out["good_count"] == 2 and out["bad_count"] == 0
