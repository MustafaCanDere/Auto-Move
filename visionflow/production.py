"""Production data of a running flow: live good/bad counting for the operator panel (Tally), a small SQLite file
the counts are sampled into every minute (ProductionDB), and the self-contained HTML / PDF production report made
from it."""
from __future__ import annotations

import collections
import datetime as dt
import html
import math
import os
import pathlib
import sqlite3
import threading
import time

from .core.graph import inner_error, node_at, walk_nodes

DB_NAME = "uretim_verisi.sqlite"
SAMPLE_S = 60.0                 # the counts are written to the production database every minute
ALARM_REPEAT_S = 60.0           # the same alarm text is recorded at most once a minute
MAX_HOURS = 24 * 31             # hourly table / chart rows in one report


def _num(v):
    if isinstance(v, bool):
        return float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def default_image(graph) -> str:
    """The block the operator panel shows when none is chosen: the first Preview block (inside sub-flows too, as
    "subflow id/inner id"), else the last block (in run order) with an image output."""
    from .core import PortType
    for key, _label, node in walk_nodes(graph):
        if node.type_name == "Preview" and node.enabled:
            return key
    try:
        order = graph.order()
    except Exception:  # noqa: BLE001
        order = list(graph.nodes)
    for nid in reversed(order):
        if any(t == PortType.IMAGE for _, t in graph.nodes[nid].outputs):
            return nid
    return ""


def auto_config(graph) -> dict:
    """A starting operator configuration for a flow that has none: its preview and a pass/fail-looking output
    (blocks inside sub-flows count too)."""
    from .core import PortType
    lamp = None
    for key, _label, node in walk_nodes(graph):
        for port, t in node.outputs:
            if t == PortType.BOOL and port.lower() in ("ok", "pass", "passed", "good", "result", "sonuc", "sağlam"):
                lamp = {"node": key, "port": port}
                break
        if lamp:
            break
    return {"title": "", "image": "", "lamp": lamp, "good": None, "bad": None, "total": None, "history": 20}


class Tally:
    """Counts good and bad parts while the flow runs. ``feed`` is called by the flow thread with every frame's
    results; the panel reads ``snapshot()`` from the GUI thread.

    With counter outputs chosen (e.g. Tolerans Kontrolü ok_count / nok_count) their increases are added up, so a
    counter that restarts at 0 (flow restarted) keeps counting. Without them every frame that produces a lamp value
    is one part (use a triggered camera so a part is one frame)."""

    def __init__(self, cfg: dict | None = None, clock=time.time):
        self.cfg = dict(cfg or {})
        self.clock = clock
        self._lock = threading.Lock()
        self.history = collections.deque(maxlen=int(self.cfg.get("history") or 20))
        self.alarms: list[tuple[float, str]] = []
        self._alarm_seen: dict[str, float] = {}
        self._pending = {k: 0 for k in ("good", "bad", "total")}   # not yet written to the minute samples
        self.reset()
        self._last: dict[str, float | None] = {"good": None, "bad": None, "total": None}

    def reset(self):
        """Operator's "Sayaçları sıfırla": counts start again (the minute samples already written stay)."""
        with self._lock:
            self.counts = {"good": 0, "bad": 0, "total": 0}
            self.history.clear()
            self.lamp: bool | None = None
            self.lamp_error = ""

    @property
    def counter_mode(self) -> bool:
        return bool(self.cfg.get("good") or self.cfg.get("bad"))

    @staticmethod
    def _read(results, ref):
        """(value, error) of a chosen output in this frame's results; (None, "") when it has nothing new."""
        if not ref:
            return None, ""
        r = results.get(ref["node"])
        if r is None or r.waiting:
            return None, ""
        if r.error:
            return None, r.error
        return r.outputs.get(ref["port"]), ""

    def _delta(self, key, value) -> int:
        v = _num(value)
        if v is None:
            return 0
        last = self._last[key]
        self._last[key] = v
        if last is None:
            # the first reading is the baseline (e.g. a daily counter kept from an earlier run at 5000); only a
            # counter that has just started (showing its first part) counts it
            return int(v) if 0 <= v <= 1 else 0
        return int(v - last) if v >= last else int(v)   # restarted at 0: what it shows now is new

    def feed(self, results: dict, graph=None) -> None:
        now = self.clock()
        with self._lock:
            lamp_v, lamp_err = self._read(results, self.cfg.get("lamp"))
            if lamp_err:
                self.lamp, self.lamp_error = None, lamp_err
            elif lamp_v is not None:
                self.lamp, self.lamp_error = bool(lamp_v), ""
            if self.counter_mode:
                g = self._delta("good", self._read(results, self.cfg.get("good"))[0])
                b = self._delta("bad", self._read(results, self.cfg.get("bad"))[0])
            else:
                g = b = 0
                if lamp_v is not None and not lamp_err:
                    g, b = (1, 0) if lamp_v else (0, 1)
            if self.cfg.get("total"):
                t = self._delta("total", self._read(results, self.cfg["total"])[0])
            else:
                t = g + b
            for k, d in (("good", g), ("bad", b), ("total", t)):
                self.counts[k] += d
                self._pending[k] += d
            if g or b:
                ok = b == 0
                self.history.appendleft((now, ok, f"{'SAĞLAM' if ok else 'HATALI'}" +
                                         (f" ×{g + b}" if g + b > 1 else "")))
            if graph is not None:
                for nid, r in results.items():
                    if r.error and r.error != "Önceki blok hata verdi":
                        node = node_at(graph, nid)
                        if node is not None and not inner_error(node, r.error):
                            self._alarm(f"{node.title}: {r.error}", now)

    def _alarm(self, text: str, now: float):
        text = text[:300]
        if now - self._alarm_seen.get(text, -1e18) < ALARM_REPEAT_S:
            return
        self._alarm_seen[text] = now
        self.alarms.append((now, text))
        del self.alarms[:-200]

    def alarm(self, text: str):
        with self._lock:
            self._alarm(str(text), self.clock())

    def snapshot(self) -> dict:
        with self._lock:
            c = dict(self.counts)
            total = c["total"]
            return {"lamp": self.lamp, "lamp_error": self.lamp_error, **c,
                    "yield": round(100.0 * c["good"] / total, 1) if total else None,
                    "history": list(self.history)}

    def take(self) -> tuple[int, int, int, list]:
        """(good, bad, total, alarms) added since the last call, for the minute sample."""
        with self._lock:
            d, self._pending = self._pending, {k: 0 for k in self._pending}
            alarms, self.alarms = self.alarms, []
        return max(0, d["good"]), max(0, d["bad"]), max(0, d["total"]), alarms


# --- the SQLite collector ------------------------------------------------------------------------------------------

def db_path(flow_path: str | None = None) -> pathlib.Path:
    """Next to the flow file when it has one, else in the user's VisionFlow folder."""
    if flow_path:
        return pathlib.Path(flow_path).resolve().parent / DB_NAME
    from .ai.config import home
    return home() / DB_NAME


def store_sample(tally: "Tally", db: "ProductionDB", flow: str) -> None:
    """Write what `tally` counted since the last call (and its new alarms) to `db` as one sample of flow `flow`.
    Used by the operator panel every minute and by headless / service runs."""
    good, bad, total, alarms = tally.take()
    db.add_sample(flow, good, bad, total, alarms=alarms)


class ProductionDB:
    def __init__(self, path):
        self.path = pathlib.Path(path)

    def _conn(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.path), timeout=5)
        con.execute("CREATE TABLE IF NOT EXISTS samples (ts REAL, flow TEXT, good INTEGER, bad INTEGER, total INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS alarms (ts REAL, flow TEXT, text TEXT)")
        con.execute("CREATE INDEX IF NOT EXISTS samples_ts ON samples (ts)")
        return con

    def add_sample(self, flow: str, good: int, bad: int, total: int, ts: float | None = None, alarms=()):
        if not (good or bad or total or alarms):
            return
        con = self._conn()
        try:
            with con:
                if good or bad or total:
                    con.execute("INSERT INTO samples VALUES (?,?,?,?,?)",
                                (ts if ts is not None else time.time(), flow, int(good), int(bad), int(total)))
                con.executemany("INSERT INTO alarms VALUES (?,?,?)", [(t, flow, text) for t, text in alarms])
        finally:
            con.close()

    def flows(self) -> list[str]:
        if not self.path.exists():
            return []
        con = self._conn()
        try:
            return [r[0] for r in con.execute("SELECT DISTINCT flow FROM samples ORDER BY flow")]
        finally:
            con.close()

    def summary(self, start: dt.datetime, end: dt.datetime, flow: str | None = None, alarms: int = 20) -> dict:
        """Totals, one row per hour of [start, end) and the last alarms."""
        if end <= start:
            raise ValueError("Bitiş zamanı başlangıçtan sonra olmalı")
        t0, t1 = start.timestamp(), end.timestamp()
        where, args = "ts >= ? AND ts < ?", [t0, t1]
        if flow:
            where += " AND flow = ?"
            args.append(flow)
        hours: dict[dt.datetime, list[int]] = {}
        h = start.replace(minute=0, second=0, microsecond=0)
        while h < end and len(hours) < MAX_HOURS:
            hours[h] = [0, 0, 0]
            h += dt.timedelta(hours=1)
        alarm_rows = []
        if self.path.exists():
            con = self._conn()
            try:
                for ts, g, b, t in con.execute(f"SELECT ts, good, bad, total FROM samples WHERE {where}", args):
                    key = dt.datetime.fromtimestamp(ts).replace(minute=0, second=0, microsecond=0)
                    if key in hours:
                        row = hours[key]
                        row[0] += g or 0
                        row[1] += b or 0
                        row[2] += t or 0
                alarm_rows = con.execute(f"SELECT ts, flow, text FROM alarms WHERE {where} ORDER BY ts DESC LIMIT ?",
                                         [*args, int(alarms)]).fetchall()
            finally:
                con.close()
        good = sum(r[0] for r in hours.values())
        bad = sum(r[1] for r in hours.values())
        total = sum(r[2] for r in hours.values())
        return {"start": start, "end": end, "flow": flow or "", "good": good, "bad": bad, "total": total,
                "yield": round(100.0 * good / total, 1) if total else None,
                "hours": [(k, *v) for k, v in hours.items()],
                "alarms": [(dt.datetime.fromtimestamp(ts), f, text) for ts, f, text in alarm_rows]}


# --- the report ----------------------------------------------------------------------------------------------------

def _yield_txt(good, total):
    return f"%{100.0 * good / total:.1f}" if total else "–"


def _svg_chart(hours) -> str:
    """Stacked good / bad bars per hour, inline SVG (no scripts, no external files)."""
    n = max(1, len(hours))
    w, h, pad_l, pad_b = 720, 220, 40, 34
    top = max([g + b for _, g, b, _ in hours] + [1])
    bw = (w - pad_l - 10) / n
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
             f'role="img" aria-label="Saatlik üretim grafiği" style="max-width:{w}px">',
             f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - 5}" y2="{h - pad_b}" stroke="#888"/>',
             f'<text x="{pad_l - 6}" y="16" font-size="11" text-anchor="end" fill="#555">{top}</text>',
             f'<text x="{pad_l - 6}" y="{h - pad_b}" font-size="11" text-anchor="end" fill="#555">0</text>']
    scale = (h - pad_b - 12) / top
    step = max(1, math.ceil(n / 24))           # at most ~24 hour labels
    for i, (hour, g, b, _t) in enumerate(hours):
        x = pad_l + i * bw + bw * 0.12
        bwid = max(1.0, bw * 0.76)
        gh, bh = g * scale, b * scale
        y = h - pad_b
        parts.append(f'<rect x="{x:.1f}" y="{y - gh:.1f}" width="{bwid:.1f}" height="{gh:.1f}" fill="#2e7d32">'
                     f'<title>{hour:%d.%m %H}:00 sağlam {g}</title></rect>')
        parts.append(f'<rect x="{x:.1f}" y="{y - gh - bh:.1f}" width="{bwid:.1f}" height="{bh:.1f}" fill="#c62828">'
                     f'<title>{hour:%d.%m %H}:00 hatalı {b}</title></rect>')
        if i % step == 0:
            parts.append(f'<text x="{x + bwid / 2:.1f}" y="{h - pad_b + 14}" font-size="10" text-anchor="middle" '
                         f'fill="#555">{hour:%H}</text>')
    parts.append(f'<rect x="{pad_l}" y="{h - 14}" width="10" height="10" fill="#2e7d32"/>'
                 f'<text x="{pad_l + 14}" y="{h - 5}" font-size="11" fill="#333">Sağlam</text>'
                 f'<rect x="{pad_l + 74}" y="{h - 14}" width="10" height="10" fill="#c62828"/>'
                 f'<text x="{pad_l + 88}" y="{h - 5}" font-size="11" fill="#333">Hatalı</text></svg>')
    return "".join(parts)


def _table_chart(hours) -> str:
    """The same chart as table cells, for the PDF (Qt's rich text has no SVG)."""
    top = max([g + b for _, g, b, _ in hours] + [1])
    rows = []
    for hour, g, b, _t in hours:
        gw, bw = round(300 * g / top), round(300 * b / top)
        cells = (f'<td bgcolor="#2e7d32" width="{gw}"></td>' if gw else "") + \
                (f'<td bgcolor="#c62828" width="{bw}"></td>' if bw else "") + '<td></td>'
        rows.append(f'<tr><td width="90" nowrap>{hour:%d.%m %H}:00</td><td><table cellspacing="0" cellpadding="2">'
                    f'<tr>{cells}</tr></table></td></tr>')
    return '<table cellspacing="0" cellpadding="1">' + "".join(rows) + "</table>"


def render_html(summary: dict, title: str = "Üretim raporu", flow_name: str = "", for_pdf: bool = False) -> str:
    e = html.escape
    s = summary
    flow_name = flow_name or s.get("flow") or "Tüm akışlar"
    hours = s["hours"]
    rows = "".join(f"<tr><td>{hr:%d.%m.%Y %H}:00–{(hr + dt.timedelta(hours=1)):%H}:00</td><td class=n>{g}</td>"
                   f"<td class=n>{b}</td><td class=n>{t}</td><td class=n>{_yield_txt(g, t)}</td></tr>"
                   for hr, g, b, t in hours if (g or b or t) or len(hours) <= 24)
    alarms = "".join(f"<tr><td>{ts:%d.%m.%Y %H:%M:%S}</td><td>{e(f)}</td><td>{e(text)}</td></tr>"
                     for ts, f, text in s["alarms"]) or '<tr><td colspan="3">Alarm yok</td></tr>'
    chart = _table_chart(hours) if for_pdf else _svg_chart(hours)
    made = dt.datetime.now()
    card = ('<td class=card><div class=k>{k}</div><div class=v style="color:{c}">{v}</div></td>')
    cards = "".join(card.format(k=k, v=v, c=c) for k, v, c in [
        ("Sağlam", s["good"], "#2e7d32"), ("Hatalı", s["bad"], "#c62828"), ("Toplam", s["total"], "#222"),
        ("Verim", _yield_txt(s["good"], s["total"]), "#1565c0")])
    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><title>{e(title)} – {e(flow_name)}</title>
<style>
body {{ font-family: "Segoe UI", Arial, sans-serif; color: #222; margin: 24px; background: #fff; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 22px 0 8px; }}
.meta {{ color: #555; font-size: 13px; }}
table {{ border-collapse: collapse; }}
table.grid td, table.grid th {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 13px; }}
table.grid th {{ background: #f0f0f0; text-align: left; }}
td.n {{ text-align: right; }}
td.card {{ border: 1px solid #ccc; padding: 8px 18px; }}
.k {{ font-size: 12px; color: #555; }} .v {{ font-size: 26px; font-weight: bold; }}
</style></head><body>
<h1>{e(title)}</h1>
<div class=meta>Akış: <b>{e(flow_name)}</b> · Dönem: {s['start']:%d.%m.%Y %H:%M} – {s['end']:%d.%m.%Y %H:%M}
 · Oluşturulma: {made:%d.%m.%Y %H:%M}</div>
<h2>Özet</h2>
<table cellspacing="6"><tr>{cards}</tr></table>
<h2>Saatlik üretim</h2>
{chart}
<table class=grid><tr><th>Saat</th><th>Sağlam</th><th>Hatalı</th><th>Toplam</th><th>Verim</th></tr>
{rows or '<tr><td colspan="5">Bu dönemde kayıt yok</td></tr>'}
</table>
<h2>Son alarmlar</h2>
<table class=grid><tr><th>Zaman</th><th>Akış</th><th>Alarm</th></tr>{alarms}</table>
<p class=meta>VisionFlow Studio üretim raporu</p>
</body></html>
"""


def write_pdf(html_text: str, path) -> str | None:
    """Render the report HTML to PDF with Qt (QTextDocument → QPdfWriter). Returns None on success, else the reason
    it was skipped (Qt missing, or no running Qt application, e.g. a headless run)."""
    try:
        from PySide6.QtGui import QGuiApplication, QPageSize, QPdfWriter, QTextDocument
    except Exception as exc:  # noqa: BLE001
        return f"PDF oluşturulamadı: Qt PDF desteği yok ({exc})"
    if QGuiApplication.instance() is None:
        return "PDF atlandı: PDF yalnızca arayüz açıkken oluşturulabilir (HTML rapor yazıldı)"
    try:
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setResolution(150)
        writer.setTitle("Üretim raporu")
        doc = QTextDocument()
        doc.setHtml(html_text)
        doc.print_(writer)
        del writer                                 # finishes the file
    except Exception as exc:  # noqa: BLE001
        return f"PDF oluşturulamadı: {exc}"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "PDF oluşturulamadı: dosya yazılamadı"
    return None


def make_report(db, start: dt.datetime, end: dt.datetime, out_html, flow: str | None = None,
                title: str = "Üretim raporu", pdf: bool = False) -> tuple[str, str | None, str | None]:
    """Write the HTML report (and the PDF next to it when asked). Returns (html path, pdf path or None,
    message about a skipped PDF or None)."""
    db = db if isinstance(db, ProductionDB) else ProductionDB(db)
    summary = db.summary(start, end, flow)
    out_html = pathlib.Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html(summary, title, flow or ""), encoding="utf-8")
    pdf_path, msg = None, None
    if pdf:
        target = out_html.with_suffix(".pdf")
        msg = write_pdf(render_html(summary, title, flow or "", for_pdf=True), target)
        pdf_path = None if msg else str(target)
    return str(out_html), pdf_path, msg
