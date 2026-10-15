"""Regression test of a flow against a recording ("Akış Testi"):

    python -m visionflow test-akis akis.json --kayit kayit_klasoru --beklenen beklenen.json [--kaydet-beklenen]

The recording ("Kare Kaydedici") is replayed through the flow as fast as possible; the chosen outputs of every frame
are compared with the expected values, so a changed flow can be checked to still decide the same.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

from .core import Graph, PortType, load_builtin, load_plugins, registry

EXPECTED_FORMAT = "visionflow-beklenen"
SCALAR = (PortType.BOOL, PortType.NUMBER)


class FlowTestError(Exception):
    """A user error (missing file, no source...); printed in Turkish, exit code 2."""


def prepare(graph: Graph, folder: str) -> list[str]:
    """Make the flow read the recording. Its 'Kayıttan Oynat' blocks get the folder; otherwise its single image
    source (camera, video...) is replaced by one. Returns notes for the user."""
    from .nodes.recording import load_manifest
    try:
        load_manifest(folder)
    except ValueError as exc:
        raise FlowTestError(str(exc)) from exc
    notes = []
    for n in graph.nodes.values():
        if n.type_name == "FrameRecorder" and n.enabled:   # do not record the replay into a new recording
            n.enabled = False
            notes.append(f"'{n.title}' bloğu testte kapalı")
    players = [n.title for n in graph.nodes.values() if n.type_name == "RecordingPlayer"]
    for n in graph.nodes.values():
        if n.type_name == "RecordingPlayer":
            n.values.update(folder=str(folder), speed="en hızlı", loop=False)
            n.enabled = True
        elif n.type_name == "SubFlow":        # the same for the blocks inside sub-flows (saved in their inner flow)
            found = _prepare_inner(n.inner, str(folder), notes, f"{n.title} › ")
            if found:
                players += found
            n.refresh()
    if players:
        return notes + [f"'{t}' bloğu kayıttan okuyor" for t in players]
    cands = [n for n in graph.nodes.values() if n.enabled and "image" in dict(n.outputs)
             and dict(n.outputs)["image"] == PortType.IMAGE and not n.inputs
             and (n.is_source or n.type_name == "ImageFile")]
    if not cands:
        raise FlowTestError("Akışta kayıt yerine geçecek bir görüntü kaynağı (kamera, video, resim) yok; "
                            "akışa bir 'Kayıttan Oynat' bloğu ekleyin")
    if len(cands) > 1:
        raise FlowTestError("Akışta birden fazla görüntü kaynağı var (" + ", ".join(n.title for n in cands) +
                            "); test edilecek olanı 'Kayıttan Oynat' bloğuyla değiştirin")
    old = cands[0]
    new = registry.create("RecordingPlayer", old.id)
    new.pos = old.pos
    new.values.update(folder=str(folder), speed="en hızlı", loop=False)
    graph.nodes[old.id] = new
    notes.append(f"'{old.title}' yerine kayıt oynatılıyor")
    keep = []
    for c in graph.connections:
        if old.id in (c.src, c.dst) and not (c.src == old.id and c.src_port == "image"):
            notes.append(f"'{old.title}' bloğunun '{c.src_port if c.src == old.id else c.dst_port}' bağlantısı "
                         "testte kullanılmıyor")
            continue
        keep.append(c)
    graph.connections = keep
    return notes


def _prepare_inner(inner, folder: str, notes: list[str], label: str, depth: int = 0) -> list[str]:
    """prepare() inside a sub-flow's inner flow (a dict): recorders off, players read `folder`. Returns the titles
    of the players found."""
    from .nodes.subflow import MAX_DEPTH, _title_of
    if not isinstance(inner, dict) or depth > MAX_DEPTH:
        return []
    players = []
    for nd in inner.get("nodes") or []:
        if not isinstance(nd, dict):
            continue
        t = nd.get("type")
        if t == "FrameRecorder" and not nd.get("disabled"):
            nd["disabled"] = True
            notes.append(f"'{label}{_title_of(nd)}' bloğu testte kapalı")
        elif t == "RecordingPlayer":
            if not isinstance(nd.get("values"), dict):
                nd["values"] = {}
            nd["values"].update(folder=folder, speed="en hızlı", loop=False)
            nd["disabled"] = False
            players.append(label + _title_of(nd))
        elif t == "SubFlow":
            players += _prepare_inner(nd.get("inner"), folder, notes, f"{label}{_title_of(nd)} › ", depth + 1)
    return players


def resolve_outputs(graph: Graph, refs: list[str]) -> list[str]:
    """'blok.port' (block id or title) → 'id.port' keys."""
    keys = []
    for ref in refs:
        ref = ref.strip()
        if not ref:
            continue
        if "." not in ref:
            raise FlowTestError(f"Çıkış 'blok.port' biçiminde olmalı: {ref}")
        block, port = ref.rsplit(".", 1)
        node = graph.nodes.get(block)
        if node is None:
            same = [n for n in graph.nodes.values() if n.title.lower() == block.lower()]
            if len(same) > 1:
                raise FlowTestError(f"'{block}' adlı birden fazla blok var; blok kimliğini kullanın "
                                    f"({', '.join(n.id for n in same)})")
            node = same[0] if same else None
        if node is None:
            raise FlowTestError(f"Blok bulunamadı: {block}")
        if port not in dict(node.outputs):
            raise FlowTestError(f"'{node.title}' bloğunun '{port}' çıkışı yok "
                                f"(çıkışlar: {', '.join(p for p, _ in node.outputs) or '-'})")
        keys.append(f"{node.id}.{port}")
    if not keys:
        raise FlowTestError("Karşılaştırılacak çıkış yok")
    return keys


def default_outputs(graph: Graph) -> list[str]:
    """Every yes/no and number output of the blocks after the source: the decisions of the flow."""
    keys = [f"{n.id}.{p}" for n in graph.nodes.values() if n.enabled and not n.is_source
            and n.type_name != "RecordingPlayer" for p, t in n.outputs if t in SCALAR]
    if not keys:
        raise FlowTestError("Akışta karşılaştırılabilecek sayı / evet-hayır çıkışı yok; --cikislar ile seçin")
    return keys


def label(graph: Graph, key: str) -> str:
    nid, port = key.rsplit(".", 1)
    node = graph.nodes.get(nid)
    return f"{node.title}.{port}" if node else key


def run_recording(graph: Graph, keys: list[str], max_frames: int = 0) -> list[dict]:
    from .nodes.recording import jsonable
    rows = []

    def on_frame(n, results):
        row = {}
        for key in keys:
            nid, port = key.rsplit(".", 1)
            r = results.get(nid)
            if r is None:
                row[key] = None
            elif r.error:
                row[key] = f"HATA: {r.error}"
            else:
                row[key] = jsonable(r.outputs.get(port))
        rows.append(row)

    graph.run(max_frames=max_frames or None, on_frame=on_frame, max_fps=0)
    return rows


def same(a, b, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= max(tol, tol * abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    return a == b


def compare(expected: list[dict], actual: list[dict], keys: list[str], tol: float) -> dict:
    """{key: {"ok": n, "bad": n, "first": [(frame, expected, actual), ...]}}"""
    report = {}
    for key in keys:
        ok, bad, first = 0, 0, []
        for i in range(max(len(expected), len(actual))):
            missing = object()
            e = expected[i].get(key, missing) if i < len(expected) else missing
            a = actual[i].get(key, missing) if i < len(actual) else missing
            if e is not missing and a is not missing and same(a, e, tol):
                ok += 1
            else:
                bad += 1
                if len(first) < 3:
                    first.append((i + 1, "(yok)" if e is missing else e, "(yok)" if a is missing else a))
        report[key] = {"ok": ok, "bad": bad, "first": first}
    return report


def table(rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line]
    for j, r in enumerate(rows):
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
        if j == 0:
            out.append(line)
    out.append(line)
    return "\n".join(out)


def short(v, n: int = 40) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 1] + "…"


def main(argv=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="python -m visionflow test-akis",
                                 description="Bir kaydı akıştan geçirip seçilen çıkışları beklenen değerlerle "
                                             "karşılaştırır (akış değişince aynı kararları veriyor mu?)")
    ap.add_argument("flow", help="akış dosyası (.json)")
    ap.add_argument("--kayit", required=True, help="'Kare Kaydedici' ile alınmış kayıt klasörü")
    ap.add_argument("--beklenen", required=True, help="beklenen değerler dosyası (.json)")
    ap.add_argument("--kaydet-beklenen", action="store_true",
                    help="karşılaştırma yapma: bu akışın çıktılarını beklenen dosyasına yaz")
    ap.add_argument("--cikislar", default="",
                    help="karşılaştırılacak çıkışlar, virgülle: 'Blok başlığı.port' veya 'blokid.port' "
                         "(varsayılan: beklenen dosyasındakiler, yoksa tüm sayı / evet-hayır çıkışları)")
    ap.add_argument("--tolerans", type=float, default=1e-6, help="sayılar için izin verilen fark")
    ap.add_argument("--kare", type=int, default=0, help="en fazla bu kadar kare (0 = kaydın tamamı)")
    ap.add_argument("--plugins", default="plugins", help="ek blok klasörü")
    args = ap.parse_args(argv)
    try:
        return _run(args, out)
    except FlowTestError as exc:
        out(f"HATA: {exc}")
        return 2


def _run(args, out) -> int:
    load_builtin()
    load_plugins(args.plugins)
    try:
        graph = Graph.load(args.flow)
    except FileNotFoundError:
        raise FlowTestError(f"Akış dosyası bulunamadı: {args.flow}") from None
    except (OSError, ValueError) as exc:
        raise FlowTestError(f"Akış dosyası okunamadı: {exc}") from None
    for w in graph.warnings:
        out(f"Uyarı: {w}")
    for note in prepare(graph, args.kayit):
        out(f"Not: {note}")
    exp_path = pathlib.Path(args.beklenen)
    expected = None
    if not args.kaydet_beklenen:
        try:
            expected = json.loads(exp_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FlowTestError(f"Beklenen dosyası yok: {exp_path} (önce --kaydet-beklenen ile oluşturun)") from None
        except (OSError, ValueError) as exc:
            raise FlowTestError(f"Beklenen dosyası okunamadı: {exc}") from None
        if not isinstance(expected, dict) or expected.get("format") != EXPECTED_FORMAT \
                or not isinstance(expected.get("kareler"), list) or not isinstance(expected.get("cikislar"), list):
            raise FlowTestError(f"{exp_path} bir beklenen değerler dosyası değil")
    if args.cikislar:
        keys = resolve_outputs(graph, args.cikislar.split(","))
    elif expected is not None:
        keys = [str(k) for k in expected["cikislar"]]
        missing = [k for k in keys if "." not in k or k.rsplit(".", 1)[0] not in graph.nodes
                   or k.rsplit(".", 1)[1] not in dict(graph.nodes[k.rsplit(".", 1)[0]].outputs)]
        if missing:
            raise FlowTestError("Beklenen dosyasındaki çıkışlar akışta yok: " + ", ".join(missing) +
                                " (blok silinmiş olabilir; --kaydet-beklenen ile yenileyin)")
    else:
        keys = default_outputs(graph)

    rows = run_recording(graph, keys, args.kare)
    if not rows:
        raise FlowTestError("Akış hiç kare işlemedi (kaynak başlatılamadı mı?)")
    errors = sorted({v for r in rows for v in r.values() if isinstance(v, str) and v.startswith("HATA: ")})
    for e in errors[:5]:
        out(f"Uyarı: blok hatası: {e[6:]}")

    if args.kaydet_beklenen:
        data = {"format": EXPECTED_FORMAT, "version": 1, "akis": pathlib.Path(args.flow).name,
                "kayit": str(args.kayit), "olusturma": dt.datetime.now().isoformat(timespec="seconds"),
                "cikislar": keys, "basliklar": {k: label(graph, k) for k in keys}, "kareler": rows}
        try:
            if exp_path.parent:
                exp_path.parent.mkdir(parents=True, exist_ok=True)
            exp_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            raise FlowTestError(f"Beklenen dosyası yazılamadı: {exc}") from None
        out(f"Beklenen değerler yazıldı: {exp_path} ({len(rows)} kare, {len(keys)} çıkış)")
        return 0

    exp_rows = [r if isinstance(r, dict) else {} for r in expected["kareler"]]
    report = compare(exp_rows, rows, keys, args.tolerans)
    titles = expected.get("basliklar") if isinstance(expected.get("basliklar"), dict) else {}
    lines = [["Çıkış", "Kare", "Uyan", "Uymayan", "Sonuç"]]
    for k in keys:
        r = report[k]
        lines.append([label(graph, k) if k.rsplit(".", 1)[0] in graph.nodes else titles.get(k, k),
                      max(len(exp_rows), len(rows)), r["ok"], r["bad"], "GEÇTİ" if not r["bad"] else "KALDI"])
    out(table(lines))
    if len(exp_rows) != len(rows):
        out(f"Kare sayısı farklı: beklenen {len(exp_rows)}, bulunan {len(rows)}")
    failed = [k for k in keys if report[k]["bad"]]
    for k in failed:
        for frame, e, a in report[k]["first"]:
            out(f"  {label(graph, k)} kare {frame}: beklenen {short(e)}, bulunan {short(a)}")
    if failed or len(exp_rows) != len(rows):
        out(f"SONUÇ: KALDI ({len(failed)}/{len(keys)} çıkış farklı)")
        return 1
    out(f"SONUÇ: GEÇTİ ({len(keys)} çıkış, {len(rows)} kare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
