"""Build or change a flow from a plain-language request, using Claude.

    python -m visionflow.assistant "Kırmızı kapakları say, 10'u geçince PLC'ye sinyal ver" -o akis.json

Needs ``pip install anthropic`` and an API key (AI Merkezi settings, or ANTHROPIC_API_KEY).
The model only proposes JSON; every block, port, type and parameter is checked here before a flow is
built, and errors are sent back once so the model can correct itself.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

from .core import Graph, GraphError, by_category, load_builtin, registry

MODEL = "claude-opus-5-5"
MAX_FIX_ROUNDS = 2

EXAMPLE_REQUESTS = [
    "Kameradan kırmızı nesneleri say, sayı 10'u geçince PLC'ye (Modbus) sinyal gönder",
    "Kamerada insan görünce fotoğrafını Telegram'a gönder, en fazla dakikada bir",
    "Bant üzerinde çizgiyi geçen ürünleri say ve her saatin sayısını veritabanına yaz",
    "Klasördeki ürün resimlerini anomali tespiti ile kontrol et, hatalıları ayrı klasöre kaydet",
    "Ekrandaki sayıyı OCR ile oku, 100'den büyükse sesli uyarı ver",
    "Etiketteki QR kodu oku, okunamazsa kırmızı uyarı yaz ve CSV'ye kaydet",
]


def catalog_text() -> str:
    """Compact description of every block: the model's whole vocabulary."""
    lines = []
    for cat, classes in sorted(by_category().items()):
        lines.append(f"\n## {cat}")
        for cls in sorted(classes, key=lambda c: c.type_name):
            ins = ", ".join(f"{n}:{t}" for n, t in cls.inputs) or "-"
            outs = ", ".join(f"{n}:{t}" for n, t in cls.outputs) or "-"
            ps = []
            for p in cls.params:
                s = f"{p.name}:{p.kind}={json.dumps(p.default, ensure_ascii=False)}"
                if p.choices:
                    s += " [" + "|".join(p.choices) + "]"
                ps.append(s)
            src = " (kaynak)" if getattr(cls, "is_source", False) else ""
            lines.append(f"- {cls.type_name}{src} \"{cls.title}\": {cls.description}\n"
                         f"  in: {ins} | out: {outs} | params: {'; '.join(ps) or '-'}")
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You design flows for VisionFlow Studio, a node-based computer vision and automation editor.
A flow is a directed acyclic graph of blocks. Data moves from an output port to an input port; port types
must match (image, number, bool, text, detections, contours), except that "any" accepts and feeds everything
and number and bool ports connect to each other (non-zero is true). Every input may stay unconnected. A flow needs at least one source
block (marked "kaynak") to run. Use only the blocks, ports and parameters listed below, with the exact
spelling. Set a parameter only when the request needs a value different from its default.

Answer in two parts:
1. One or two short sentences in Turkish telling the user what the flow does and anything they must fill in
   themselves (camera number, PLC IP address, Telegram token, model file...).
2. A single ```json code block with exactly this shape:
{{"nodes": [{{"id": "n1", "type": "<block type>", "values": {{"<param>": <value>}}}}],
  "connections": [{{"src": "n1", "src_port": "<output>", "dst": "n2", "dst_port": "<input>"}}]}}

Prefer a Preview (Önizleme) block on the main image so the user sees the result, and add a ShowValue block
for important numbers. Keep flows as small as the request allows. When a current flow is given, return the
complete changed flow (keep the ids of blocks you keep). Passwords, tokens and keys of the current flow are shown
as "<gizli>"; leave them as "<gizli>" and they are kept.

# Blocks
{catalog}
"""


@dataclass
class Proposal:
    graph: Graph
    explanation: str
    warnings: list[str] = field(default_factory=list)
    raw: str = ""


class AssistantError(Exception):
    pass


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    blob = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise AssistantError(f"Cevapta geçerli JSON yok: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise AssistantError("JSON'da 'nodes' listesi yok")
    return data


def _explanation(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S).strip()


def _coerce(param, value):
    return param.coerce(value)


HIDDEN = "<gizli>"
_SECRET = re.compile(r"password|token|secret|api_?key|şifre", re.I)


def _is_secret(name: str) -> bool:
    """Passwords, bot tokens and keys stay on this computer: they are never sent to the AI."""
    return bool(_SECRET.search(name))


def build_graph(data: dict, current: Graph | None = None) -> tuple[Graph, list[str]]:
    """Validate a proposed flow and build it. Returns (graph, errors); the graph holds whatever was valid.
    With ``current`` (changing an existing flow) blocks the AI kept by id keep their place on the canvas, their
    on/off state and their hidden passwords/tokens."""
    g, errors, ids = Graph(), [], {}
    old = current.nodes if current is not None else {}
    if not isinstance(data, dict):
        return g, ["Cevap bir JSON nesnesi olmalı: {\"nodes\": [...], \"connections\": [...]}"]
    nodes = data.get("nodes") or []
    if not isinstance(nodes, list):
        return g, ["'nodes' bir liste olmalı"]
    for i, nd in enumerate(nodes):
        if not isinstance(nd, dict):
            errors.append(f"Blok {i + 1}: {{\"id\", \"type\", \"values\"}} içeren bir nesne olmalı, {nd!r} değil")
            continue
        t, nid = nd.get("type"), str(nd.get("id") or f"n{i + 1}")
        if not isinstance(t, str) or t not in registry.NODE_REGISTRY:
            errors.append(f"Bilinmeyen blok tipi '{t}' (id {nid})")
            continue
        if nid in ids:
            errors.append(f"İki blok aynı id'yi kullanıyor: '{nid}'; her bloğa ayrı id verin")
            continue
        prev = old.get(nid)
        prev = prev if prev is not None and prev.type_name == t and nid not in g.nodes else None
        node = registry.create(t, nid if prev is not None else None)
        if prev is not None:
            node.pos, node.enabled, node.color = prev.pos, prev.enabled, prev.color
            node.values.update(prev.values)
        params = {p.name: p for p in node.params}
        values = nd.get("values") or {}
        if not isinstance(values, dict):
            errors.append(f"{nid} ({t}): 'values' bir nesne olmalı")
            values = {}
        for k, v in values.items():
            if k not in params:
                errors.append(f"{nid} ({t}): '{k}' diye bir parametre yok; olanlar: {', '.join(params) or '-'}")
                continue
            if v == HIDDEN:                       # the AI kept a hidden secret: keep the real one
                continue
            try:
                node.values[k] = _coerce(params[k], v)
            except (TypeError, ValueError) as exc:
                errors.append(f"{nid} ({t}).{k} = {v!r} geçersiz ({exc})")
        if prev is not None:                      # a kept sub-flow keeps the blocks inside it (and so its ports)
            node.load_extra(prev.extra(), [])
        g.add(node)
        ids[nid] = node.id
    conns = data.get("connections") or []
    if not isinstance(conns, list):
        errors.append("'connections' bir liste olmalı")
        conns = []
    for cd in conns:
        if not isinstance(cd, dict):
            errors.append(f"Bağlantı {cd!r}: {{\"src\", \"src_port\", \"dst\", \"dst_port\"}} nesnesi olmalı")
            continue
        src, dst = ids.get(str(cd.get("src"))), ids.get(str(cd.get("dst")))
        label = f"{cd.get('src')}.{cd.get('src_port')} → {cd.get('dst')}.{cd.get('dst_port')}"
        if not src or not dst:
            errors.append(f"Bağlantı {label}: blok bulunamadı")
            continue
        try:
            g.connect(src, str(cd.get("src_port")), dst, str(cd.get("dst_port")))
        except (GraphError, KeyError) as exc:
            s_ports = ", ".join(f"{n}:{t}" for n, t in g.nodes[src].outputs)
            d_ports = ", ".join(f"{n}:{t}" for n, t in g.nodes[dst].inputs)
            errors.append(f"Bağlantı {label}: {exc} (çıkışlar: {s_ports}; girişler: {d_ports})")
    if g.nodes and not any(getattr(n, "is_source", False) for n in g.nodes.values()):
        errors.append("Akışta kaynak blok yok (Kamera, Video, Resim, Test Görüntüsü...)")
    kept = [n for n in g.nodes.values() if n.id in old]
    if not kept:
        g.auto_layout()
    else:                                         # new blocks go to the right of the existing ones
        x = max(n.pos[0] for n in kept) + 260
        y = min(n.pos[1] for n in kept)
        for n in g.nodes.values():
            if n.id not in old:
                n.pos, y = (x, y), y + 160
    return g, errors


def _current_flow_json(graph: Graph) -> str:
    nodes = [{"id": n.id, "type": n.type_name,
              "values": {k: (HIDDEN if _is_secret(k) and v else v) for k, v in n.values.items()
                         if v != next(p.default for p in n.params if p.name == k)}}
             for n in graph.nodes.values() if n.type_name != "Note"]
    for nd in nodes:                    # a sub-flow's ports come from the blocks inside, which the AI does not see
        n = graph.nodes[nd["id"]]
        if n.type_name == "SubFlow":
            nd["ports"] = {"in": [f"{p}:{t}" for p, t in n.inputs], "out": [f"{p}:{t}" for p, t in n.outputs]}
    conns = [c.to_dict() for c in graph.connections]
    return json.dumps({"nodes": nodes, "connections": conns}, ensure_ascii=False)


def make_client(api_key: str | None = None):
    from .ai.claude import AIError, client_for
    try:
        return client_for(api_key=api_key)
    except AIError as exc:
        raise AssistantError(str(exc)) from None


def _ask(client, model: str, system: str, messages: list) -> str:
    from .ai.claude import AIError, _create
    try:
        resp = _create(client, model=model, max_tokens=16000, system=system, messages=messages,
                       output_config={"effort": "medium"})
    except AIError as exc:            # same messages, spending limit and usage log as the AI blocks
        raise AssistantError(str(exc)) from None
    if resp.stop_reason == "refusal":
        raise AssistantError("İstek yapay zeka tarafından reddedildi; farklı ifade etmeyi deneyin")
    if resp.stop_reason == "max_tokens":
        raise AssistantError("Cevap çok uzun oldu; isteği küçük parçalara bölün")
    return "".join(b.text for b in resp.content if b.type == "text")


def propose(request: str, current: Graph | None = None, client=None, model: str = MODEL,
            api_key: str | None = None) -> Proposal:
    """Ask the model for a flow; validate it; send errors back up to MAX_FIX_ROUNDS times."""
    if not request.strip():
        raise AssistantError("Ne yapmak istediğinizi yazın")
    client = client or make_client(api_key)
    system = SYSTEM_TEMPLATE.format(catalog=catalog_text())
    text = request.strip()
    if current is not None and current.nodes:
        text = f"Mevcut akış:\n```json\n{_current_flow_json(current)}\n```\n\nİstek: {text}"
    messages = [{"role": "user", "content": text}]
    errors: list[str] = []
    for _ in range(MAX_FIX_ROUNDS + 1):
        answer = _ask(client, model, system, messages)
        try:
            data = _extract_json(answer)
        except AssistantError as exc:
            errors = [str(exc)]
        else:
            graph, errors = build_graph(data, current)
            if not errors:
                return Proposal(graph, _explanation(answer), [], answer)
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": "Akışta şu hatalar var, düzeltip tam akışı tekrar ver:\n- "
                                                 + "\n- ".join(errors)}]
    if "graph" not in locals() or not graph.nodes:
        raise AssistantError("Geçerli bir akış oluşturulamadı: " + "; ".join(errors[:5]))
    return Proposal(graph, _explanation(answer), errors, answer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Yazılı istekten VisionFlow akışı oluşturur (Claude ile).")
    ap.add_argument("istek")
    ap.add_argument("-o", "--out", default="akis.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mevcut", help="değiştirilecek akış dosyası")
    args = ap.parse_args(argv)
    load_builtin()
    try:
        prop = propose(args.istek, Graph.load(args.mevcut) if args.mevcut else None, model=args.model)
    except AssistantError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    prop.graph.save(args.out)
    print(prop.explanation)
    for w in prop.warnings:
        print(f"Uyarı: {w}")
    print(f"Kaydedildi: {args.out} ({len(prop.graph.nodes)} blok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
