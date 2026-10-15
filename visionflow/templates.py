"""Flow templates for the "Yeni akış" gallery: the example flows with a title, description, categories and a few
simple questions whose answers are put into block settings.

A flow file may carry an optional ``"template"`` key (ignored when the flow is opened normally):

    {"title": "...", "description": "...", "categories": ["Kalite kontrol", "PLC"],
     "questions": [{"id": "tol", "label": "Tolerans ± (mm)", "kind": "float", "default": 0.5, "min": 0, "max": 10,
                    "targets": [{"node": "tolerans", "param": "low", "expr": "nominal - tol"}]}, ...]}

Question kinds: str, int, float, bool, choice (with "choices") and source (the picture source: sample pictures,
USB camera, IP camera, video file or picture folder; the source block is swapped). A target sets a block setting
(``param``, optionally computed with ``expr`` from the answers), or switches a block on/off (``enabled``).
The older examples get their text from EXAMPLE_META and a source question when they have one picture source."""
from __future__ import annotations

import ast
import copy
import json
import operator
import pathlib
import sys

from .core import registry
from .core.graph import Graph

EXAMPLES = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).resolve().parents[1])) / "examples"
CATEGORIES = ["Kalite kontrol", "Sayma", "Ölçüm", "PLC", "Yapay zeka", "Otomasyon"]

SAMPLE = "Örnek resimler"
SOURCES = [SAMPLE, "USB kamera", "IP kamera (RTSP/HTTP)", "Video dosyası", "Resim klasörü"]
SOURCE_HINT = {SOURCES[1]: "Kamera no (0, 1, …)", SOURCES[2]: "Adres (rtsp://kullanıcı:şifre@ip:554/…)",
               SOURCES[3]: "Video dosyasının yolu", SOURCES[4]: "Klasörün yolu"}
SOURCE_DEFAULT = {SOURCES[1]: "0", SOURCES[2]: "rtsp://192.168.1.64:554/stream1", SOURCES[3]: "", SOURCES[4]: ""}
SOURCE_TYPES = ("ImageFile", "ImageFolder", "Camera", "VideoFile", "TestPattern")

# the examples made before templates existed: title, categories, description
EXAMPLE_META = {
    "alt_akis_nesne_sayma": ("Alt akışla nesne sayma", ["Sayma"],
                             "Renkli nesneleri bir alt akış içinde bulup sayar: tekrar kullanılabilir blok grubu."),
    "anomali_kalite_kontrol": ("Anomali ile kalite kontrol", ["Kalite kontrol", "Yapay zeka"],
                               "Sadece sağlam parça örnekleriyle öğrenir; farklı görünen parçayı hatalı sayar."),
    "arastir_seslendir_video": ("Araştır, seslendir, video yap", ["Yapay zeka", "Otomasyon"],
                                "Bir konuyu internette araştırır, yapay zekayla metin yazar, seslendirip video yapar."),
    "az_ornekle_nesne_bulma": ("Az örnekle nesne bulma", ["Yapay zeka", "Sayma"],
                               "Birkaç resimde kutu çizerek öğretilen modelle parçaları bulur, sınıfına göre sayar."),
    "az_ornekle_segmentasyon": ("Az örnekle segmentasyon", ["Yapay zeka", "Kalite kontrol"],
                                "Birkaç işaretli örnekle parçayı/kusuru piksel piksel ayırır."),
    "bant_cizgi_gecis_sayma": ("Bant üzerinde çizgi geçiş sayma", ["Sayma"],
                               "Bantta akan nesneleri takip eder, çizgiyi geçenleri sayar."),
    "bolge_alarmi_veritabani": ("Bölge alarmı + veritabanı", ["Otomasyon", "Sayma"],
                                "Bölgeye giren nesneleri sayar, veritabanına yazar, grafik çizer ve bildirim atar."),
    "derin_anomali_patchcore": ("Derin anomali (PatchCore)", ["Kalite kontrol", "Yapay zeka"],
                                "Sağlam örneklerden öğrenen derin anomali modeliyle küçük kusurları bulur."),
    "doviz_kuru_bildirimi": ("Döviz kuru bildirimi", ["Otomasyon"],
                             "Belirli saatlerde döviz kurunu internetten alır ve sohbet kanalına gönderir."),
    "eposta_asistani": ("E-posta asistanı", ["Yapay zeka", "Otomasyon"],
                        "Gelen e-postaları okur, yapay zekayla cevap taslağı yazar."),
    "fatura_okuyucu": ("Fatura okuyucu", ["Yapay zeka", "Otomasyon"],
                       "Klasöre düşen faturaları okur, bilgileri çıkarır, veritabanına yazar ve dosyayı taşır."),
    "haber_ozeti_telegram": ("Haber özeti → Telegram", ["Yapay zeka", "Otomasyon"],
                             "Haber akışını özetleyip Telegram'a gönderir."),
    "hareket_olunca_video_kaydet": ("Hareket olunca video kaydet", ["Otomasyon"],
                                    "Kamerada hareket olunca video kaydı başlatır."),
    "hizli_siniflandirici_sekiller": ("Hızlı sınıflandırıcı (şekiller)", ["Yapay zeka", "Sayma"],
                                      "Birkaç örnekle öğretilen sınıflandırıcıyla nesneleri ayırır."),
    "kamera_yuz_tespiti_kayit": ("Yüz tespiti ve kayıt", ["Yapay zeka", "Otomasyon"],
                                 "Kamerada yüz görünce resmini kaydeder."),
    "kenar_algilama": ("Kenar algılama", ["Kalite kontrol"], "Gri tonlama, yumuşatma ve Canny ile kenarları bulur."),
    "kendi_kendine_ogrenen": ("Kendi kendine öğrenen sınıflandırıcı", ["Yapay zeka"],
                              "Emin olmadığı örnekleri toplar; onayladıkça daha iyi tanır."),
    "klasorden_sekil_tanima": ("Klasörden şekil tanıma", ["Sayma", "Kalite kontrol"],
                               "Klasördeki resimlerde şekilleri tanır, sınıfına göre sayar ve grafiğe çizer."),
    "opcua_sunucu": ("OPC UA / Modbus sunucusu", ["PLC", "Otomasyon"],
                     "Sayımları OPC UA ve Modbus sunucusu olarak yayınlar; PLC/SCADA okur."),
    "renkli_nesne_sayma": ("Renkli nesne sayma", ["Sayma"], "Belli renkteki nesneleri bulur ve sayar."),
    "ret_kapisi_simulasyon": ("Ret kapısı simülasyonu", ["PLC", "Otomasyon"],
                              "Hatalı parçayı ret kapısına doğru zamanda yönlendirir; OEE ve SPC hesaplar."),
    "siparis_toplayici": ("Sipariş toplayıcı", ["Otomasyon", "Yapay zeka"],
                          "Gelen siparişlerden bilgileri çıkarır, Excel'e yazar ve bildirim atar."),
    "telegram_asistani": ("Telegram asistanı", ["Yapay zeka", "Otomasyon"],
                          "Telegram'dan gelen soruları yapay zekayla cevaplar."),
    "tepsi_kontrol": ("Tepsi kontrol (boş göz)", ["Kalite kontrol", "Sayma"],
                      "Tepsideki gözlerin dolu/boş olduğunu ızgara ile kontrol eder."),
    "web_panel_canli_izleme": ("Web panelden canlı izleme", ["Otomasyon"],
                               "Akışın görüntüsünü ve sayılarını tarayıcıdan izlenebilen bir panelde yayınlar."),
    "yolo_kisi_sayma_bildirim": ("YOLO ile kişi sayma + bildirim", ["Yapay zeka", "Sayma"],
                                 "YOLO ile kişileri bulur, sayar ve webhook ile bildirir."),
}


def _clean_questions(raw) -> list[dict]:
    out = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q.get("id"):
            continue
        kind = q.get("kind", "str")
        if kind not in ("str", "int", "float", "bool", "choice", "source"):
            continue
        q = dict(q)
        q["kind"] = kind
        q["label"] = str(q.get("label") or q["id"])
        q["targets"] = [t for t in q.get("targets", []) if isinstance(t, dict) and isinstance(t.get("node"), str)]
        if kind == "choice" and not q.get("choices"):
            continue
        out.append(q)
    return out


def _auto_source_question(data: dict) -> list[dict]:
    srcs = [n for n in data.get("nodes", []) if isinstance(n, dict) and n.get("type") in SOURCE_TYPES]
    if len(srcs) != 1:
        return []
    return [{"id": "kaynak", "label": "Görüntü kaynağı", "kind": "source", "default": SAMPLE,
             "targets": [{"node": str(srcs[0].get("id"))}]}]


def load_template(path) -> dict | None:
    """The gallery entry of a flow file, or None when it is not a readable flow."""
    path = pathlib.Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        return None
    meta = data.get("template") if isinstance(data.get("template"), dict) else None
    if meta is not None:
        cats = meta.get("categories", meta.get("category", []))
        cats = [cats] if isinstance(cats, str) else [str(c) for c in cats if isinstance(c, (str,))]
        entry = {"title": str(meta.get("title") or path.stem.replace("_", " ")),
                 "description": str(meta.get("description") or ""), "categories": cats,
                 "questions": _clean_questions(meta.get("questions"))}
    else:
        title, cats, desc = EXAMPLE_META.get(path.stem, (path.stem.replace("_", " ").capitalize(), [], ""))
        entry = {"title": title, "description": desc, "categories": list(cats),
                 "questions": _auto_source_question(data)}
    entry.update(path=str(path), name=path.stem, data=data, new=meta is not None)
    return entry


def list_templates(folder=None) -> list[dict]:
    """Every flow of the examples folder as a gallery entry; the templates made for the gallery first."""
    folder = pathlib.Path(folder) if folder else EXAMPLES
    items = [t for t in (load_template(p) for p in sorted(folder.glob("*.json"))) if t is not None]
    items.sort(key=lambda t: (not t["new"], t["title"].casefold()))
    return items


_FOLD = str.maketrans("İIıŞşÇçĞğÖöÜü", "iiissccggoouu")


def fold(text: str) -> str:
    """For searching: Turkish capitals right (İ/I → i) and without accents, so 'ETİKET', 'olcum' and 'Ölçüm' all
    find what they mean."""
    return str(text).translate(_FOLD).casefold()


def matches(t: dict, category: str = "", text: str = "") -> bool:
    """Gallery filter: category ('' = all) and words to find in the title, description or categories."""
    if category and category not in t["categories"]:
        return False
    hay = fold(" ".join([t["title"], t["description"], " ".join(t["categories"]), t["name"]]))
    return all(w in hay for w in fold(text).split())


# --- answers → settings ---------------------------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(expr: str, names: dict) -> float:
    """A small arithmetic expression over the answers ("nominal - tol"): numbers, names, + - * / and brackets."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and \
                not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"bilinmeyen ad: {node.id}")
            v = names[node.id]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"'{node.id}' sayı değil")
            return v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError("izin verilmeyen ifade")
    try:
        return ev(ast.parse(str(expr), mode="eval"))
    except ZeroDivisionError:
        raise ValueError("sıfıra bölme") from None
    except SyntaxError:
        raise ValueError(f"ifade anlaşılmadı: {expr}") from None


def coerce_answer(q: dict, value):
    """The answer converted to the question's kind (Turkish ValueError when it does not fit)."""
    kind = q["kind"]
    label = q["label"]
    try:
        if kind == "source":
            if isinstance(value, dict):
                typ, addr = value.get("type", SAMPLE), value.get("address", "")
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                typ, addr = value
            else:
                typ, addr = value, None
            typ = str(typ)
            if typ not in SOURCES:
                raise ValueError(f"seçenekler: {', '.join(SOURCES)}")
            addr = SOURCE_DEFAULT.get(typ, "") if addr is None else str(addr).strip()
            if typ in (SOURCES[3], SOURCES[4]) and not addr:
                raise ValueError("dosya / klasör yolunu yazın")
            if typ == SOURCES[2] and not addr:
                raise ValueError("kamera adresini yazın")
            return {"type": typ, "address": addr or ("0" if typ == SOURCES[1] else "")}
        if kind == "bool":
            if isinstance(value, bool):
                return value
            s = str(value).strip().casefold()
            if s in ("1", "true", "evet", "yes", "on", "açık"):
                return True
            if s in ("0", "false", "hayır", "hayir", "no", "off", "kapalı", "", "none"):
                return False
            raise ValueError("evet/hayır bekleniyor")
        if kind in ("int", "float"):
            if isinstance(value, bool) or value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError("sayı bekleniyor")
            v = float(str(value).replace(",", ".")) if isinstance(value, str) else float(value)
            if v != v:
                raise ValueError("sayı bekleniyor")
            if kind == "int":
                if v != int(v):
                    raise ValueError("tam sayı bekleniyor")
                v = int(v)
            lo, hi = q.get("min"), q.get("max")
            if lo is not None and v < lo or hi is not None and v > hi:
                raise ValueError(f"{lo if lo is not None else '-∞'} ile {hi if hi is not None else '∞'} arasında "
                                 "olmalı")
            return v
        if kind == "choice":
            if str(value) not in [str(c) for c in q["choices"]]:
                raise ValueError(f"seçenekler: {', '.join(map(str, q['choices']))}")
            return str(value)
        return "" if value is None else str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{label}': {exc}") from None


def default_answers(t: dict) -> dict:
    return {q["id"]: q.get("default", SAMPLE if q["kind"] == "source" else None) for q in t["questions"]}


def _swap_source(data: dict, node_id: str, answer: dict) -> None:
    """Replace the picture source block of the flow dict by the chosen one (same id, place and image link)."""
    typ, addr = answer["type"], answer["address"]
    if typ == SAMPLE:
        return
    node = next((n for n in data["nodes"] if str(n.get("id")) == node_id), None)
    if node is None:
        raise ValueError(f"Şablonda kaynak blok yok: {node_id}")
    if typ in (SOURCES[1], SOURCES[2]):
        new_type, values = "Camera", {"source": addr or "0"}
    elif typ == SOURCES[3]:
        new_type, values = "VideoFile", {"path": addr}
    else:
        new_type, values = "ImageFolder", {"folder": addr, "loop": False}
    if node.get("type") == new_type:
        node.setdefault("values", {}).update(values)
    else:
        node.update(type=new_type, values=values)
        node.pop("disabled", None)
        outs = {name for name, _ in registry.NODE_REGISTRY[new_type].outputs}
        data["connections"] = [c for c in data.get("connections", [])
                               if not (isinstance(c, dict) and c.get("src") == node_id and c.get("src_port") not in outs)]


def apply_answers(t: dict, answers: dict | None = None) -> Graph:
    """A new flow from the template with the answers put into its blocks (no 'template' key left). Raises
    ValueError with a Turkish message when an answer does not fit."""
    data = copy.deepcopy(t["data"])
    data.pop("template", None)
    given = dict(answers or {})
    vals: dict = {}
    for q in t["questions"]:
        raw = given.get(q["id"], q.get("default", SAMPLE if q["kind"] == "source" else None))
        vals[q["id"]] = coerce_answer(q, raw)
    nodes = {str(n.get("id")): n for n in data.get("nodes", []) if isinstance(n, dict)}
    for q in t["questions"]:
        v = vals[q["id"]]
        for tg in q["targets"]:
            if q["kind"] == "source":
                _swap_source(data, tg["node"], v)
                continue
            node = nodes.get(tg["node"])
            if node is None:
                raise ValueError(f"'{q['label']}': şablonda '{tg['node']}' bloğu yok")
            if "enabled" in tg:
                on = bool(v) == bool(tg["enabled"])
                if on:
                    node.pop("disabled", None)
                else:
                    node["disabled"] = True
                continue
            if not tg.get("param"):
                continue
            value = v
            if tg.get("expr"):
                try:
                    value = _eval(tg["expr"], vals)
                except ValueError as exc:
                    raise ValueError(f"'{q['label']}': {exc}") from None
            cls = registry.NODE_REGISTRY.get(node.get("type"))
            param = next((p for p in cls.params if p.name == tg["param"]), None) if cls else None
            if param is not None:
                try:
                    value = param.coerce(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"'{q['label']}': {param.display}: {exc}") from None
            node.setdefault("values", {})[tg["param"]] = value
    return Graph.from_dict(data)
