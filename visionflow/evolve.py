"""Kendini Geliştir: the program examines itself, asks Claude for a fix or an improvement of its own code, tests the
change on a copy, corrects itself when the tests fail, and applies it only when the user approves.

Safety rules (they are enforced here, and this file itself can never be changed by a proposal):
- a change is never applied without the user's approval; before that it only runs in a temporary copy
- only the program's own files are changed (visionflow/, plugins/, tests/); in the packaged .exe only plugins/
- the spending limit, settings and this module are off limits
- every applied change is backed up and can be undone with one click
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field

from .ai.config import AIConfig, home

PKG = pathlib.Path(__file__).resolve().parent            # .../visionflow (the package)
SOURCE_ROOT = PKG.parent                                  # folder with visionflow/, tests/, plugins/
PROTECTED = {"visionflow/evolve.py", "visionflow/ai/claude.py", "visionflow/ai/config.py"}
MAX_ROUNDS = 3                    # proposal → test → "this failed, fix it" rounds before giving up
MAX_CONTEXT_CHARS = 160_000       # source code sent with one question (about 40k tokens)


class EvolveError(Exception):
    pass


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def editable_root() -> pathlib.Path:
    """Where changes go: the source folder, or next to the .exe (only its plugins/ folder) in the packaged app."""
    return pathlib.Path(sys.executable).parent if frozen() else SOURCE_ROOT


PROTECTED_INIT = {"visionflow/__init__.py", "visionflow/ai/__init__.py", "visionflow/__main__.py"}
BAD_CHARS = set('<>:"|?*')


def allowed(rel: str) -> str | None:
    """None if the program may change this file, else why not."""
    rel = _clean(rel)
    p = pathlib.PurePosixPath(rel)
    parts = rel.split("/")
    if not rel or p.is_absolute() or any(part in ("", ".", "..") for part in parts) or BAD_CHARS & set(rel) \
            or any(part != part.strip() or part.endswith(".") for part in parts):
        # (Windows drops a trailing dot or space: "ai./claude.py" would be ai/claude.py)
        return "dosya yolu programın klasörü içinde olmalı"
    folded = rel.casefold()                  # Windows: visionflow/Evolve.py is the same file as evolve.py
    if p.suffix != ".py":
        return "sadece .py dosyaları değiştirilebilir"
    shadows = any(folded.startswith(prot[:-3] + "/") for prot in PROTECTED)   # ai/config/__init__.py beats config.py
    if folded in PROTECTED or folded in PROTECTED_INIT or shadows:
        return "bu dosya korumalı (güvenlik, harcama sınırı veya kendini geliştirme kuralları)"
    if frozen():
        return None if parts[0] == "plugins" and len(parts) == 2 else \
            "paketlenmiş programda sadece plugins/ klasörüne eklenti yazılabilir"
    if parts[0] in ("visionflow", "tests") or (parts[0] == "plugins" and len(parts) == 2):
        return None
    return "sadece visionflow/, tests/ ve plugins/ altındaki dosyalar değiştirilebilir"


def code_problems(text: str) -> str | None:
    """New code must not reach for the self-improvement rules themselves (they are checked on every change)."""
    if re.search(r"\bevolve\b|\bPROTECTED\b", text):
        return "değişiklik Kendini Geliştir kurallarına (evolve) dokunamaz"
    return None


# --- what went wrong while the program was used --------------------------------------------------------------

def error_key(block_type, error: str) -> tuple:
    return block_type, re.sub(r"\d+", "#", error or "")[:160]


class ErrorLog:
    """Errors blocks hit while flows ran, one line each in hata_gunlugu.jsonl; the same error is written once per
    run, so a camera failing 30 times a second does not fill the disk."""
    MAX_LINES = 2000

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path or home() / "hata_gunlugu.jsonl"
        self._seen: set = set()
        self._lock = threading.Lock()

    def record(self, block_type: str, error: str, where: str = "") -> None:
        if not error or error == "Önceki blok hata verdi":
            return
        key = error_key(block_type, error)        # "frame 1201" and "frame 1202" are the same error
        with self._lock:
            if key in self._seen:
                return
            if len(self._seen) > 5000:
                self._seen.clear()
            self._seen.add(key)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "block": block_type,
                                        "error": error[:500], "where": where[:300]}, ensure_ascii=False) + "\n")
                if self.path.stat().st_size > 400_000:            # keep the newest lines, well under the limit
                    lines, size = [], 0
                    for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
                        size += len(line.encode("utf-8")) + 1
                        if size > 150_000 or len(lines) >= self.MAX_LINES // 2:
                            break
                        lines.append(line)
                    self.path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
            except OSError:
                pass

    def summary(self, limit: int = 25) -> list[dict]:
        """Distinct errors, most frequent first: {'block', 'error', 'where', 'count', 'last'}."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        groups: dict = {}
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            k = error_key(e.get("block"), e.get("error", ""))
            g = groups.setdefault(k, {"block": e.get("block"), "error": e.get("error"), "where": e.get("where", ""),
                                      "count": 0, "last": ""})
            g["count"] += 1
            g["last"] = max(g["last"], e.get("time", ""))
            g["where"] = e.get("where") or g["where"]
        return sorted(groups.values(), key=lambda g: (-g["count"], g["last"]))[:limit]

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        self._seen.clear()


ERRORS = ErrorLog.__new__(ErrorLog)              # created lazily so VISIONFLOW_HOME set by tests is honoured


def error_log() -> ErrorLog:
    global ERRORS
    if not hasattr(ERRORS, "path") or ERRORS.path != home() / "hata_gunlugu.jsonl":
        ERRORS = ErrorLog()
    return ERRORS


# --- examining itself ---------------------------------------------------------------------------------------------

def run_checks(root: pathlib.Path | None = None, timeout: int = 900) -> tuple[bool, str]:
    """The program's own tests (when its source and pytest are there) and its installation self-test, run in a
    separate process with a throw-away settings folder. Returns (all passed, last lines of the output)."""
    root = root or SOURCE_ROOT
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONUTF8": "1",
           "VISIONFLOW_HOME": tempfile.mkdtemp(prefix="vf-dene-")}
    env.pop("PYTHONPATH", None)
    outputs, ok = [], True
    commands = []
    if (root / "tests").is_dir() and _has_pytest():
        commands.append(("Testler", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                     "--maxfail=5", "tests"]))
    commands.append(("Kurulum testi", [sys.executable, "-m", "visionflow", "--selftest", "--no-ui",
                                       "--out", os.path.join(env["VISIONFLOW_HOME"], "rapor.txt")]))
    for label, cmd in commands:
        try:
            r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout)
            out, code = (r.stdout + r.stderr), r.returncode
        except subprocess.TimeoutExpired:
            out, code = f"{timeout} sn içinde bitmedi", 1
        ok = ok and code == 0
        tail = "\n".join(out.strip().splitlines()[-40:])
        outputs.append(f"== {label}: {'GEÇTİ' if code == 0 else 'BAŞARISIZ'}\n{tail}")
    shutil.rmtree(env["VISIONFLOW_HOME"], ignore_errors=True)
    return ok, "\n\n".join(outputs)


def _has_pytest() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("pytest") is not None
    except Exception:  # noqa: BLE001
        return False


def self_report(run_tests: bool = True) -> dict:
    """Everything the program knows about its own health."""
    from . import __version__
    from .core.registry import PLUGIN_ERRORS
    rep = {"version": __version__, "frozen": frozen(), "errors": error_log().summary(),
           "plugin_errors": dict(PLUGIN_ERRORS), "time": dt.datetime.now().isoformat(timespec="seconds")}
    if run_tests and not frozen():
        rep["tests_ok"], rep["tests"] = run_checks()
    return rep


def report_text(rep: dict) -> str:
    lines = [f"VisionFlow {rep['version']} kendini inceleme raporu ({rep['time']})"]
    if rep["errors"]:
        lines.append(f"\nKullanırken karşılaşılan hatalar ({len(rep['errors'])} farklı):")
        for e in rep["errors"]:
            lines.append(f"- [{e['count']}×] {e['block']}: {e['error']}" + (f"  ({e['where']})" if e["where"] else ""))
    else:
        lines.append("\nKullanırken kaydedilmiş hata yok.")
    for name, err in rep["plugin_errors"].items():
        lines.append(f"- Eklenti {name} yüklenemedi: {err}")
    if "tests" in rep:
        lines.append(f"\nKendi testleri: {'hepsi geçti' if rep['tests_ok'] else 'BAŞARISIZ'}")
        if not rep["tests_ok"]:
            lines.append(rep["tests"])
    return "\n".join(lines)


# --- asking Claude for a change ----------------------------------------------------------------------------------

def _new_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def _clean(rel: str) -> str:
    rel = rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


@dataclass
class Edit:
    path: str                     # relative to the program folder, e.g. "visionflow/nodes/more.py"
    old: str = ""                 # exact text to replace ("" with new_file=True: create the file)
    new: str = ""
    new_file: bool = False


@dataclass
class Proposal:
    summary: str
    edits: list[Edit]
    rounds: int = 1
    tests_ok: bool | None = None
    test_output: str = ""
    goal: str = ""
    id: str = field(default_factory=lambda: _new_id())
    fingerprint: str = ""          # code_fingerprint() of the program the tests passed on

    def diff(self) -> str:
        """Readable diff for the approval window."""
        import difflib
        out = []
        root = editable_root()
        for e in self.edits:
            if e.new_file:
                out.append(f"+++ YENİ DOSYA {e.path}\n" + "\n".join("+" + ln for ln in e.new.splitlines()))
                continue
            try:
                text = (root / e.path).read_text(encoding="utf-8")
                start = text[:text.index(e.old)].count("\n") + 1 if e.old in text else 0
            except OSError:
                start = 0
            out.append("".join(difflib.unified_diff((e.old + "\n").splitlines(True), (e.new + "\n").splitlines(True),
                                                    f"{e.path} (satır {start})", e.path, n=2)))
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"id": self.id, "summary": self.summary, "goal": self.goal, "rounds": self.rounds,
                "tests_ok": self.tests_ok, "test_output": self.test_output, "fingerprint": self.fingerprint,
                "edits": [e.__dict__ for e in self.edits]}

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(d.get("summary", ""), [Edit(**e) for e in d.get("edits", [])], d.get("rounds", 1),
                   d.get("tests_ok"), d.get("test_output", ""), d.get("goal", ""), d.get("id") or _new_id(),
                   d.get("fingerprint", ""))


SYSTEM = """You improve VisionFlow Studio, a Python/PySide6 desktop app (node editor for computer vision and
automation; UI text is Turkish). You are given a self-examination report and some of the app's own source files.
Make the smallest correct change that achieves the goal: fix the reported errors at their cause, or add the
requested improvement. Keep the code style of the file. When you fix a bug, also add or extend a pytest test in
tests/ that would have caught it, if a tests/ folder exists.

Rules you must follow:
- Only change files under {allowed}. Never touch: {protected}.
- Do not add network calls, new third-party packages, or code that deletes user files.
- Never delete, skip or weaken an existing test to make the tests pass; fix the code instead.
- For an edit, "eski" must be copied EXACTLY from the given file (including indentation) and appear once in it.
- If nothing should change (the errors are caused by the user's setup, e.g. a missing camera), return no changes
  and explain in "ozet" what the user should do.

Answer with one JSON object only:
{{"ozet": "Turkish explanation for the user, 1-4 sentences",
  "degisiklikler": [{{"dosya": "visionflow/nodes/x.py", "eski": "exact old text", "yeni": "new text"}},
                    {{"dosya": "plugins/yeni.py", "yeni_dosya": true, "yeni": "full content of a new file"}}]}}
"""


def _files_for(rep: dict, goal: str, limit: int = MAX_CONTEXT_CHARS) -> dict[str, str]:
    """The source files the question is about: those in the error locations, those of the blocks named in the
    goal or the errors, then core files, until the size limit."""
    root = editable_root()
    if frozen():
        return {f"plugins/{p.name}": p.read_text(encoding="utf-8", errors="replace")
                for p in sorted((root / "plugins").glob("*.py"))} if (root / "plugins").is_dir() else {}
    wanted: list[str] = []
    for e in rep.get("errors", []):
        wanted += re.findall(r"(visionflow/[\w/]+\.py):\d+", e.get("where", ""))
    from .core.registry import NODE_REGISTRY
    text = (goal + " " + " ".join(f"{e.get('block')}" for e in rep.get("errors", []))).casefold()
    for tname, cls in NODE_REGISTRY.items():
        if tname.casefold() in text or (cls.title and cls.title.casefold() in text):
            mod = sys.modules.get(cls.__module__)
            f = getattr(mod, "__file__", None)
            if f:
                try:
                    wanted.append(pathlib.Path(f).resolve().relative_to(root).as_posix())
                except ValueError:
                    pass
    wanted += ["visionflow/core/node.py", "visionflow/core/graph.py"]
    out, total = {}, 0
    for rel in dict.fromkeys(wanted):
        p = root / rel
        if rel in out or not p.is_file():
            continue
        body = p.read_text(encoding="utf-8", errors="replace")
        if total + len(body) > limit:
            continue
        out[rel], total = body, total + len(body)
    return out


def _tree() -> str:
    root = editable_root()
    names = sorted(p.relative_to(root).as_posix() for d in ("visionflow", "plugins", "tests") if (root / d).is_dir()
                   for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts)
    return "\n".join(names)


def _parse(answer: str) -> tuple[str, list[Edit]]:
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        raise EvolveError("Cevapta JSON yok")
    try:
        data = json.loads(m.group(0))
    except ValueError as exc:
        raise EvolveError(f"Cevaptaki JSON bozuk: {exc}") from None
    edits = []
    for d in data.get("degisiklikler") or []:
        if not isinstance(d, dict) or not isinstance(d.get("dosya"), str) or not isinstance(d.get("yeni"), str):
            raise EvolveError(f"Geçersiz değişiklik: {str(d)[:200]}")
        edits.append(Edit(_clean(d["dosya"]), d.get("eski") or "", d["yeni"],
                          bool(d.get("yeni_dosya"))))
    return str(data.get("ozet") or ""), edits


def _result_texts(edits: list[Edit], root: pathlib.Path) -> tuple[dict, list[str]]:
    """Every changed file's final text, applying the edits one after another exactly as apply() will, and the
    problems found on the way. Each 'eski' must appear exactly once in the text as it is at that point."""
    texts: dict[str, str] = {}
    problems = []
    for e in edits:
        why = allowed(e.path)
        if why:
            problems.append(f"{e.path}: {why}")
            continue
        bad = code_problems(e.new)
        if bad:
            problems.append(f"{e.path}: {bad}")
            continue
        p = root / e.path
        if e.new_file:
            if p.exists() or e.path in texts:
                problems.append(f"{e.path}: dosya zaten var; yeni dosya yerine değişiklik verin")
                continue
            texts[e.path] = e.new
            continue
        if e.path not in texts:
            if not p.is_file():
                problems.append(f"{e.path}: böyle bir dosya yok")
                continue
            texts[e.path] = p.read_text(encoding="utf-8")
        n = texts[e.path].count(e.old) if e.old else 0
        if n != 1:
            problems.append(f"{e.path}: 'eski' metin dosyada {'bulunamadı' if n == 0 else f'{n} kez geçiyor'}; "
                            "dosyadan aynen kopyalayın ve tek bir yeri gösterecek kadar uzun tutun "
                            "(aynı dosyadaki önceki değişiklikler uygulandıktan sonraki hâline göre)")
            continue
        texts[e.path] = texts[e.path].replace(e.old, e.new, 1)
    return texts, problems


def check_edits(edits: list[Edit], root: pathlib.Path | None = None) -> list[str]:
    """Problems that stop a change from being tried at all."""
    texts, problems = _result_texts(edits, root or editable_root())
    for rel, text in texts.items():
        try:
            compile(text, rel, "exec")
        except SyntaxError as exc:
            problems.append(f"{rel}: değişiklikten sonra sözdizimi hatası, satır {exc.lineno}: {exc.msg}")
    return problems


def _apply_to(root: pathlib.Path, edits: list[Edit]) -> list[pathlib.Path]:
    texts, problems = _result_texts(edits, root)
    if problems:
        raise EvolveError("; ".join(problems))
    touched = []
    for rel, text in texts.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        touched.append(p)
    return touched


def code_fingerprint(root: pathlib.Path | None = None) -> str:
    """Changes whenever any file a proposal could touch changes: a proposal tested on other code must be re-tested."""
    import hashlib
    root = root or editable_root()
    h = hashlib.sha256()
    for d in (("plugins",) if frozen() else ("visionflow", "tests", "plugins")):
        if (root / d).is_dir():
            for f in sorted((root / d).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                h.update(f.relative_to(root).as_posix().encode())
                try:
                    h.update(f.read_bytes())
                except OSError:
                    pass
    return h.hexdigest()


def _sha(path: pathlib.Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check_plugins(folder, out_file) -> int:
    """`--check-plugins DIR OUT`: load the plugins of DIR in this (throw-away) process and write their load errors
    to OUT as JSON. The packaged program tries a proposed plugin this way, never inside the running program."""
    from .core import load_builtin
    from .core.registry import PLUGIN_ERRORS, load_plugins
    load_builtin()
    PLUGIN_ERRORS.clear()
    try:
        load_plugins(folder)
        errs = dict(PLUGIN_ERRORS)
    except Exception as exc:  # noqa: BLE001
        errs = {"(yükleme)": f"{type(exc).__name__}: {exc}"}
    pathlib.Path(out_file).write_text(json.dumps(errs, ensure_ascii=False), encoding="utf-8")
    return 0 if not errs else 1


def try_in_copy(edits: list[Edit]) -> tuple[bool, str]:
    """Apply the change to a temporary copy of the program and run its tests there. Nothing real is touched."""
    root = editable_root()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vf-kopya-"))
    try:
        if frozen():
            # no Python to run tests with: at least the new plugins must compile and load without errors
            plug = tmp / "plugins"
            plug.mkdir()
            if (root / "plugins").is_dir():
                for p in (root / "plugins").glob("*.py"):
                    shutil.copy2(p, plug / p.name)
            _apply_to(tmp, edits)
            out = tmp / "sonuc.json"
            cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "visionflow"]
            env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "VISIONFLOW_HOME": str(tmp / "ayar")}
            try:
                subprocess.run(cmd + ["--check-plugins", str(plug), str(out)], env=env, capture_output=True,
                               timeout=300, cwd=str(tmp if getattr(sys, "frozen", False) else SOURCE_ROOT))
                errs = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                errs = {"(deneme)": f"eklentiler ayrı süreçte denenemedi: {exc}"}
            return (not errs), ("Eklentiler yüklendi" if not errs else "\n".join(f"{k}: {v}" for k, v in errs.items()))
        for d in ("visionflow", "tests", "plugins", "examples"):
            if (root / d).is_dir():
                shutil.copytree(root / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in ("requirements.txt", "pyproject.toml", "setup.cfg", "pytest.ini", "conftest.py"):
            if (root / f).is_file():
                shutil.copy2(root / f, tmp / f)
        _apply_to(tmp, edits)
        return run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def propose(goal: str = "", rep: dict | None = None, cfg: AIConfig | None = None,
            on_step=lambda text: None, ask=None) -> Proposal:
    """Self-examination → Claude's change → test on a copy → if it fails, show Claude why and let it correct
    itself, up to MAX_ROUNDS times. ``ask(prompt, system, history) -> str`` is replaceable for tests."""
    cfg = cfg or AIConfig.load()
    if ask is None:
        from .ai import claude

        def ask(prompt, system, history):
            return claude.ask(prompt, system=system, history=history, cfg=cfg, max_tokens=16000, effort="high")
    if rep is None:
        on_step("Kendini inceliyor (hata günlüğü ve testler)…")
        rep = self_report()
    goal = goal.strip()
    if not goal and not rep["errors"] and not rep.get("plugin_errors") and rep.get("tests_ok", True):
        goal = ("Kaydedilmiş hata yok. Programı kullanıcılar için daha kullanışlı yapacak küçük ve güvenli bir "
                "iyileştirme öner (ör. daha anlaşılır bir hata mesajı veya eksik bir kontrol).")
    files = _files_for(rep, goal)
    allowed_txt = "plugins/" if frozen() else "visionflow/, tests/, plugins/"
    system = SYSTEM.format(allowed=allowed_txt, protected=", ".join(sorted(PROTECTED)))
    if frozen():
        system += ("\nThis is the packaged program: you can only add plugin files in plugins/. A plugin can fix a "
                   "built-in block by subclassing it with the same class name and @register (it replaces it).\n")
    prompt = (f"Hedef: {goal or 'Rapordaki hataları gider.'}\n\n{report_text(rep)}\n\nProgramın dosyaları:\n{_tree()}\n\n"
              + "\n\n".join(f"=== {rel} ===\n{body}" for rel, body in files.items()))
    history: list[dict] = []
    for rnd in range(1, MAX_ROUNDS + 1):
        on_step(f"Claude'a soruyor (tur {rnd}/{MAX_ROUNDS})…")
        answer = ask(prompt, system, history)
        history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        try:
            summary, edits = _parse(answer)
            problems = check_edits(edits)
        except EvolveError as exc:
            summary, edits, problems = "", [], [str(exc)]
        if not problems and not edits:
            return Proposal(summary or "Değişiklik gerekmiyor.", [], rnd, None, "", goal)
        if not problems:
            on_step(f"Değişikliği programın bir kopyasında deniyor (tur {rnd})…")
            fp = code_fingerprint()
            ok, out = try_in_copy(edits)
            if ok:
                return Proposal(summary, edits, rnd, True, out, goal, fingerprint=fp)
            problems = [f"Testler başarısız oldu:\n{out[-6000:]}"]
        if rnd == MAX_ROUNDS:
            return Proposal(summary or "Geçerli bir değişiklik bulunamadı.", edits, rnd, False, "\n".join(problems), goal)
        on_step("Kendi önerisini sorguluyor: hatayı Claude'a gösterip düzelttiriyor…")
        prompt = ("Önerin kullanılamadı:\n- " + "\n- ".join(problems) +
                  "\n\nSorunu bul ve düzeltilmiş tam öneriyi aynı JSON biçiminde tekrar ver.")
    raise AssertionError("unreachable")


# --- applying, history, undo --------------------------------------------------------------------------------------

def _changes_dir() -> pathlib.Path:
    return home() / "gelistirmeler"


def _folder(change_id: str) -> pathlib.Path:
    if not re.fullmatch(r"[\w-]{1,80}", change_id or ""):
        raise EvolveError(f"Geçersiz değişiklik numarası: {change_id!r}")
    return _changes_dir() / change_id


def needs_retest(prop: Proposal) -> bool:
    """The program changed since the proposal was tested (another change applied, an update, a hand edit)."""
    return not prop.fingerprint or prop.fingerprint != code_fingerprint()


def retest(prop: Proposal) -> Proposal:
    fp = code_fingerprint()
    problems = check_edits(prop.edits)
    if problems:
        prop.tests_ok, prop.test_output = False, "\n".join(problems)
    else:
        prop.tests_ok, prop.test_output = try_in_copy(prop.edits)
    prop.fingerprint = fp
    return prop


def apply(prop: Proposal) -> pathlib.Path:
    """Apply an approved and tested change. Originals are backed up first; returns the backup folder. If writing
    fails half-way, the files already written are put back and nothing is recorded."""
    if prop.tests_ok is not True:
        raise EvolveError("Sadece kopyada testleri geçen bir değişiklik uygulanabilir")
    problems = check_edits(prop.edits)
    if problems:
        raise EvolveError("Değişiklik artık uygulanamıyor (dosyalar değişmiş olabilir): " + "; ".join(problems))
    if needs_retest(prop):
        raise EvolveError("Program bu öneri test edildikten sonra değişti; önce yeniden test edilmeli")
    root = editable_root()
    folder = _folder(prop.id)
    if (folder / "bilgi.json").exists():
        raise EvolveError("Bu değişiklik zaten uygulandı")
    texts, _ = _result_texts(prop.edits, root)
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "yedek").mkdir(parents=True, exist_ok=True)
    backups = []
    try:
        for rel in texts:
            p = root / rel
            if p.exists():
                dst = folder / "yedek" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
            backups.append({"path": rel, "existed": p.exists()})
        written = []
        try:
            for rel, text in texts.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                written.append(rel)
                p.write_text(text, encoding="utf-8")
        except OSError as exc:
            for b in backups:                       # put back what was already written
                if b["path"] in written:
                    p = root / b["path"]
                    if b["existed"]:
                        shutil.copy2(folder / "yedek" / b["path"], p)
                    else:
                        p.unlink(missing_ok=True)
            raise EvolveError(f"Dosya yazılamadı, hiçbir şey değişmedi: {exc}") from None
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    for b in backups:
        b["sha"] = _sha(root / b["path"])           # what undo expects to find
    meta = {**prop.to_dict(), "applied": dt.datetime.now().isoformat(timespec="seconds"), "backups": backups,
            "undone": False}
    (folder / "bilgi.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    waiting = load_pending()
    if waiting and waiting.id == prop.id:
        discard_pending()
    return folder


def history() -> list[dict]:
    out = []
    d = _changes_dir()
    if d.is_dir():
        for f in sorted(d.glob("*/bilgi.json"), reverse=True):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
    return out


def undo(change_id: str) -> None:
    """Put the files of an applied change back. Refused when a file was changed again afterwards (a newer change
    or a hand edit), so undoing an old change never wipes out a newer one: undo the newer one first."""
    folder = _folder(change_id)
    try:
        meta = json.loads((folder / "bilgi.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EvolveError(f"Değişiklik bulunamadı: {change_id}") from None
    if meta.get("undone"):
        return
    root = editable_root()
    backups = [b for b in meta.get("backups", []) if isinstance(b, dict) and not allowed(str(b.get("path", "")))]
    for b in backups:
        p = root / b["path"]
        if "sha" in b and _sha(p) != b["sha"]:
            raise EvolveError(f"{b['path']} bu değişiklikten sonra yeniden değişti; önce daha yeni değişikliği "
                              "geri alın (veya dosyayı elle düzeltin)")
        if b.get("existed") and not (folder / "yedek" / b["path"]).is_file():
            raise EvolveError(f"{b['path']} dosyasının yedeği kayıp; geri alınamıyor")
    for b in backups:
        p = root / b["path"]
        if b.get("existed"):
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "yedek" / b["path"], p)
        else:
            p.unlink(missing_ok=True)
    meta["undone"] = dt.datetime.now().isoformat(timespec="seconds")
    (folder / "bilgi.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# --- waiting proposal and the daily self-review -----------------------------------------------------------------

def settings_file() -> pathlib.Path:
    return home() / "kendini_gelistir.json"


def settings() -> dict:
    try:
        return {"daily": False, "last": "", **json.loads(settings_file().read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"daily": False, "last": ""}


def save_settings(s: dict) -> None:
    settings_file().parent.mkdir(parents=True, exist_ok=True)
    settings_file().write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def pending_file() -> pathlib.Path:
    return _changes_dir() / "bekleyen.json"


def save_pending(prop: Proposal) -> None:
    pending_file().parent.mkdir(parents=True, exist_ok=True)
    pending_file().write_text(json.dumps(prop.to_dict(), ensure_ascii=False), encoding="utf-8")


def discard_pending() -> None:
    pending_file().unlink(missing_ok=True)


def load_pending() -> Proposal | None:
    try:
        return Proposal.from_dict(json.loads(pending_file().read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def daily_due(now: dt.datetime | None = None) -> bool:
    s = settings()
    if not s["daily"]:
        return False
    try:
        last = dt.datetime.fromisoformat(s["last"])
    except ValueError:
        return True
    return (now or dt.datetime.now()) - last >= dt.timedelta(hours=23)


def _key_text(e: dict) -> str:
    return "|".join(map(str, error_key(e.get("block"), e.get("error", ""))))


def daily_review(cfg: AIConfig | None = None, ask=None) -> Proposal | None:
    """Run in the background at start-up when 'every day' is on: examine itself and, only if something went wrong,
    prepare a tested proposal that waits for the user. Never applies anything."""
    s = settings()
    s["last"] = dt.datetime.now().isoformat(timespec="seconds")
    save_settings(s)
    rep = self_report()
    # only errors not asked about on an earlier day: an error that stays (fix not applied, or caused by the
    # setup) must not cost a Claude question every day
    asked = set(s.get("asked", []))
    rep["errors"] = [e for e in rep["errors"] if _key_text(e) not in asked]
    new_plugin = {k: v for k, v in rep["plugin_errors"].items() if f"plugin|{k}|{v}" not in asked}
    rep["plugin_errors"] = new_plugin
    tests_key = ""
    if rep.get("tests_ok") is False:
        failed = [ln for ln in rep.get("tests", "").splitlines() if ln.startswith(("FAILED", "ERROR"))]
        tests_key = "tests|" + re.sub(r"\d+", "#", "\n".join(failed) or rep.get("tests", "")[-300:])
        if tests_key in asked:
            tests_key = ""                                  # the same tests failed before: already asked
    if not rep["errors"] and not new_plugin and not tests_key:
        return None
    s["asked"] = (list(asked) + [_key_text(e) for e in rep["errors"]]
                  + [f"plugin|{k}|{v}" for k, v in new_plugin.items()] + ([tests_key] if tests_key else []))[-500:]
    save_settings(s)
    prop = propose("", rep, cfg, ask=ask)
    if prop.edits and prop.tests_ok:
        save_pending(prop)
        return prop
    return None


def main(argv=None) -> int:
    """`python -m visionflow.evolve` prints the self-examination report (no Claude call, nothing changed)."""
    argv = list(argv if argv is not None else sys.argv[1:])
    print(report_text(self_report(run_tests="--no-tests" not in argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
