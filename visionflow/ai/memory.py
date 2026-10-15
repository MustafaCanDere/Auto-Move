"""Your own AI's memory: a local knowledge base (SQLite full-text search) plus conversation history.

Everything stays in one file on this computer. Documents are split into passages; the passages that match a
question best are handed to Claude together with the question, so answers use your own information.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import sqlite3
import threading

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm", ".xml", ".py", ".yaml", ".yml", ".ini"}
DOC_EXT = TEXT_EXT | {".pdf", ".docx", ".eml"}
CHUNK = 900         # characters per passage
OVERLAP = 150
_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {"ve", "ile", "bir", "bu", "şu", "da", "de", "mi", "mı", "mu", "mü", "ne", "için", "gibi", "çok", "daha",
         "the", "and", "for", "with", "that", "this", "what", "nasıl", "neden", "nedir", "olan", "olarak", "ama"}


def read_document(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF için: pip install pypdf") from None
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ValueError("Word için: pip install python-docx") from None
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if ext == ".eml":
        from email import policy
        from email.parser import BytesParser
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body = msg.get_body(preferencelist=("plain", "html"))
        return f"Kimden: {msg['from']}\nKonu: {msg['subject']}\n\n{body.get_content() if body else ''}"
    raw = path.read_bytes()
    # Notepad's "Unicode" is UTF-16 with a byte-order mark; "UTF-8 with BOM" would leave a stray \ufeff
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", "replace")
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    for enc in ("utf-8", "cp1254", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def split_passages(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= CHUNK:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK)
        if end < len(text):          # end on a sentence or line break when possible
            cut = max(text.rfind("\n", start + CHUNK // 2, end), text.rfind(". ", start + CHUNK // 2, end))
            if cut > 0:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [p for p in out if p]


def _fts_query(question: str) -> str:
    """Words → prefix search, so Turkish suffixes still match ('faturası' finds 'fatura')."""
    terms = []
    for w in _WORD.findall(question.replace("İ", "i").lower()):
        if len(w) < 3 or w in _STOP or w.isdigit() and len(w) < 2:
            continue
        stem = w if len(w) <= 5 else w[: max(5, len(w) - 3)]
        terms.append(f'"{stem}"*')
    return " OR ".join(dict.fromkeys(terms))


class Memory:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, name TEXT UNIQUE, added TEXT, passages INT);
            CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(text, source UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2');
            CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY, session TEXT, role TEXT, text TEXT, time TEXT);
        """)
        self.con.commit()

    def close(self):
        self.con.close()

    # knowledge -----------------------------------------------------------------------------------
    def add_text(self, text: str, source: str = "not", replace: bool = False) -> int:
        """Store text under a source name. Notes with the same source accumulate unless replace=True."""
        parts = split_passages(text)
        if not parts:
            if replace:                       # the document is now empty (or a scan without text): forget the old
                with self._lock:
                    self.con.execute("DELETE FROM passages WHERE source = ?", (source,))
                    self.con.execute("DELETE FROM sources WHERE name = ?", (source,))
                    self.con.commit()
            return 0
        with self._lock:
            if replace:
                self.con.execute("DELETE FROM passages WHERE source = ?", (source,))
            self.con.executemany("INSERT INTO passages (text, source) VALUES (?, ?)", [(p, source) for p in parts])
            n = self.con.execute("SELECT count(*) FROM passages WHERE source = ?", (source,)).fetchone()[0]
            self.con.execute("INSERT INTO sources (name, added, passages) VALUES (?, ?, ?) ON CONFLICT(name) "
                             "DO UPDATE SET added = excluded.added, passages = excluded.passages",
                             (source, dt.datetime.now().isoformat(timespec="seconds"), n))
            self.con.commit()
        return len(parts)

    def add_file(self, path: str | pathlib.Path, source: str | None = None) -> int:
        """Adding the same file again replaces it. ``source`` is the name it is listed under (default: file name)."""
        p = pathlib.Path(path)
        return self.add_text(read_document(p), source=source or p.name, replace=True)

    def add_folder(self, folder: str | pathlib.Path, recursive: bool = True) -> dict[str, int | str]:
        root = pathlib.Path(folder)
        files = root.rglob("*") if recursive else root.iterdir()
        result: dict[str, int | str] = {}
        for f in sorted(files):
            if f.is_file() and f.suffix.lower() in DOC_EXT:
                # 2023/fatura.txt and 2024/fatura.txt are different documents: name them by their sub-path
                name = f.relative_to(root).as_posix()
                try:
                    result[name] = self.add_file(f, source=name)
                except Exception as exc:  # noqa: BLE001
                    result[name] = f"okunamadı: {exc}"
        return result

    def remove(self, source: str) -> None:
        with self._lock:
            self.con.execute("DELETE FROM passages WHERE source = ?", (source,))
            self.con.execute("DELETE FROM sources WHERE name = ?", (source,))
            self.con.commit()

    def sources(self) -> list[tuple[str, str, int]]:
        return self.con.execute("SELECT name, added, passages FROM sources ORDER BY added DESC").fetchall()

    def search(self, question: str, k: int = 4) -> list[tuple[str, str]]:
        """Best matching (source, passage) pairs, best first."""
        q = _fts_query(question)
        if not q:
            return []
        with self._lock:
            rows = self.con.execute("SELECT source, text FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) "
                                    "LIMIT ?", (q, k)).fetchall()
        return [(s, t) for s, t in rows]

    def context_for(self, question: str, k: int = 4) -> str:
        hits = self.search(question, k)
        if not hits:
            return ""
        body = "\n\n".join(f"[{s}]\n{t}" for s, t in hits)
        return f"Hafızadan ilgili bilgiler (kaynak adlarıyla):\n{body}"

    # conversation --------------------------------------------------------------------------------
    def log(self, session: str, role: str, text: str) -> None:
        with self._lock:
            self.con.execute("INSERT INTO chat (session, role, text, time) VALUES (?, ?, ?, ?)",
                             (session, role, text, dt.datetime.now().isoformat(timespec="seconds")))
            self.con.commit()

    def history(self, session: str, turns: int = 6) -> list[dict]:
        rows = self.con.execute("SELECT role, text FROM chat WHERE session = ? ORDER BY id DESC LIMIT ?",
                                (session, turns * 2)).fetchall()[::-1]
        while rows and rows[0][0] != "user":     # the API wants the history to start with the user
            rows.pop(0)
        return [{"role": r, "content": t} for r, t in rows]

    def clear_history(self, session: str) -> None:
        with self._lock:
            self.con.execute("DELETE FROM chat WHERE session = ?", (session,))
            self.con.commit()


_shared: dict[str, Memory] = {}


def shared(path: str | pathlib.Path) -> Memory:
    """One connection per memory file for the whole program (blocks and the AI centre window share it)."""
    key = str(pathlib.Path(path).resolve())
    if key not in _shared:
        _shared[key] = Memory(path)
    return _shared[key]
