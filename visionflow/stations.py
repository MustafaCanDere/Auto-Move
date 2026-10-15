"""İstasyon Yöneticisi: one PC runs several flows (stations / cameras) at once, each in its OWN process.

    python -m visionflow istasyonlar hat.json          runs every auto-start station until Ctrl+C / service stop

A station file (JSON) lists the stations::

    {"istasyonlar": [
        {"ad": "Hat 1", "akis": "hat1.json", "recete": "A ürünü", "otomatik_baslat": true,
         "yeniden_baslat": true, "en_fazla_yeniden": 5, "bekleme_sn": 1, "en_fazla_bekleme_sn": 60,
         "oncelik": "normal", "cekirdekler": [0, 1], "api_portu": 8791, "api_ag": false, "fps": 30, "kare": 0}
    ]}

Every station is a child process running the headless runner (``--headless akis.json``), so a camera driver that
crashes (or a flow that calls os._exit) only takes down its own station; the supervisor starts it again after
1, 2, 4 … s (``bekleme_sn`` doubling up to ``en_fazla_bekleme_sn``), at most ``en_fazla_yeniden`` times in a row
(0 = forever; a run that lasted ``STABLE_SECONDS`` starts the count again).

Status channel: each child writes a small status JSON file (running, FPS, frames, errors, last error, recipe) twice
a second and a small preview JPEG once a second into the run folder (<VisionFlow klasörü>/istasyonlar/<dosya>/).
Files rather than the REST API because they need no port, no key and no extra thread per station in the
supervisor, work the same on Windows and Linux, survive the child's crash (the last error stays readable) and cost
the child almost nothing. A station may still open its own REST API (``api_portu``) for MES / ERP.

Stopping is clean on both systems without signals that only POSIX has: the supervisor creates the station's stop
file (the child polls it, finishes its frame and closes recordings), sends SIGTERM where there is one, and kills the
process only after ``kill_timeout`` seconds. Every child is waited for (no zombies), and a child whose supervisor
disappeared (killed, crashed) notices it through ``--ust-pid`` and stops by itself (no orphans).
"""
from __future__ import annotations

import atexit
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Callable

STABLE_SECONDS = 60.0      # a station that ran this long without crashing gets its full number of restarts again
KILL_TIMEOUT = 10.0        # seconds a station gets to stop by itself before it is killed
LOG_MAX = 2_000_000        # a station log bigger than this is moved to <ad>.log.1 when the station starts
STATUS_EVERY = 0.5         # the child writes its status file this often
PREVIEW_EVERY = 1.0        # ... and its preview JPEG this often
PREVIEW_WIDTH = 320
PRIORITIES = ("dusuk", "normal", "yuksek")
PRIORITY_LABELS = {"dusuk": "Düşük", "normal": "Normal", "yuksek": "Yüksek"}
NICE = {"dusuk": 10, "normal": 0, "yuksek": -5}

# state code -> (Turkish text, lamp color)
STATES = {
    "durdu": ("Durdu", "#607d8b"),
    "basliyor": ("Başlıyor", "#1e88e5"),
    "calisiyor": ("Çalışıyor", "#2e7d32"),
    "uyari": ("Çalışıyor (blok hatası)", "#f9a825"),
    "bekliyor": ("Akış durdu, API'den 'baslat' bekleniyor", "#f9a825"),
    "yeniden": ("Çöktü, yeniden başlatılacak", "#ef6c00"),
    "duruyor": ("Duruyor", "#78909c"),
    "bitti": ("Bitti", "#607d8b"),
    "coktu": ("Çöktü (vazgeçildi)", "#c62828"),
    "hata": ("Başlatılamadı", "#c62828"),
}


# =============================================================================================== station file
@dataclass
class Station:
    ad: str
    akis: str
    recete: str = ""
    otomatik_baslat: bool = True
    yeniden_baslat: bool = True
    en_fazla_yeniden: int = 5
    bekleme_sn: float = 1.0
    en_fazla_bekleme_sn: float = 60.0
    oncelik: str = "normal"
    cekirdekler: list[int] = field(default_factory=list)
    api_portu: int = 0                # 0 = no REST API
    api_ag: bool = False
    fps: float = 30.0
    kare: int = 0                     # stop after this many frames (0 = never)

    def flow_path(self, base: pathlib.Path | None = None) -> pathlib.Path:
        p = pathlib.Path(self.akis).expanduser()
        return p if p.is_absolute() or base is None else base / p

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if k in ("ad", "akis") or v != _DEFAULTS.get(k)}


_DEFAULTS = {f.name: (f.default if f.default is not MISSING else f.default_factory())
             for f in fields(Station) if f.name not in ("ad", "akis")}


def _num(raw: dict, key: str, kind, lo, hi, where: str):
    v = raw.get(key, _DEFAULTS[key])
    if v is None:
        v = _DEFAULTS[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: '{key}' bir sayı olmalı")
    if kind is int and v != int(v):
        raise ValueError(f"{where}: '{key}' tam sayı olmalı")
    if not lo <= v <= hi:
        raise ValueError(f"{where}: '{key}' {lo} ile {hi} arasında olmalı (şu an {v})")
    return kind(v)


def _bool(raw: dict, key: str, where: str) -> bool:
    v = raw.get(key, _DEFAULTS[key])
    if not isinstance(v, bool):
        raise ValueError(f"{where}: '{key}' true ya da false olmalı")
    return v


def station_from_dict(raw, index: int = 0) -> Station:
    """One station of a station file, checked (ValueError with a Turkish message)."""
    where = f"{index + 1}. istasyon"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: bir JSON nesnesi olmalı ({{\"ad\": ..., \"akis\": ...}})")
    known = {f.name for f in fields(Station)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{where}: bilinmeyen alan(lar): {', '.join(unknown)}")
    name = raw.get("ad")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: 'ad' (istasyon adı) eksik")
    where = f"'{name.strip()}' istasyonu"
    flow = raw.get("akis")
    if not isinstance(flow, str) or not flow.strip():
        raise ValueError(f"{where}: 'akis' (akış dosyası) eksik")
    recipe = raw.get("recete", "")
    if recipe is None:
        recipe = ""
    if isinstance(recipe, bool) or not isinstance(recipe, (str, int)):
        raise ValueError(f"{where}: 'recete' bir ad ya da numara olmalı")
    prio = raw.get("oncelik", "normal")
    if prio not in PRIORITIES:
        raise ValueError(f"{where}: 'oncelik' şunlardan biri olmalı: {', '.join(PRIORITIES)}")
    cpus = raw.get("cekirdekler", [])
    if cpus is None:
        cpus = []
    n_cpu = os.cpu_count() or 1
    if not isinstance(cpus, list) or any(isinstance(c, bool) or not isinstance(c, int) for c in cpus):
        raise ValueError(f"{where}: 'cekirdekler' çekirdek numaralarının listesi olmalı, örn. [0, 1]")
    bad = [c for c in cpus if not 0 <= c < n_cpu]
    if bad:
        raise ValueError(f"{where}: bu bilgisayarda {n_cpu} çekirdek var (0-{n_cpu - 1}); geçersiz: {bad}")
    st = Station(
        ad=name.strip(), akis=flow.strip(), recete=str(recipe).strip(),
        otomatik_baslat=_bool(raw, "otomatik_baslat", where), yeniden_baslat=_bool(raw, "yeniden_baslat", where),
        en_fazla_yeniden=_num(raw, "en_fazla_yeniden", int, 0, 1_000_000, where),
        bekleme_sn=_num(raw, "bekleme_sn", float, 0.0, 3600.0, where),
        en_fazla_bekleme_sn=_num(raw, "en_fazla_bekleme_sn", float, 0.0, 86400.0, where),
        oncelik=prio, cekirdekler=sorted(set(cpus)),
        api_portu=_num(raw, "api_portu", int, 0, 65535, where), api_ag=_bool(raw, "api_ag", where),
        fps=_num(raw, "fps", float, 0.0, 1000.0, where), kare=_num(raw, "kare", int, 0, 10**12, where))
    if st.en_fazla_bekleme_sn < st.bekleme_sn:
        st.en_fazla_bekleme_sn = st.bekleme_sn
    return st


def check_stations(stations: list[Station]) -> None:
    """Names (also as file names) and API ports must not repeat."""
    from .service import safe_name
    seen, files, ports = set(), {}, {}
    for st in stations:
        key = st.ad.casefold()
        if key in seen:
            raise ValueError(f"'{st.ad}' adlı iki istasyon var: adlar farklı olmalı")
        seen.add(key)
        f = safe_name(st.ad).casefold()
        if f in files:
            raise ValueError(f"'{st.ad}' ile '{files[f]}' adları dosya adı olarak aynı ({f}): birini değiştirin")
        files[f] = st.ad
        if st.api_portu:
            if st.api_portu in ports:
                raise ValueError(f"'{st.ad}' ile '{ports[st.api_portu]}' aynı API portunu ({st.api_portu}) "
                                 "kullanıyor")
            ports[st.api_portu] = st.ad


def load_stations(path) -> list[Station]:
    """The stations of a station file. ValueError (Turkish) when it is missing or wrong."""
    path = pathlib.Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"İstasyon dosyası bulunamadı: {path}") from None
    except OSError as exc:
        raise ValueError(f"İstasyon dosyası okunamadı ({path}): {exc}") from None
    except ValueError as exc:
        raise ValueError(f"İstasyon dosyası geçerli JSON değil ({path}): {exc}") from None
    items = raw.get("istasyonlar") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"İstasyon dosyasında 'istasyonlar' listesi yok: {path}")
    stations = [station_from_dict(r, i) for i, r in enumerate(items)]
    check_stations(stations)
    return stations


def save_stations(path, stations: list[Station]) -> None:
    check_stations(stations)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"surum": 1, "istasyonlar": [s.to_json() for s in stations]}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def relative_flow(flow: str, station_file) -> str:
    """A flow path as written into the station file: relative when it is in the file's folder (or below)."""
    try:
        p = pathlib.Path(flow).resolve()
        return p.relative_to(pathlib.Path(station_file).resolve().parent).as_posix()
    except (ValueError, OSError):
        return str(flow)


# =============================================================================================== small helpers
def run_root() -> pathlib.Path:
    from .ai.config import home
    return home() / "istasyonlar"


def _write_atomic(path: pathlib.Path, data: bytes) -> bool:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, path)           # a reader never sees half a file
        return True
    except OSError:                     # Windows: the reader has it open right now; next time
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def pid_alive(pid: int) -> bool:
    """Is process ``pid`` still running? (psutil when installed; else os.kill(pid, 0) / the Win32 API)."""
    if not pid or pid < 0:
        return False
    try:
        import psutil
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))       # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259    # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)                 # POSIX: signal 0 only checks
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _python() -> str:
    """The Python to start stations with (a Windows service runs inside pythonservice.exe, not python.exe)."""
    exe = pathlib.Path(sys.executable)
    if exe.name.lower().startswith("pythonservice"):
        for cand in (exe.with_name("python.exe"), pathlib.Path(sys.exec_prefix) / "python.exe"):
            if cand.is_file():
                return str(cand)
    return sys.executable


def program_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [_python(), "-m", "visionflow"]


# =============================================================================================== child side
class StationLink:
    """Inside a station's process (headless --durum-dosyasi ...): writes the status file and the preview JPEG from
    the run's FlowState, and stops the flow when the stop file appears or the supervisor is gone. ``ctl`` is the
    api_server.HeadlessController whose hooks fill the FlowState (its ``graph`` while it runs, ``restarts``)."""

    def __init__(self, state, ctl, status_path=None, preview_path=None, stop_path=None, parent_pid=None,
                 on_stop: Callable[[], None] = lambda: None):
        self.state, self.ctl = state, ctl
        self.status_path = pathlib.Path(status_path) if status_path else None
        self.preview_path = pathlib.Path(preview_path) if preview_path else None
        self.stop_path = pathlib.Path(stop_path) if stop_path else None
        self.parent_pid = parent_pid
        self.on_stop = on_stop
        self.started = time.time()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._img_key = None
        self._preview_id = None
        self._preview_graph = None
        self._stop_sent = False

    def start(self) -> "StationLink":
        self.write_status()
        self._thread = threading.Thread(target=self._loop, name="VisionFlow istasyon", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._quit.set()
        if self._thread is not None:
            self._thread.join(3)
            self._thread = None
        self.write_status(final=True)

    def _parent_gone(self) -> bool:
        if not self.parent_pid:
            return False
        if os.name == "posix":
            return os.getppid() != self.parent_pid      # re-parented: the supervisor died
        return not pid_alive(self.parent_pid)

    def _loop(self) -> None:
        last_status = last_preview = 0.0
        while not self._quit.wait(0.1):
            if not self._stop_sent and ((self.stop_path is not None and self.stop_path.exists())
                                        or self._parent_gone()):
                self._stop_sent = True
                print("Durdurma isteği alındı: akış durduruluyor...", flush=True)
                self.on_stop()
            now = time.monotonic()
            if now - last_status >= STATUS_EVERY:
                last_status = now
                self.write_status()
            if self.preview_path is not None and now - last_preview >= PREVIEW_EVERY:
                last_preview = now
                try:
                    self.write_preview()
                except Exception as exc:  # noqa: BLE001  (a preview must never stop the station)
                    print(f"[UYARI] Önizleme yazılamadı: {exc}", flush=True)

    def status(self) -> dict:
        st = self.state.status()
        last = st["son_hatalar"][-1] if st["son_hatalar"] else None
        last_text = ""
        if last:
            who = last.get("baslik") or last.get("blok") or ""
            last_text = f"{who}: {last.get('hata', '')}" if who else str(last.get("hata", ""))
        return {"pid": os.getpid(), "zaman": time.time(), "baslangic": self.started,
                "calisiyor": st["calisiyor"], "fps": st["fps"], "kare": st["kare"], "recete": st["recete"],
                "hata_sayisi": st["hata_sayisi"], "hatali_blok": len(st["hatalar"]), "son_hata": last_text,
                "yeniden": int(getattr(self.ctl, "restarts", 0) or 0)}

    def write_status(self, final: bool = False) -> None:
        if self.status_path is None:
            return
        d = self.status()
        if final:
            d["calisiyor"], d["fps"], d["bitti"] = False, 0.0, True
        _write_atomic(self.status_path, json.dumps(d, ensure_ascii=False).encode("utf-8"))

    def write_preview(self) -> None:
        graph = getattr(self.ctl, "graph", None)
        if graph is None:
            return
        if graph is not self._preview_graph:
            from .production import default_image
            op = getattr(graph, "operator", None) or {}
            self._preview_id = op.get("image") or default_image(graph)
            self._preview_graph = graph
        img = self.state.image(self._preview_id) if self._preview_id else None
        if img is None:
            with self.state.lock:      # no preview block: any picture of the flow
                img = next(iter(self.state.images.values()), None)
        if img is None or (id(img), self.state.frames) == self._img_key:
            return
        self._img_key = (id(img), self.state.frames)
        from .api_server import to_jpeg
        _write_atomic(self.preview_path, to_jpeg(img, 70, PREVIEW_WIDTH))


# =============================================================================================== supervisor side
@dataclass
class _Run:
    """What the supervisor knows about one station."""
    st: Station
    state: str = "durdu"
    proc: subprocess.Popen | None = None
    want: bool = False                 # should be running
    stopping: bool = False
    started: float = 0.0               # monotonic, of the current process
    kill_at: float = 0.0
    next_start: float = 0.0
    delay: float = 0.0
    streak: int = 0                    # restarts in a row (since the last stable run / manual start)
    restarts: int = 0                  # restarts after crashes since the supervisor started
    exit_code: int | None = None
    last_error: str = ""
    child: dict = field(default_factory=dict)
    status_mtime: float = 0.0
    last_pid: int = 0                  # of the newest process (its status stays shown after it exited)
    child_pid: int = 0                 # PID of the child process reported in status file


class Supervisor:
    """Starts, watches and stops the stations of a station file. Not a thread: the owner calls ``tick()`` a few
    times a second (the CLI loop, the station window's timer, the Windows service), so nothing here needs locks."""

    def __init__(self, stations: list[Station], source=None, run_dir=None, kill_timeout: float = KILL_TIMEOUT,
                 log: Callable[[str], None] | None = None):
        check_stations(stations)
        self.source = pathlib.Path(source).resolve() if source else None
        self.base = self.source.parent if self.source else pathlib.Path.cwd()
        from .service import safe_name
        self.run_dir = pathlib.Path(run_dir) if run_dir else \
            run_root() / safe_name(self.source.stem if self.source else "istasyonlar")
        self.kill_timeout = float(kill_timeout)
        self.log = log or (lambda msg: None)
        self.runs: dict[str, _Run] = {st.ad: _Run(st) for st in stations}
        _LIVE.add(self)

    # --- files ----------------------------------------------------------------------------------------------
    def _file(self, name: str, ext: str) -> pathlib.Path:
        from .service import safe_name
        return self.run_dir / f"{safe_name(name)}.{ext}"

    def log_path(self, name: str) -> pathlib.Path:
        return self._file(name, "log")

    def preview_path(self, name: str) -> pathlib.Path:
        return self._file(name, "jpg")

    def status_path(self, name: str) -> pathlib.Path:
        return self._file(name, "json")

    def stop_path(self, name: str) -> pathlib.Path:
        return self._file(name, "dur")

    def log_tail(self, name: str, lines: int = 200, max_bytes: int = 64_000) -> str:
        try:
            with open(self.log_path(name), "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                data = f.read()
        except OSError:
            return ""
        text = data.decode("utf-8", "replace").replace("\r\n", "\n")
        return "\n".join(text.split("\n")[-lines:])

    def _note(self, run: _Run, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [İstasyon yöneticisi] {msg}\n"
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_path(run.st.ad), "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass
        self.log(f"{run.st.ad}: {msg}")

    # --- station list ---------------------------------------------------------------------------------------
    @property
    def stations(self) -> list[Station]:
        return [r.st for r in self.runs.values()]

    def set_stations(self, stations: list[Station]) -> None:
        """The edited list: removed stations are stopped; changed ones use their new settings from the next start."""
        check_stations(stations)
        new = {st.ad: st for st in stations}
        for name in [n for n in self.runs if n not in new]:
            run = self.runs[name]
            if run.proc is not None:
                self.stop(name)
                self._wait_exit([run], self.kill_timeout)
            del self.runs[name]
        runs = {}
        for st in stations:
            run = self.runs.get(st.ad) or _Run(st)
            run.st = st
            runs[st.ad] = run
        self.runs = runs

    def _get(self, name: str) -> _Run:
        try:
            return self.runs[name]
        except KeyError:
            raise ValueError(f"'{name}' adlı istasyon yok") from None

    # --- control --------------------------------------------------------------------------------------------
    def start(self, name: str) -> None:
        run = self._get(name)
        run.want = True
        run.streak, run.delay = 0, run.st.bekleme_sn
        if run.proc is None:
            self._spawn(run)
        elif run.stopping:            # stopping right now: start again once it has exited
            run.state = "duruyor"

    def stop(self, name: str) -> None:
        run = self._get(name)
        run.want = False
        if run.proc is None:
            if run.state in ("yeniden", "basliyor"):
                run.state = "durdu"
            return
        if run.stopping:
            return
        run.stopping = True
        run.state = "duruyor"
        run.kill_at = time.monotonic() + self.kill_timeout
        try:
            self.stop_path(run.st.ad).touch()       # the child polls it: works on every system
        except OSError:
            pass
        if os.name == "posix":
            try:
                run.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
        self._note(run, "durduruluyor")

    def restart(self, name: str) -> None:
        run = self._get(name)
        if run.proc is not None:
            self.stop(name)
        run.want = True               # started again by tick() once the old process has exited
        run.streak, run.delay = 0, run.st.bekleme_sn
        if run.proc is None:
            self._spawn(run)

    def start_all(self, only_auto: bool = False) -> None:
        for name, run in self.runs.items():
            if not only_auto or run.st.otomatik_baslat:
                if run.proc is None or run.stopping:
                    self.start(name)

    def stop_all(self) -> None:
        for name in self.runs:
            self.stop(name)

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop every station and wait until every process has exited (killed after the timeout)."""
        self.stop_all()
        self._wait_exit(list(self.runs.values()), self.kill_timeout if timeout is None else timeout)
        self.release()
        _LIVE.discard(self)

    def _wait_exit(self, runs: list[_Run], timeout: float) -> None:
        end = time.monotonic() + timeout
        for run in runs:
            if run.proc is not None and not run.stopping:
                self.stop(run.st.ad)
        while any(r.proc is not None for r in runs):
            if time.monotonic() >= end + 15:        # cannot even be killed (hung in a driver): give up waiting
                for r in runs:
                    if r.proc is not None:
                        self._note(r, f"işlem {r.proc.pid} kapatılamadı")
                        r.proc, r.state, r.stopping, r.want = None, "hata", False, False
                break
            if time.monotonic() >= end:
                for r in runs:
                    if r.proc is not None:
                        r.kill_at = 0.0             # tick() kills it now
            self.tick()
            if any(r.proc is not None for r in runs):
                time.sleep(0.05)

    # --- processes ------------------------------------------------------------------------------------------
    def command(self, st: Station) -> list[str]:
        name = st.ad
        cmd = program_command() + [
            "--headless", str(st.flow_path(self.base).resolve()), "--fps", f"{st.fps:g}", "--stats", "30",
            "--durum-dosyasi", str(self.status_path(name)), "--onizleme-dosyasi", str(self.preview_path(name)),
            "--durdur-dosyasi", str(self.stop_path(name)), "--ust-pid", str(os.getpid())]
        if st.recete:
            cmd += ["--recete", st.recete]
        if st.kare:
            cmd += ["--frames", str(st.kare)]
        if st.api_portu:
            # the child keeps its API up: a crash inside restarts the flow there (the API stays reachable)
            cmd += ["--api", str(st.api_portu), "--yeniden-baslat", "--en-fazla-yeniden", str(st.en_fazla_yeniden)]
            if st.api_ag:
                cmd.append("--api-ag")
        return cmd

    def _spawn(self, run: _Run) -> None:
        st = run.st
        run.exit_code, run.child, run.child_pid, run.status_mtime = None, {}, 0, 0.0
        flow = st.flow_path(self.base)
        if not flow.is_file():
            run.state, run.want = "hata", False
            run.last_error = f"Akış dosyası bulunamadı: {flow}"
            self._note(run, run.last_error)
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            run.state, run.want = "hata", False
            run.last_error = f"Çalışma klasörü oluşturulamadı ({self.run_dir}): {exc}"
            return
        for p in (self.stop_path(st.ad), self.status_path(st.ad)):
            try:
                p.unlink()
            except OSError:
                pass
        log = self.log_path(st.ad)
        try:
            if log.stat().st_size > LOG_MAX:
                os.replace(log, log.with_name(log.name + ".1"))
        except OSError:
            pass
        from .service import working_dir
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            flags |= {"dusuk": subprocess.BELOW_NORMAL_PRIORITY_CLASS,
                      "yuksek": subprocess.ABOVE_NORMAL_PRIORITY_CLASS}.get(st.oncelik, 0)
        self._note(run, f"başlatılıyor: {st.flow_path(self.base)}" + (f" (reçete: {st.recete})" if st.recete else ""))
        try:
            with open(log, "ab") as out:
                run.proc = subprocess.Popen(self.command(st), stdin=subprocess.DEVNULL, stdout=out,
                                            stderr=subprocess.STDOUT, cwd=working_dir(), env=env, creationflags=flags)
        except OSError as exc:
            run.proc, run.state, run.want = None, "hata", False
            run.last_error = f"İşlem başlatılamadı: {exc}"
            self._note(run, run.last_error)
            return
        run.started = time.monotonic()
        run.last_pid = run.proc.pid
        run.stopping = False
        run.state = "basliyor"
        for warn in apply_limits(run.proc.pid, st):
            self._note(run, f"Uyarı: {warn}")

    def _exited(self, run: _Run, code: int) -> None:
        now = time.monotonic()
        ran = now - run.started
        run.proc, run.exit_code = None, code
        self._read_status(run)
        try:
            self.stop_path(run.st.ad).unlink()
        except OSError:
            pass
        was_stopping, run.stopping = run.stopping, False
        if was_stopping or not run.want:
            run.state = "durdu"
            self._note(run, f"durdu (çıkış kodu {code})")
            if run.want:                      # restart(): start the new process now
                self._spawn(run)
            return
        if code == 0:
            run.state, run.want = "bitti", False
            self._note(run, "akış bitti")
            return
        run.last_error = self._crash_reason(run, code)
        if code == 2:                         # wrong arguments / recipe not found: restarting does not help
            run.state, run.want = "hata", False
            self._note(run, f"başlatılamadı: {run.last_error}")
            return
        self._note(run, f"çöktü (çıkış kodu {code}): {run.last_error}")
        if ran >= STABLE_SECONDS:
            run.streak, run.delay = 0, run.st.bekleme_sn
        if not run.st.yeniden_baslat:
            run.state, run.want = "coktu", False
            return
        if run.st.en_fazla_yeniden and run.streak >= run.st.en_fazla_yeniden:
            run.state, run.want = "coktu", False
            self._note(run, f"en fazla yeniden başlatma sayısına ({run.st.en_fazla_yeniden}) ulaşıldı, vazgeçildi")
            return
        run.state = "yeniden"
        run.next_start = now + run.delay
        self._note(run, f"{run.delay:g} sn sonra yeniden başlatılacak")

    def _crash_reason(self, run: _Run, code: int) -> str:
        tail = [ln.strip() for ln in self.log_tail(run.st.ad, 40).split("\n") if ln.strip()
                and "[İstasyon yöneticisi]" not in ln]
        for ln in reversed(tail):            # a Python traceback ends with "SomeError: message"
            head = ln.split(":", 1)[0]
            if head.endswith(("Error", "Exception", "Interrupt")) and " " not in head:
                return ln
        msg = run.child.get("son_hata") or ""
        if not msg and code < 0:              # POSIX: killed by a signal (a driver's segmentation fault ...)
            try:
                return f"işlem {signal.Signals(-code).name} sinyaliyle kapandı"
            except ValueError:
                pass
        return msg or f"işlem beklenmedik şekilde kapandı (çıkış kodu {code})"

    def _read_status(self, run: _Run) -> None:
        p = self.status_path(run.st.ad)
        try:
            mtime = p.stat().st_mtime
            if mtime == run.status_mtime:
                return
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(d, dict):
            run.child, run.status_mtime = d, mtime
            cpid = d.get("pid")
            if cpid:
                try:
                    run.child_pid = int(cpid)
                except (ValueError, TypeError):
                    pass

    def tick(self) -> None:
        now = time.monotonic()
        for run in list(self.runs.values()):
            p = run.proc
            if p is not None:
                code = p.poll()               # also reaps it: no zombie
                if code is not None:
                    self._exited(run, code)
                    continue
                if run.stopping and now >= run.kill_at:
                    self._note(run, f"{self.kill_timeout:g} sn içinde durmadı: zorla kapatılıyor")
                    try:
                        p.kill()
                    except OSError:
                        pass
                    if run.child_pid and run.child_pid != p.pid:
                        try:
                            if pid_alive(run.child_pid):
                                os.kill(run.child_pid, signal.SIGTERM)
                        except OSError:
                            pass
                    try:
                        p.wait(5)
                    except subprocess.TimeoutExpired:
                        pass
                    code = p.poll()
                    if code is not None:
                        self._exited(run, code)
                    continue
                self._read_status(run)
                cpid = run.child.get("pid")
                match = (cpid == p.pid) or (run.child_pid and cpid == run.child_pid)
                if not run.stopping and match:
                    if run.child.get("calisiyor"):
                        run.state = "uyari" if run.child.get("hatali_blok") else "calisiyor"
                    elif run.st.api_portu and run.state in ("calisiyor", "uyari"):
                        run.state = "bekliyor"        # the flow ended / crashed; the API process waits
                    if run.child.get("son_hata"):
                        run.last_error = run.child["son_hata"]
            elif run.state == "yeniden" and run.want and now >= run.next_start:
                run.streak += 1
                run.restarts += 1
                run.delay = min(max(run.delay, run.st.bekleme_sn) * 2 or 0.0, run.st.en_fazla_bekleme_sn)
                self._spawn(run)

    # --- status ---------------------------------------------------------------------------------------------
    def status(self) -> list[dict]:
        out = []
        now = time.monotonic()
        for run in self.runs.values():
            text, color = STATES.get(run.state, (run.state, "#607d8b"))
            alive = run.proc is not None
            cpid = run.child.get("pid")
            match = (cpid == run.last_pid) or (run.child_pid and cpid == run.child_pid) if cpid else False
            ch = run.child if match else {}
            d = {"ad": run.st.ad, "akis": run.st.akis, "durum": run.state, "durum_metni": text, "renk": color,
                 "pid": run.proc.pid if alive else None, "fps": ch.get("fps", 0.0) if alive and ch.get("calisiyor") else 0.0,
                 "kare": ch.get("kare", 0), "hata_sayisi": ch.get("hata_sayisi", 0), "recete": ch.get("recete", ""),
                 "son_hata": run.last_error, "sure_sn": round(now - run.started, 1) if alive else 0.0,
                 "yeniden": run.restarts + int(ch.get("yeniden", 0) or 0), "cikis_kodu": run.exit_code,
                 "bekleme_sn": round(max(0.0, run.next_start - now), 1) if run.state == "yeniden" else 0.0,
                 "api_portu": run.st.api_portu}
            out.append(d)
        return out

    def claim(self) -> int:
        """Mark this station file as run by this process. Returns the pid of another live supervisor that runs
        it already (then nothing is marked), else 0."""
        lock = self.run_dir / "yonetici.pid"
        try:
            other = int(lock.read_text().strip())
        except (OSError, ValueError):
            other = 0
        if other and other != os.getpid() and pid_alive(other):
            return other
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            lock.write_text(str(os.getpid()))
        except OSError:
            pass
        return 0

    def release(self) -> None:
        lock = self.run_dir / "yonetici.pid"
        try:
            if lock.read_text().strip() == str(os.getpid()):
                lock.unlink()
        except (OSError, ValueError):
            pass

    def pids(self) -> list[int]:
        return [r.proc.pid for r in self.runs.values() if r.proc is not None]

    @property
    def busy(self) -> bool:
        """Any station process alive or about to be (re)started?"""
        return any(r.proc is not None or (r.want and r.state == "yeniden") for r in self.runs.values())

    def run_forever(self, should_stop: Callable[[], bool], on_tick: Callable[[], None] | None = None,
                    interval: float = 0.25, autostart: bool = True) -> None:
        """Run the auto-start stations until ``should_stop()``; then stop them all cleanly."""
        if autostart:
            self.start_all(only_auto=True)
        try:
            while not should_stop():
                self.tick()
                if on_tick is not None:
                    on_tick()
                time.sleep(interval)
        finally:
            self.shutdown()


def apply_limits(pid: int, st: Station) -> list[str]:
    """CPU priority and cores of a started station. Returns warnings (never raises)."""
    warns = []
    try:
        import psutil
    except ImportError:
        psutil = None
    if st.cekirdekler:
        try:
            if psutil is not None:
                psutil.Process(pid).cpu_affinity(list(st.cekirdekler))
            elif hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(pid, set(st.cekirdekler))
            else:
                warns.append("çekirdek seçimi için 'psutil' paketi gerekli (pip install psutil)")
        except Exception as exc:  # noqa: BLE001
            warns.append(f"çekirdekler ayarlanamadı: {exc}")
    if st.oncelik != "normal" and sys.platform != "win32":   # Windows: set by the creation flags
        try:
            if psutil is not None:
                psutil.Process(pid).nice(NICE[st.oncelik])
            else:
                os.setpriority(os.PRIO_PROCESS, pid, NICE[st.oncelik])
        except Exception as exc:  # noqa: BLE001
            warns.append(f"öncelik ayarlanamadı ({exc}); yüksek öncelik yönetici (root) yetkisi ister")
    return warns


_LIVE: "weakref.WeakSet[Supervisor]" = weakref.WeakSet()


@atexit.register
def _kill_leftovers() -> None:
    """The program ends with stations still running (an exception, a window closed without asking): stop them."""
    for sup in list(_LIVE):
        try:
            sup.shutdown(timeout=3.0)
        except Exception:  # noqa: BLE001
            pass


# =============================================================================================== command line
def run_supervisor(path, should_stop: Callable[[], bool], out: Callable[[str], None] = print,
                   every: float = 0.0, duration: float = 0.0) -> int:
    """Run the auto-start stations of a station file until ``should_stop()`` (or ``duration`` seconds)."""
    try:
        stations = load_stations(path)
    except ValueError as exc:
        out(str(exc))
        return 2
    if not stations:
        out("İstasyon dosyasında istasyon yok.")
        return 2
    sup = Supervisor(stations, source=path, log=lambda m: out(f"{time.strftime('%H:%M:%S')} {m}"))
    other = sup.claim()
    if other:
        _LIVE.discard(sup)
        out(f"Bu istasyon dosyası zaten çalışıyor (işlem {other}); iki yönetici aynı istasyonları çalıştıramaz.")
        return 3
    out(f"İstasyon yöneticisi: {len(stations)} istasyon ({sum(s.otomatik_baslat for s in stations)} otomatik "
        f"başlıyor); kayıtlar: {sup.run_dir}")
    end = time.monotonic() + duration if duration else None
    last = {"t": time.monotonic()}

    def show():
        if every and time.monotonic() - last["t"] >= every:
            last["t"] = time.monotonic()
            out(status_table(sup.status()))

    sup_stop = sup.run_dir / "dur"
    try:
        sup_stop.unlink()
    except OSError:
        pass
    try:
        sup.run_forever(lambda: should_stop() or sup_stop.exists() or (end is not None and time.monotonic() >= end), show)
    finally:
        sup.release()
        try:
            sup_stop.unlink()
        except OSError:
            pass
    out(status_table(sup.status()))
    out("İstasyon yöneticisi durdu.")
    return 0


def status_table(rows: list[dict]) -> str:
    head = f"{'İstasyon':<20} {'Durum':<28} {'FPS':>6} {'Kare':>8} {'Hata':>5} {'Yeniden':>7}  Son hata"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['ad'][:20]:<20} {r['durum_metni'][:28]:<28} {r['fps']:>6.1f} {r['kare']:>8} "
                     f"{r['hata_sayisi']:>5} {r['yeniden']:>7}  {str(r['son_hata'])[:60]}")
    return "\n".join(lines)


EXAMPLE = {"surum": 1, "istasyonlar": [
    {"ad": "Hat 1", "akis": "hat1.json", "otomatik_baslat": True, "yeniden_baslat": True, "en_fazla_yeniden": 5}]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m visionflow istasyonlar",
                                 description="Bir istasyon dosyasındaki akışları (kameraları) aynı anda, her birini "
                                             "ayrı bir işlemde çalıştırır; çökeni yeniden başlatır.")
    ap.add_argument("dosya", help="istasyon dosyası (.json)")
    ap.add_argument("--denetle", action="store_true", help="sadece dosyayı denetle, çalıştırma")
    ap.add_argument("--ornek", action="store_true", help="dosya yoksa örnek bir istasyon dosyası oluştur")
    ap.add_argument("--durum-araligi", type=float, default=30.0, metavar="SN",
                    help="durum tablosu kaç saniyede bir yazılsın (0 = yazma; varsayılan 30)")
    ap.add_argument("--sure", type=float, default=0.0, metavar="SN", help="bu kadar saniye sonra dur (0 = sınırsız)")
    args = ap.parse_args(argv)
    path = pathlib.Path(args.dosya)
    if args.ornek:
        if path.exists():
            print(f"Dosya zaten var, dokunulmadı: {path}")
            return 1
        path.write_text(json.dumps(EXAMPLE, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Örnek istasyon dosyası yazıldı: {path}  ('akis' alanını kendi akışınızla değiştirin)")
        return 0
    if args.denetle:
        try:
            stations = load_stations(path)
        except ValueError as exc:
            print(exc)
            return 1
        missing = [s for s in stations if not s.flow_path(path.resolve().parent).is_file()]
        for s in stations:
            print(f"  {s.ad}: {s.akis}" + ("   [AKIŞ DOSYASI YOK]" if s in missing else ""))
        print(f"{len(stations)} istasyon" + (f", {len(missing)} akış dosyası bulunamadı." if missing else ", sorun yok."))
        return 1 if missing else 0

    stop = {"flag": False}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), lambda *_: stop.update(flag=True))
            except ValueError:
                pass
    return run_supervisor(path, lambda: stop["flag"], every=max(0.0, args.durum_araligi), duration=args.sure)


if __name__ == "__main__":
    sys.exit(main())
