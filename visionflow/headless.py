"""Run a saved flow without the editor: ``python -m visionflow.headless akis.json``.

Meant for deploying a finished flow on a factory PC, as a Windows service or in Docker.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys
import time
import traceback
from typing import Callable

from .core import Graph, load_builtin, load_plugins
from .core.graph import flatten, inner_error, node_at
from .evolve import error_log

LOG_NAME = "visionflow.headless"
BACKOFF_START = 1.0         # seconds before the first restart after a crash
BACKOFF_MAX = 60.0          # never wait longer than this between restarts
STABLE_SECONDS = 60.0       # a run that lasted this long resets the backoff

_sleep = time.sleep          # patched in tests
_clock = time.monotonic


def log_dir() -> pathlib.Path:
    from .ai.config import home
    return home() / "kayitlar"


def get_logger(path: pathlib.Path | None = None) -> logging.Logger:
    """Logger that writes to <home>/kayitlar/headless.log, rotated at 1 MB (5 old files kept)."""
    from logging.handlers import RotatingFileHandler
    log = logging.getLogger(LOG_NAME)
    log.setLevel(logging.INFO)
    path = pathlib.Path(path) if path else log_dir() / "headless.log"
    target = str(path.resolve())
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == target for h in log.handlers):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            h = RotatingFileHandler(target, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(h)
        except OSError as exc:
            print(f"Kayıt dosyası açılamadı ({path}): {exc}", file=sys.stderr)
    return log


def run_flow(flow: str, frames: int = 0, fps: float = 30, stats: float = 5.0, plugins: str = "plugins",
             should_stop: Callable[[], bool] = lambda: False, check_only: bool = False,
             recete: str | None = None, api=None) -> int:
    """Load and run a flow until it ends or ``should_stop`` says so. Returns an exit code. ``api`` (remote control,
    see api_server.HeadlessController) gets attach(graph) / publish(graph, results) every frame / detach()."""
    load_builtin()
    if plugins == "plugins" and not pathlib.Path(plugins).is_dir():
        # a service starts in another folder: use the plugins next to the program
        base = pathlib.Path(sys.executable).parent if getattr(sys, "frozen", False) \
            else pathlib.Path(__file__).resolve().parents[1]
        plugins = str(base / "plugins")
    load_plugins(plugins)
    graph = Graph.load(flow)
    if recete is not None:
        from .core import recipes
        try:
            r = recipes.find(graph, recete)
        except ValueError as exc:
            print(f"Reçete seçilemedi: {exc}", file=sys.stderr)
            return 2
        for e in recipes.apply(graph, r):
            print(f"[UYARI] Reçete '{r['name']}': {e}", file=sys.stderr)
        # 'Reçete Seç' blocks start with it too instead of switching back to their own setting
        _set_selectors(graph, r["name"])
        graph.active_recipe = r["name"]          # parts archived by 'Parçayı Arşivle' carry it
        if api is not None and hasattr(api, "recipe_applied"):
            api.recipe_applied(r["name"])        # its name, also when it was chosen by number
        print(f"Reçete: {r['name']}")
    from .check import check
    issues = check(graph)
    for i in issues:
        print(i, file=sys.stderr if i.level == "hata" else sys.stdout)
    if check_only:
        errors = sum(i.level == "hata" for i in issues)
        print("Akış denetimi: sorun yok." if not issues else f"Akış denetimi: {errors} hata, {len(issues) - errors} uyarı/bilgi.")
        return 1 if errors else 0

    last = {"t": time.perf_counter(), "n": 0}
    shown: dict[str, str | None] = {}
    production = _production(graph, flow)       # counts for "Rapor Oluştur", like the operator panel collects

    def on_frame(n, results):
        while graph.recipe_events:                          # switched by a 'Reçete Seç' block
            name, errors = graph.recipe_events.pop(0)
            print(f"Reçete uygulandı: {name}")
            if api is not None:
                api.recipe_applied(name)
            for e in errors:
                print(f"[UYARI] Reçete '{name}': {e}", file=sys.stderr)
        flat = flatten(graph, results)                      # blocks inside sub-flows too ("subflow/inner" ids)
        if api is not None:
            api.publish(graph, flat)
        if production is not None:
            production.feed(flat)
        for nid, r in flat.items():
            if r.error == "Önceki blok hata verdi":
                continue
            node = node_at(graph, nid)
            if node is None or inner_error(node, r.error):
                continue
            if r.error and shown.get(nid) != r.error:   # print each new error once, not every frame
                print(f"[HATA] {node.title} ({nid}): {r.error}", file=sys.stderr)
                try:
                    error_log().record(node.type_name, r.error, getattr(r, "where", ""))
                except Exception:  # noqa: BLE001
                    pass
            shown[nid] = r.error
        now = time.perf_counter()
        if now - last["t"] >= stats:
            rate = (n - last["n"]) / (now - last["t"])
            slow = max(results.items(), key=lambda kv: kv[1].ms)
            print(f"{n} kare, {rate:.1f} FPS, en yavaş: {graph.nodes[slow[0]].title} {slow[1].ms:.1f} ms")
            last.update(t=now, n=n)

    if api is not None:
        api.attach(graph)
    try:
        done = graph.run(max_frames=frames or None, on_frame=on_frame, should_stop=should_stop, max_fps=fps)
    finally:
        if api is not None:
            api.detach()
        if production is not None:
            production.flush()                          # what was counted since the last minute sample
    print(f"Bitti: {done} kare işlendi.")
    return 0


class _Production:
    """Production data of a headless run of a flow with operator panel settings: a Tally fed every frame, written to
    the flow's production database every minute and at the end (the same as the operator panel does)."""

    def __init__(self, graph, flow: str):
        from .production import ProductionDB, Tally, db_path
        self.graph = graph
        self.tally = Tally(graph.operator)
        self.db = ProductionDB(db_path(flow))
        self.name = pathlib.Path(flow).stem
        self.last = _clock()

    def feed(self, flat: dict) -> None:
        from .production import SAMPLE_S
        try:          # counting must never stop the flow
            self.tally.feed(flat, self.graph)
        except Exception as exc:  # noqa: BLE001
            print(f"[UYARI] Sayım hatası: {exc}", file=sys.stderr)
        if _clock() - self.last >= SAMPLE_S:
            self.flush()

    def flush(self) -> None:
        from .production import store_sample
        self.last = _clock()
        try:
            store_sample(self.tally, self.db, self.name)
        except Exception as exc:  # noqa: BLE001  (a full disk must not stop the line)
            print(f"[UYARI] Üretim verisi kaydedilemedi: {exc}", file=sys.stderr)


def _production(graph, flow: str):
    return _Production(graph, flow) if graph.operator else None


def _set_selectors(graph, name: str) -> None:
    """Every 'Reçete Seç' block, inside sub-flows too, set to recipe `name`."""
    def walk(nodes):
        for nd in nodes:
            if not isinstance(nd, dict):
                continue
            if nd.get("type") == "RecipeSelect" and isinstance(nd.get("values"), dict):
                nd["values"]["recipe"] = name
            elif nd.get("type") == "SubFlow" and isinstance(nd.get("inner"), dict):
                walk(nd["inner"].get("nodes") or [])
    for node in graph.nodes.values():
        if node.type_name == "RecipeSelect":
            node.values["recipe"] = name
        elif node.type_name == "SubFlow":
            walk(node.inner.get("nodes") or [])


def watchdog(run: Callable[[], int], should_stop: Callable[[], bool] = lambda: False,
             log: logging.Logger | None = None, max_restarts: int = 0) -> int:
    """Run ``run()`` and start it again after a crash, waiting 1, 2, 4 … 60 s between attempts. A run that
    lasted STABLE_SECONDS resets the wait. ``max_restarts`` 0 = forever. Returns the last exit code."""
    log = log or logging.getLogger(LOG_NAME)
    delay, restarts = BACKOFF_START, 0
    while True:
        started = _clock()
        log.info("Akış başlatılıyor (yeniden başlatma: %d)", restarts)
        try:
            code = run()
            log.info("Akış bitti (çıkış kodu %s)", code)
            return code
        except KeyboardInterrupt:
            log.info("Kullanıcı durdurdu")
            return 0
        except Exception as exc:  # noqa: BLE001
            msg = f"Akış çöktü: {type(exc).__name__}: {exc}"
            print(f"[HATA] {msg}", file=sys.stderr)
            log.error("%s\n%s", msg, traceback.format_exc())
        if should_stop():
            return 1
        if max_restarts and restarts >= max_restarts:
            log.error("En fazla yeniden başlatma sayısına (%d) ulaşıldı, vazgeçildi", max_restarts)
            print(f"En fazla yeniden başlatma sayısına ({max_restarts}) ulaşıldı.", file=sys.stderr)
            return 1
        if _clock() - started >= STABLE_SECONDS:
            delay = BACKOFF_START
        print(f"{delay:g} sn sonra yeniden başlatılacak...", file=sys.stderr)
        log.warning("%g sn sonra yeniden başlatılacak", delay)
        waited = 0.0
        while waited < delay and not should_stop():    # stays responsive to a stop request while waiting
            step = min(0.5, delay - waited)
            _sleep(step)
            waited += step
        if should_stop():
            return 1
        restarts += 1
        delay = min(delay * 2, BACKOFF_MAX)


def _serve_api(args, should_stop, link_factory=None) -> int:
    """--api: the flow runs here and can be started / stopped / switched over the REST API (api_server.py) until
    the program is stopped. When the flow ends it waits for a 'baslat' request."""
    from . import api_server
    if not 0 <= args.api <= 65535:
        print("API portu 0-65535 arasında olmalı", file=sys.stderr)
        return 2
    settings = api_server.load_settings()
    settings.update(port=args.api, host="0.0.0.0" if args.api_ag else "127.0.0.1", require_key=True)
    if args.api_sertifika or args.api_sertifika_anahtari:
        settings.update(https=True, cert=args.api_sertifika or "", keyfile=args.api_sertifika_anahtari or "")
    else:
        settings["https"] = False
    state = api_server.FlowState()
    ctl = api_server.HeadlessController(args.flow, state, fps=args.fps, plugins=args.plugins, recete=args.recete,
                                        frames=args.frames, stats=args.stats, autostart=not args.api_bekle)
    ctl.restart_on_crash = args.yeniden_baslat
    ctl.max_restarts = max(0, args.en_fazla_yeniden)        # the watchdog's limit works with --api too
    if args.yeniden_baslat:
        ctl.log = get_logger()
    server = api_server.ApiServer(ctl, state, settings)
    try:
        url = server.start()
    except (ValueError, RuntimeError) as exc:
        print(f"API açılamadı: {exc}", file=sys.stderr)
        return 2
    print(f"Uzaktan kontrol API'si: {url}  (açıklama: {url}/openapi.json)")
    if args.api_anahtar_goster:
        print(f"API anahtarı: {server.settings['key']}")
    else:
        print(f"API anahtarı: {api_server.settings_path()} dosyasında (göstermek için --api-anahtar-goster)")
    link = link_factory(state, ctl) if link_factory else None
    try:
        return ctl.serve(should_stop)
    finally:
        server.stop()
        if link is not None:
            link.close()


def main(argv=None, should_stop: Callable[[], bool] | None = None) -> int:
    """Command line of the headless runner. ``should_stop`` (a Windows service's stop event) is checked besides
    Ctrl+C / SIGTERM."""
    ap = argparse.ArgumentParser(description="VisionFlow akışını arayüz olmadan çalıştırır")
    ap.add_argument("flow", help="akış dosyası (.json)")
    ap.add_argument("--frames", type=int, default=0, help="bu kadar kareden sonra dur (0 = sınırsız)")
    ap.add_argument("--plugins", default="plugins", help="ek blok klasörü")
    ap.add_argument("--fps", type=float, default=30, help="en fazla FPS (0 = sınırsız; varsayılan 30, işlemciyi boşuna yormaz)")
    ap.add_argument("--check", action="store_true", help="sadece akışı denetle (eksik bağlantı, ayar), çalıştırma")
    ap.add_argument("--recete", metavar="AD", help="bu reçeteyle (ürün tipiyle) başla: adı veya numarası (1'den)")
    ap.add_argument("--stats", type=float, default=5.0, help="kaç saniyede bir istatistik yazılsın")
    ap.add_argument("--yeniden-baslat", action="store_true",
                    help="akış çökerse bekleyip yeniden başlat (servis kullanımı); kayıt: <VisionFlow klasörü>/kayitlar")
    ap.add_argument("--en-fazla-yeniden", type=int, default=0, help="en fazla yeniden başlatma (0 = sınırsız)")
    ap.add_argument("--api", type=int, metavar="PORT",
                    help="uzaktan kontrol REST API'sini bu portta aç (MES/ERP için; anahtar X-API-Key başlığında)")
    ap.add_argument("--api-ag", action="store_true", help="API'yi ağa aç (0.0.0.0); varsayılan sadece bu bilgisayar")
    ap.add_argument("--api-sertifika", metavar="PEM", help="HTTPS için sertifika dosyası")
    ap.add_argument("--api-sertifika-anahtari", metavar="PEM", help="HTTPS için sertifikanın özel anahtarı")
    ap.add_argument("--api-anahtar-goster", action="store_true", help="API anahtarını ekrana yaz")
    ap.add_argument("--api-bekle", action="store_true", help="akışı hemen başlatma, API'den 'baslat' gelmesini bekle")
    # used by the station manager (stations.py) for the flows it runs
    ap.add_argument("--durum-dosyasi", metavar="JSON", help="durum (çalışıyor, FPS, kare, hatalar) bu dosyaya yazılır")
    ap.add_argument("--onizleme-dosyasi", metavar="JPG", help="önizleme görüntüsü saniyede bir bu dosyaya yazılır")
    ap.add_argument("--durdur-dosyasi", metavar="DOSYA", help="bu dosya oluşunca akış düzgünce durdurulur")
    ap.add_argument("--ust-pid", type=int, metavar="PID", help="bu numaralı işlem kapanınca akış durdurulur")
    args = ap.parse_args(argv)

    stop = {"flag": False}
    outer = should_stop or (lambda: False)

    def stopped() -> bool:
        return stop["flag"] or outer()
    # Ctrl+C, a service stop or `docker stop`: finish the frame and close recordings properly
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), lambda *_: stop.update(flag=True))
            except ValueError:          # not the main thread (e.g. inside a Windows service)
                pass

    linked = not args.check and any(v is not None for v in (args.durum_dosyasi, args.onizleme_dosyasi,
                                                                  args.durdur_dosyasi, args.ust_pid))

    def make_link(state, ctl):
        from .stations import StationLink
        return StationLink(state, ctl, status_path=args.durum_dosyasi, preview_path=args.onizleme_dosyasi,
                           stop_path=args.durdur_dosyasi, parent_pid=args.ust_pid,
                           on_stop=lambda: stop.update(flag=True)).start()

    if args.api is not None and not args.check:
        return _serve_api(args, stopped, make_link if linked else None)

    hook = link = None
    if linked:                  # the station manager reads this run's status from a file (no API needed)
        from . import api_server
        state = api_server.FlowState()
        hook = api_server.HeadlessController(args.flow, state, recete=args.recete)
        link = make_link(state, hook)

    def run():
        try:
            return run_flow(args.flow, args.frames, args.fps, args.stats, args.plugins,
                            should_stop=stopped, check_only=args.check, recete=args.recete, api=hook)
        except Exception as exc:
            if hook is not None:
                hook.state.note_error("akış", f"Akış çöktü: {type(exc).__name__}: {exc}")
            raise

    try:
        if not args.yeniden_baslat or args.check:
            return run()
        return watchdog(run, stopped, get_logger(), args.en_fazla_yeniden)
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    sys.exit(main())
