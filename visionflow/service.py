"""Run a flow unattended at boot: Windows service (pywin32), Windows Task Scheduler or a Linux systemd unit.

    python -m visionflow servis kur|kaldir|baslat|durdur|durum --akis akis.json [--ad AD] [--sistem] [--yazdir]
    python -m visionflow servis kur --akis akis.json --api 8790          (with the remote control REST API)
    python -m visionflow servis kur --istasyonlar hat.json               (every station of a station file)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
import threading
from typing import Callable

ACTIONS = ("kur", "kaldir", "baslat", "durdur", "durum")
PREFIX = "VisionFlow"
TR_LIMIT = 261              # schtasks /TR is limited to 261 characters

PYWIN32_MISSING = ("pywin32 kurulu değil, bu yüzden gerçek bir Windows servisi kurulamıyor.\n"
                   "  Servis olarak kurmak için:  pip install pywin32   (sonra komutu yeniden çalıştırın)\n"
                   "  Şimdilik Görev Zamanlayıcı (schtasks) kullanılıyor: akış bilgisayar açılınca / oturum açılınca "
                   "kendiliğinden başlar, çökerse yeniden başlatılır.")


def safe_name(name: str) -> str:
    """Service / unit / task name: letters, digits, '-' and '_' only ("Hat akışı" → "Hat_akisi")."""
    import unicodedata
    s = (name or "").translate(str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_-")
    return s[:60] or "akis"


def _program() -> list[str]:
    """How to start VisionFlow: the packaged .exe or python -m visionflow."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "visionflow"]


def headless_command(flow: str, api: int | None = None, api_net: bool = False) -> list[str]:
    """The command line that runs the flow headless with crash restarts (the packaged .exe or python -m); with
    ``api`` the remote control REST API listens on that port (``api_net``: on the network too)."""
    flow = str(pathlib.Path(flow).resolve())
    cmd = _program() + ["--headless", flow, "--yeniden-baslat"]
    if api is not None:
        cmd += ["--api", str(int(api))] + (["--api-ag"] if api_net else [])
    return cmd


def stations_command(path: str) -> list[str]:
    """The command line that runs every station of a station file (stations.py) under its supervisor."""
    return _program() + ["istasyonlar", str(pathlib.Path(path).resolve())]


def command_for(args) -> list[str]:
    """The command a service installed with these 'servis' arguments runs."""
    if getattr(args, "istasyonlar", None):
        return stations_command(args.istasyonlar)
    return headless_command(args.akis, getattr(args, "api", None), getattr(args, "api_ag", False))


def working_dir() -> str:
    """Folder the service starts in: the program folder (so relative models/plugins are found)."""
    if getattr(sys, "frozen", False):
        return str(pathlib.Path(sys.executable).parent)
    return str(pathlib.Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------- Linux / systemd
def _systemd_quote(arg: str) -> str:
    if arg and not re.search(r'[\s"\'\\$%;]', arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def unit_name(name: str) -> str:
    return f"visionflow-{safe_name(name)}.service"


def unit_path(name: str, system: bool) -> pathlib.Path:
    if system:
        return pathlib.Path("/etc/systemd/system") / unit_name(name)
    cfg = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config")
    return cfg / "systemd" / "user" / unit_name(name)


def systemd_unit(flow: str, name: str, system: bool = False, user: str | None = None,
                 command: list[str] | None = None) -> str:
    cmd = " ".join(_systemd_quote(a) for a in (command or headless_command(flow)))
    lines = [
        "[Unit]",
        f"Description=VisionFlow akışı: {safe_name(name)}",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={_systemd_quote(working_dir())}",
        f"ExecStart={cmd}",
        "Restart=always",
        "RestartSec=5",
        "KillSignal=SIGTERM",
        "TimeoutStopSec=20",
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=QT_QPA_PLATFORM=offscreen",
    ]
    if system and user:
        lines.append(f"User={user}")
    lines += ["", "[Install]", f"WantedBy={'multi-user.target' if system else 'default.target'}", ""]
    return "\n".join(lines)


def systemd_commands(action: str, name: str, system: bool) -> list[list[str]]:
    ctl = ["systemctl"] + ([] if system else ["--user"])
    unit = unit_name(name)
    return {
        "kur": [ctl + ["daemon-reload"], ctl + ["enable", "--now", unit]],
        "kaldir": [ctl + ["disable", "--now", unit], ctl + ["daemon-reload"]],
        "baslat": [ctl + ["start", unit]],
        "durdur": [ctl + ["stop", unit]],
        "durum": [ctl + ["status", "--no-pager", unit]],
    }[action]


# ---------------------------------------------------------------- Windows / Task Scheduler
def task_name(name: str) -> str:
    return f"{PREFIX}\\{safe_name(name)}"


def task_run_line(flow: str, command: list[str] | None = None) -> str:
    """The /TR value: every part in double quotes when it has spaces."""
    return subprocess.list2cmdline(command or headless_command(flow))


def schtasks_commands(action: str, name: str, flow: str | None = None, system: bool = False,
                      command: list[str] | None = None) -> list[list[str]]:
    tn = task_name(name)
    if action == "kur":
        cmd = ["schtasks", "/Create", "/TN", tn, "/TR", task_run_line(flow, command)]
        # at boot as SYSTEM (needs an administrator console), else when this user logs on
        cmd += ["/SC", "ONSTART", "/RU", "SYSTEM", "/RL", "HIGHEST"] if system else ["/SC", "ONLOGON"]
        return [cmd + ["/F"]]
    return [{
        "kaldir": ["schtasks", "/Delete", "/TN", tn, "/F"],
        "baslat": ["schtasks", "/Run", "/TN", tn],
        "durdur": ["schtasks", "/End", "/TN", tn],
        "durum": ["schtasks", "/Query", "/TN", tn, "/V", "/FO", "LIST"],
    }[action]]


def show(cmd: list[str], windows: bool) -> str:
    return subprocess.list2cmdline(cmd) if windows else shlex.join(cmd)


# ---------------------------------------------------------------- Windows / pywin32 service
def have_pywin32() -> bool:
    try:
        import win32serviceutil  # noqa: F401
        return True
    except ImportError:
        return False


def _make_service_class():
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil

    class VisionFlowService(win32serviceutil.ServiceFramework):
        _svc_name_ = "VisionFlow"
        _svc_display_name_ = "VisionFlow akışı"

        def __init__(self, args):
            super().__init__(args)
            self.svc_name = args[0]            # the name it was installed under (VisionFlow_<ad>)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.stopping = threading.Event()

        def SvcStop(self):                     # noqa: N802 (pywin32 API)
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            self.stopping.set()                # the frame loop finishes its frame and closes recordings
            win32event.SetEvent(self.stop_event)

        def SvcDoRun(self):                    # noqa: N802
            root = str(pathlib.Path(__file__).resolve().parents[1])
            if root not in sys.path:           # pythonservice.exe loads this file by path, not as a package
                sys.path.insert(0, root)
            from visionflow import headless
            opt = lambda key: win32serviceutil.GetServiceCustomOption(self.svc_name, key)  # noqa: E731
            stations, flow, api = opt("istasyonlar"), opt("akis"), opt("api")
            servicemanager.LogInfoMsg(f"VisionFlow servisi başladı: {stations or flow}")
            os.chdir(working_dir())
            if stations:                       # a station file: every station in its own process
                from visionflow.stations import run_supervisor
                run_supervisor(stations, self.stopping.is_set)
            else:
                argv = [flow, "--yeniden-baslat"]
                if api not in (None, ""):
                    argv += ["--api", str(api)] + (["--api-ag"] if opt("api_ag") else [])
                headless.main(argv, should_stop=self.stopping.is_set)
            servicemanager.LogInfoMsg("VisionFlow servisi durdu")

    return VisionFlowService


def __getattr__(name):                         # pythonservice.exe imports visionflow.service.VisionFlowService
    if name == "VisionFlowService":
        cls = _make_service_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


def pywin32_action(action: str, name: str, flow: str | None, out=print, stations: str | None = None,
                   api: int | None = None, api_net: bool = False) -> int:
    import win32service
    import win32serviceutil
    svc = f"{PREFIX}_{safe_name(name)}"
    if action == "kur":
        # full path form: pythonservice.exe finds the module without the program folder on its sys.path
        cls_string = str(pathlib.Path(__file__).resolve().with_suffix("")) + ".VisionFlowService"
        win32serviceutil.InstallService(cls_string, svc, f"VisionFlow akışı: {safe_name(name)}",
                                        startType=win32service.SERVICE_AUTO_START,
                                        description="VisionFlow akışını arayüz olmadan çalıştırır")
        if stations:
            win32serviceutil.SetServiceCustomOption(svc, "istasyonlar", str(pathlib.Path(stations).resolve()))
        else:
            win32serviceutil.SetServiceCustomOption(svc, "akis", str(pathlib.Path(flow).resolve()))
            if api is not None:
                win32serviceutil.SetServiceCustomOption(svc, "api", int(api))
                win32serviceutil.SetServiceCustomOption(svc, "api_ag", 1 if api_net else 0)
        win32serviceutil.StartService(svc)
        out(f"Windows servisi kuruldu ve başlatıldı: {svc}")
    elif action == "kaldir":
        try:
            win32serviceutil.StopService(svc)
        except Exception:  # noqa: BLE001 - already stopped
            pass
        win32serviceutil.RemoveService(svc)
        out(f"Windows servisi kaldırıldı: {svc}")
    elif action == "baslat":
        win32serviceutil.StartService(svc)
        out(f"Başlatıldı: {svc}")
    elif action == "durdur":
        win32serviceutil.StopService(svc)
        out(f"Durduruldu: {svc}")
    else:
        state = win32serviceutil.QueryServiceStatus(svc)[1]
        names = {1: "durmuş", 2: "başlıyor", 3: "duruyor", 4: "çalışıyor", 7: "duraklatılmış"}
        out(f"{svc}: {names.get(state, state)}")
    return 0


# ---------------------------------------------------------------- command line
def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m visionflow servis",
                                 description="Bir akışı bilgisayar açılınca kendiliğinden çalıştırır "
                                             "(Windows servisi / Görev Zamanlayıcı, Linux systemd)")
    ap.add_argument("islem", choices=ACTIONS, help="kur, kaldir, baslat, durdur veya durum")
    ap.add_argument("--akis", help="akış dosyası (.json); 'kur' için bu ya da --istasyonlar gerekli")
    ap.add_argument("--istasyonlar", metavar="DOSYA",
                    help="istasyon dosyası (.json): içindeki bütün istasyonlar birlikte çalışır (İstasyon yöneticisi)")
    ap.add_argument("--api", type=int, metavar="PORT",
                    help="uzaktan kontrol REST API'sini bu portta aç (anahtar: uzaktan_api.json)")
    ap.add_argument("--api-ag", action="store_true", help="API'yi ağa aç (0.0.0.0); varsayılan sadece bu bilgisayar")
    ap.add_argument("--ad", help="servis adı (varsayılan: akış dosyasının adı)")
    ap.add_argument("--sistem", action="store_true",
                    help="bilgisayar açılınca, oturum açılmadan başlat (yönetici / root gerekir)")
    ap.add_argument("--kullanici", help="Linux sistem servisi hangi kullanıcıyla çalışsın")
    ap.add_argument("--yazdir", action="store_true", help="hiçbir şey kurma, sadece dosyayı ve komutları yazdır")
    ap.add_argument("--yontem", choices=("otomatik", "pywin32", "gorev", "systemd"), default="otomatik",
                    help="kurulum yöntemi (varsayılan: işletim sistemine göre)")
    args = ap.parse_args(argv)
    if args.akis and args.istasyonlar:
        ap.error("--akis ile --istasyonlar birlikte verilemez: ya bir akış ya da bir istasyon dosyası")
    if args.islem == "kur" and not args.akis and not args.istasyonlar:
        ap.error("'kur' için --akis akis.json (ya da --istasyonlar hat.json) gerekli")
    if args.akis and args.islem == "kur" and not pathlib.Path(args.akis).is_file():
        ap.error(f"akış dosyası bulunamadı: {args.akis}")
    if args.api is not None and not 1 <= args.api <= 65535:
        ap.error("API portu 1-65535 arasında olmalı")
    if args.istasyonlar:
        if args.api is not None:
            ap.error("--api istasyon dosyasıyla kullanılmaz: her istasyonun API portu dosyada (api_portu) seçilir")
        if args.islem == "kur":
            from .stations import load_stations
            try:
                load_stations(args.istasyonlar)
            except ValueError as exc:
                ap.error(str(exc))
    if not args.ad and not args.akis and not args.istasyonlar:
        ap.error("--ad, --akis veya --istasyonlar verilmeli")
    args.ad = safe_name(args.ad or pathlib.Path(args.akis or args.istasyonlar).stem)
    return args


def method_for(args, platform: str | None = None) -> str:
    platform = platform or sys.platform
    if args.yontem != "otomatik":
        return args.yontem
    if platform == "win32":
        # the packaged .exe cannot host a pywin32 service: it uses the Task Scheduler
        return "pywin32" if have_pywin32() and not getattr(sys, "frozen", False) else "gorev"
    return "systemd"


def _run(cmd: list[str]):
    # no console window flashing up from the windowed .exe (installer, editor)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    return subprocess.run(cmd, creationflags=flags)


def run_commands(cmds: list[list[str]], windows: bool, dry: bool, out=print, runner: Callable = _run) -> int:
    code = 0
    for cmd in cmds:
        out(("  " if dry else "$ ") + show(cmd, windows))
        if dry:
            continue
        try:
            res = runner(cmd)
        except FileNotFoundError:
            out(f"Komut bulunamadı: {cmd[0]}")
            return 1
        code = getattr(res, "returncode", 0) or code
    return code


def main(argv=None, out=print, runner: Callable = _run, platform: str | None = None) -> int:
    args = parse_args(argv)
    method = method_for(args, platform)
    if method == "pywin32":
        if not have_pywin32():
            out(PYWIN32_MISSING)
            method = "gorev"
        elif args.yazdir:
            what = f"istasyonlar: {args.istasyonlar}" if args.istasyonlar else f"akış: {args.akis or '-'}"
            out(f"pywin32 servisi: VisionFlow_{args.ad} ({args.islem}); çalışacak {what}"
                + (f", API portu {args.api}" if args.api is not None else ""))
            return 0
        else:
            try:
                return pywin32_action(args.islem, args.ad, args.akis, out, args.istasyonlar, args.api, args.api_ag)
            except Exception as exc:  # noqa: BLE001
                out(f"Servis işlemi başarısız: {exc}\n  (Yönetici olarak açılmış bir komut isteminde deneyin.)")
                return 1
    elif (platform or sys.platform) == "win32" and method == "gorev" and args.yontem == "otomatik" \
            and not getattr(sys, "frozen", False):
        out(PYWIN32_MISSING)

    if method == "gorev":
        command = command_for(args) if args.islem == "kur" else None
        cmds = schtasks_commands(args.islem, args.ad, args.akis, args.sistem, command)
        if args.islem == "kur" and len(task_run_line(args.akis, command)) > TR_LIMIT:
            out(f"Uyarı: çalıştırma satırı {TR_LIMIT} karakterden uzun, Görev Zamanlayıcı kabul etmeyebilir; "
                "akışı daha kısa bir klasöre taşıyın.")
        if args.yazdir:
            out("Görev Zamanlayıcı komutları:")
        code = run_commands(cmds, True, args.yazdir, out, runner)
        if not args.yazdir and code == 0:
            out({"kur": f"Görev kuruldu: {task_name(args.ad)} "
                        f"({'bilgisayar açılınca' if args.sistem else 'oturum açılınca'} başlar)",
                 "kaldir": f"Görev kaldırıldı: {task_name(args.ad)}"}.get(args.islem, "Tamam."))
        return code

    # systemd
    path = unit_path(args.ad, args.sistem)
    cmds = systemd_commands(args.islem, args.ad, args.sistem)
    if args.islem == "kur":
        text = systemd_unit(args.akis, args.ad, args.sistem, args.kullanici, command_for(args))
        if args.yazdir:
            out(f"# {path}\n{text}")
            out("Kurmak için:")
            return run_commands(cmds, False, True, out, runner)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            out(f"Servis dosyası yazılamadı ({path}): {exc}\n  Sistem servisi için 'sudo' ile çalıştırın "
                "veya --sistem olmadan kullanıcı servisi kurun.")
            return 1
        out(f"Servis dosyası yazıldı: {path}")
        code = run_commands(cmds, False, False, out, runner)
        if not args.sistem:
            out("Not: oturum açmadan da başlaması için bir kez çalıştırın:  loginctl enable-linger " +
                (os.environ.get("USER") or "$USER"))
        return code
    if args.islem == "kaldir":
        code = run_commands(cmds[:1], False, args.yazdir, out, runner)
        if args.yazdir:
            out(f"  rm {shlex.quote(str(path))}")
        else:
            try:
                path.unlink()
                out(f"Servis dosyası silindi: {path}")
            except FileNotFoundError:
                out(f"Servis dosyası zaten yok: {path}")
            except OSError as exc:
                out(f"Servis dosyası silinemedi ({path}): {exc}")
                code = code or 1
        return run_commands(cmds[1:], False, args.yazdir, out, runner) or code
    return run_commands(cmds, False, args.yazdir, out, runner)


if __name__ == "__main__":
    sys.exit(main())
