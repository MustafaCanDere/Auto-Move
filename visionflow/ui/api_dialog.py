"""Araçlar → Uzaktan kontrol (API)…: the REST API of api_server.py inside the editor.

The API's request threads never touch the editor: start / stop / recipe changes are run on the GUI thread (through
a queued Qt signal, waiting for the answer), and the values / pictures come from the FlowState that the flow worker
publishes to after every frame.
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

from .. import api_server

GUI_TIMEOUT = 15.0
HOST_LABELS = {"127.0.0.1": "Sadece bu bilgisayar (127.0.0.1)", "0.0.0.0": "Yerel ağ (0.0.0.0, tüm ağ kartları)"}


class _Invoker(QObject):
    call = Signal(object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run, Qt.QueuedConnection)

    def _run(self, job):
        job()

    def run(self, fn, timeout: float = GUI_TIMEOUT):
        """Run fn() on the GUI thread and return its result (raises what it raised)."""
        if QThread.currentThread() is self.thread():
            return fn()
        done, box = threading.Event(), {}

        def job():
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done.set()
        self.call.emit(job)
        if not done.wait(timeout):
            raise TimeoutError(f"Program {timeout:.0f} sn içinde cevap vermedi (açık bir pencere bekliyor olabilir)")
        if "error" in box:
            raise box["error"]
        return box.get("value")


class EditorController:
    """api_server's controller for the editor window (every call is done on the GUI thread)."""

    def __init__(self, win, state: api_server.FlowState):
        self.win, self.state = win, state
        self.invoker = _Invoker()

    def flow_name(self) -> str:
        path = self.win.path                      # a plain attribute: read without bothering the GUI thread
        return os.path.basename(path) if path else "Yeni akış"

    def _panel(self):
        """The operator panel, while it runs the flow instead of the (hidden) editor."""
        fn = getattr(self.win, "operator_panel", None)
        return fn() if fn is not None else None

    def start(self) -> str:
        def go():
            w = self.win
            panel = self._panel()
            if panel is not None:              # the panel runs the flow: never a second copy in the editor
                if panel.worker is not None:
                    return "Akış zaten çalışıyor"
                panel.start()
                if panel.worker is None:
                    raise ValueError(f"Akış başlatılamadı: {panel.status.text().strip()}")
                return "Akış başlatıldı (operatör paneli)"
            if w.worker is not None:
                return "Akış zaten çalışıyor"
            if not w.root.nodes:
                raise ValueError("Akışta blok yok")
            try:
                w.root.order()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Akış hatalı: {exc}") from None
            w.start()
            if w.worker is None:
                raise ValueError("Akış başlatılamadı")
            return "Akış başlatıldı"
        return self.invoker.run(go)

    def stop(self) -> str:
        def go():
            panel = self._panel()
            if panel is not None:
                if panel.worker is None:
                    return "Akış zaten durmuş"
                panel.stop()
                if panel.worker is not None:
                    raise TimeoutError("Akış durmadı (bir cihaz yanıt vermiyor olabilir)")
                return "Akış durduruldu (operatör paneli)"
            if self.win.worker is None:
                return "Akış zaten durmuş"
            self.win.stop()
            if self.win.worker is not None:
                raise TimeoutError("Akış durmadı (bir cihaz yanıt vermiyor olabilir)")
            return "Akış durduruldu"
        return self.invoker.run(go)

    def set_recipe(self, wanted):
        def go():
            from ..core import recipes
            self.win._sync()
            r = recipes.find(self.win.root, wanted)          # ValueError: no such recipe
            errors = self.win.apply_recipe(r["name"])
            # 'Reçete Seç' blocks follow, so the next start does not switch back (as headless --api does)
            from ..headless import _set_selectors
            _set_selectors(self.win.root, r["name"])
            panel = self._panel()
            if panel is not None:                            # its flow copy follows too (win.apply_recipe)
                _set_selectors(panel.graph, r["name"])
                panel.graph_data = panel.graph.to_dict()
            self.win._values_changed()
            self.win.statusBar().showMessage(f"Reçete (API): {r['name']}", 4000)
            return r["name"], list(errors)
        return self.invoker.run(go)

    def recipes(self) -> list[str]:
        from ..core import recipes
        return self.invoker.run(lambda: recipes.names(self.win.root))

    def blocks(self) -> list[dict]:
        return self.invoker.run(lambda: api_server.blocks_of(self.win.root))

    def current_recipe(self) -> str:
        """The recipe of the flow open now (a plain attribute of the window; "" for a newly opened flow)."""
        return getattr(self.win, "active_recipe", "") or ""


class PanelController:
    """api_server's controller for an operator panel started without an editor (``--operator akis.json``)."""

    def __init__(self, panel, state: api_server.FlowState):
        self.panel, self.state = panel, state
        self.invoker = _Invoker()

    def flow_name(self) -> str:
        path = self.panel.flow_path
        return os.path.basename(path) if path else "Adsız akış"

    def start(self) -> str:
        def go():
            p = self.panel
            if p.worker is not None:
                return "Akış zaten çalışıyor"
            p.start()
            if p.worker is None:
                raise ValueError(f"Akış başlatılamadı: {p.status.text().strip()}")
            return "Akış başlatıldı (operatör paneli)"
        return self.invoker.run(go)

    def stop(self) -> str:
        def go():
            p = self.panel
            if p.worker is None:
                return "Akış zaten durmuş"
            p.stop()
            if p.worker is not None:
                raise TimeoutError("Akış durmadı (bir cihaz yanıt vermiyor olabilir)")
            return "Akış durduruldu (operatör paneli)"
        return self.invoker.run(go)

    def set_recipe(self, wanted):
        def go():
            from ..core import recipes
            p = self.panel
            r = recipes.find(p.graph, wanted)               # ValueError: no such recipe
            # into the panel's flow (its designed screen shows it, the next start keeps it, 'Reçete Seç' blocks
            # follow) and the running flow switches between frames
            errors = p.apply_recipe(r["name"])
            return r["name"], list(errors)
        return self.invoker.run(go)

    def recipes(self) -> list[str]:
        from ..core import recipes
        return recipes.names(self.panel.graph)

    def blocks(self) -> list[dict]:
        return api_server.blocks_of(self.panel.graph)

    def current_recipe(self) -> str:
        return getattr(self.panel, "recipe_name", "") or ""


class ApiManager:
    """The API of one editor window (or of an operator panel without an editor): its FlowState, controller and
    (while switched on) server."""

    def __init__(self, win):
        self.win = win
        self.state = api_server.FlowState()
        is_panel = hasattr(win, "graph_data") and not hasattr(win, "root")
        self.controller = PanelController(win, self.state) if is_panel else EditorController(win, self.state)
        self.server: api_server.ApiServer | None = None

    @property
    def running(self) -> bool:
        return self.server is not None and self.server.running

    def start(self, settings: dict) -> str:
        self.stop()
        server = api_server.ApiServer(self.controller, self.state, settings)
        url = server.start()                    # ValueError / RuntimeError with a Turkish message
        self.server = server
        for w in self._workers():
            if w.api is None:                   # a flow already running: report from now on
                self.state.attach(getattr(w, "recipe_name", ""))
                w.api = self.state
        return url

    def _workers(self) -> list:
        """The running flow workers of the editor and of the operator panel opened from it."""
        out = [getattr(self.win, "worker", None)]
        fn = getattr(self.win, "operator_panel", None)
        panel = fn() if fn is not None else None
        if panel is not None:
            out.append(panel.worker)
        return [w for w in out if w is not None]

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        for w in self._workers():
            w.api = None
        self.state.detach()

    def worker_state(self):
        return self.state if self.running else None


def manager(win) -> ApiManager:
    m = getattr(win, "_api_manager", None)
    if m is None:
        m = win._api_manager = ApiManager(win)
    return m


def worker_state(win):
    """For MainWindow.start(): the FlowState a new worker reports to, None when the API is off."""
    m = getattr(win, "_api_manager", None)
    return m.worker_state() if m is not None else None


def autostart(win, log=None) -> None:
    """Editor (or ``--operator`` panel) start-up: switch the API on when it was left on."""
    log = log or (lambda m: win.logp.log(m))
    try:
        s = api_server.load_settings()
    except Exception:  # noqa: BLE001
        return
    if not s["enabled"]:
        return
    try:
        url = manager(win).start(s)
        log(f"Uzaktan kontrol API'si açık: {url}")
    except Exception as exc:  # noqa: BLE001
        log(f"Uzaktan kontrol API'si açılamadı: {exc}")


def shutdown(win) -> None:
    m = getattr(win, "_api_manager", None)
    if m is not None:
        m.stop()


class ApiDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.mgr = manager(win)
        self.setWindowTitle("Uzaktan kontrol (API)")
        self.resize(760, 640)
        self.settings = api_server.load_settings()
        lay = QVBoxLayout(self)
        hint = QLabel("MES, ERP, SCADA veya kendi programınız VisionFlow'u HTTP üzerinden izleyip kontrol edebilir: "
                      "durum, başlat/durdur, blok değerleri, görüntü, 'API Tetik' blokları, reçete ve değişkenler. "
                      "Her istekte X-API-Key başlığında anahtar gönderilmelidir. Ayrıntılar: /api/v1/openapi.json")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)

        form = QFormLayout()
        self.enabled = QCheckBox("API açık (program açılınca da açılsın)")
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.host = QComboBox()
        for k, v in HOST_LABELS.items():
            self.host.addItem(v, k)
        self.require = QCheckBox("API anahtarı zorunlu (önerilir; ağdan erişimde her zaman zorunlu)")
        self.key = QLineEdit()
        self.key.setReadOnly(True)
        self.show_key = QCheckBox("göster")
        self.show_key.toggled.connect(lambda on: self.key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        copy_key = QPushButton("Kopyala")
        copy_key.clicked.connect(lambda: self._copy(self.key.text(), "Anahtar kopyalandı"))
        regen = QPushButton("Yeni anahtar…")
        regen.clicked.connect(self.regenerate)
        key_row = QHBoxLayout()
        for wdg in (self.key, self.show_key, copy_key, regen):
            key_row.addWidget(wdg)
        key_row.setStretch(0, 1)
        self.rate = QSpinBox()
        self.rate.setRange(0, 10000)
        self.rate.setSuffix(" istek/sn")
        self.rate.setSpecialValueText("sınırsız")
        self.https = QCheckBox("HTTPS (şifreli bağlantı)")
        self.cert = QLineEdit()
        self.cert.setPlaceholderText("sertifika (.pem / .crt)")
        self.keyfile = QLineEdit()
        self.keyfile.setPlaceholderText("özel anahtar (.pem / .key)")
        pick_cert = QPushButton("…")
        pick_cert.clicked.connect(lambda: self._pick(self.cert, "Sertifika seç"))
        pick_key = QPushButton("…")
        pick_key.clicked.connect(lambda: self._pick(self.keyfile, "Özel anahtar seç"))
        make = QPushButton("Kendinden imzalı oluştur")
        make.setToolTip("Deneme / kapalı ağ için: tarayıcı ve programlar 'güvenilmeyen sertifika' uyarısı verir")
        make.clicked.connect(self.make_cert)
        cert_row, key_file_row = QHBoxLayout(), QHBoxLayout()
        cert_row.addWidget(self.cert, 1)
        cert_row.addWidget(pick_cert)
        cert_row.addWidget(make)
        key_file_row.addWidget(self.keyfile, 1)
        key_file_row.addWidget(pick_key)
        form.addRow("", self.enabled)
        form.addRow("Port:", self.port)
        form.addRow("Kim erişebilir:", self.host)
        form.addRow("", self.require)
        form.addRow("API anahtarı:", key_row)
        form.addRow("Hız sınırı:", self.rate)
        form.addRow("", self.https)
        form.addRow("Sertifika:", cert_row)
        form.addRow("Anahtar dosyası:", key_file_row)
        lay.addLayout(form)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Örnek komutlar (curl):"))
        self.examples = QPlainTextEdit()
        self.examples.setReadOnly(True)
        lay.addWidget(self.examples, 1)
        buttons = QHBoxLayout()
        copy_ex = QPushButton("Örnekleri kopyala")
        copy_ex.clicked.connect(lambda: self._copy(self.examples.toPlainText(), "Örnekler kopyalandı"))
        copy_url = QPushButton("Adresi kopyala")
        copy_url.clicked.connect(lambda: self._copy(self._url(), "Adres kopyalandı"))
        self.apply_btn = QPushButton("Uygula")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.apply)
        close = QPushButton("Kapat")
        close.clicked.connect(self.close)
        for b in (copy_ex, copy_url):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self._load()

    # ------------------------------------------------------------------------------------------------------
    def _load(self):
        s = self.settings
        self.enabled.setChecked(s["enabled"])
        self.port.setValue(min(max(s["port"], 1024), 65535))
        self.host.setCurrentIndex(max(0, self.host.findData(s["host"])))
        self.require.setChecked(s["require_key"])
        self.key.setText(s["key"])
        self.key.setEchoMode(QLineEdit.Password)
        self.rate.setValue(s["rate"])
        self.https.setChecked(s["https"])
        self.cert.setText(s["cert"])
        self.keyfile.setText(s["keyfile"])
        self._refresh()

    def _gather(self) -> dict:
        s = dict(self.settings)
        s.update(enabled=self.enabled.isChecked(), port=self.port.value(), host=self.host.currentData(),
                 require_key=self.require.isChecked(), key=self.key.text(), rate=self.rate.value(),
                 https=self.https.isChecked(), cert=self.cert.text().strip(), keyfile=self.keyfile.text().strip())
        return s

    def _url(self) -> str:
        if self.mgr.running:
            return self.mgr.server.url
        s = self._gather()
        return api_server.ApiServer(None, self.mgr.state, s).url

    def _refresh(self):
        s = self._gather()
        url = self._url()
        if self.mgr.running:
            self.status.setText(f"<b>Açık:</b> {url}")
        else:
            self.status.setText("<b>Kapalı.</b> Açmak için 'API açık' seçip Uygula'ya basın.")
        key = s["key"] if s["require_key"] else ""
        self.examples.setPlainText(api_server.curl_examples(url, key or "<anahtar>", s["https"]))

    def _copy(self, text: str, msg: str):
        QGuiApplication.clipboard().setText(text)
        self.status.setText(f"{self.status.text()}  ·  {msg}")

    def _pick(self, field: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(self, title, field.text() or "", "PEM (*.pem *.crt *.key *.cer);;"
                                              "Tüm dosyalar (*)")
        if path:
            field.setText(path)

    def make_cert(self, host: str | None = None):
        """Self-signed certificate in <VisionFlow folder>/https, filled in the fields."""
        from ..ai.config import home
        folder = home() / "https"
        if host is None:
            from ..nodes.web import local_ip
            host = local_ip()
        try:
            api_server.make_self_signed(folder / "sertifika.pem", folder / "anahtar.pem", host)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Sertifika oluşturulamadı", str(exc))
            return False
        self.cert.setText(str(folder / "sertifika.pem"))
        self.keyfile.setText(str(folder / "anahtar.pem"))
        self.https.setChecked(True)
        self.status.setText(f"Sertifika oluşturuldu: {folder}")
        return True

    def regenerate(self, confirm: bool = True):
        if confirm and QMessageBox.question(
                self, "Yeni anahtar", "Yeni bir API anahtarı oluşturulsun mu? Eski anahtarı kullanan programlar "
                "bağlanamaz.") != QMessageBox.Yes:
            return
        self.key.setText(api_server.new_key())
        self.apply()

    def apply(self) -> bool:
        s = api_server.clean_settings(self._gather())
        ok, error = True, ""
        try:
            if s["enabled"]:
                self.mgr.start(s)
                self.win.logp.log(f"Uzaktan kontrol API'si açık: {self.mgr.server.url}")
            else:
                self.mgr.stop()
        except (ValueError, RuntimeError) as exc:
            ok, error = False, str(exc)
            s["enabled"] = False                # do not try again at every start-up
            self.enabled.setChecked(False)
        try:
            api_server.save_settings(s)
            self.settings = s
        except OSError as exc:
            ok, error = False, error or f"Ayarlar kaydedilemedi: {exc}"
        self._refresh()
        if not ok:
            QMessageBox.warning(self, "API", error)
        return ok
