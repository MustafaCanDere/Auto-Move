"""Kendini Geliştir window: the program shows what went wrong while it was used, asks Claude to improve its own code,
tests the change on a copy (correcting itself when the tests fail) and applies it only when the user approves."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import evolve
from ..ai.config import AIConfig


class _Job(QObject):
    """Runs fn(step) in a daemon thread (a proposal with its test runs can take minutes, and closing the program
    must not wait for it or crash on a running QThread); results come back to the window through signals."""
    step = Signal(str)
    ok = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.thread: threading.Thread | None = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="kendini-gelistir")
        self.thread.start()

    def _send(self, signal, *args):
        try:
            signal.emit(*args)
        except RuntimeError:                   # the window was closed meanwhile
            pass

    def _run(self):
        try:
            self._send(self.ok, self.fn(lambda text: self._send(self.step, text)))
        except Exception as exc:  # noqa: BLE001
            self._send(self.failed, f"{type(exc).__name__}: {exc}")
        self._send(self.finished)


def _mono(widget):
    f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    widget.setFont(f)
    widget.setReadOnly(True)
    widget.setLineWrapMode(QPlainTextEdit.NoWrap)
    return widget


class EvolveDialog(QDialog):
    def __init__(self, parent=None, ask=None):
        super().__init__(parent)
        self.setWindowTitle("Kendini Geliştir")
        self.resize(900, 700)
        self.ask = ask                       # replaceable for tests; None = the user's own Claude key
        self.job: _Job | None = None
        self.proposal: evolve.Proposal | None = None

        info = QLabel("Program kullanırken yaşadığı hataları kaydeder, kendini inceler ve Claude'dan kendi kodu için "
                      "düzeltme ister. Her öneri önce programın bir kopyasında test edilir; testler geçmezse hatayı "
                      "Claude'a gösterip öneriyi kendisi düzeltir. Siz <b>Uygula</b> demeden hiçbir şey değişmez, "
                      "her değişiklik yedeklenir ve geri alınabilir."
                      + ("<br><i>Paketlenmiş programda sadece plugins/ klasörüne eklenti yazılabilir.</i>"
                         if evolve.frozen() else ""))
        info.setWordWrap(True)
        self.goal = QLineEdit()
        self.goal.setPlaceholderText("İsteğe bağlı hedef: ör. \"Kamera bağlantısı kopunca kendini toparlasın\" "
                                     "(boş bırakılırsa kaydedilen hataları düzeltir)")
        self.btn_check = QPushButton("Kendini incele")
        self.btn_check.clicked.connect(self.examine)
        self.btn_propose = QPushButton("Geliştirme öner")
        self.btn_propose.setDefault(True)
        self.btn_propose.clicked.connect(self.request)
        self.btn_apply = QPushButton("Uygula")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.apply)
        self.btn_discard = QPushButton("Öneriyi sil")
        self.btn_discard.setEnabled(False)
        self.btn_discard.clicked.connect(self.discard)
        self.btn_clear = QPushButton("Hata günlüğünü temizle")
        self.btn_clear.clicked.connect(self.clear_log)
        row = QHBoxLayout()
        for b in (self.btn_check, self.btn_propose, self.btn_apply, self.btn_discard):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.btn_clear)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.report = _mono(QPlainTextEdit())
        self.diff = _mono(QPlainTextEdit())
        self.tests = _mono(QPlainTextEdit())
        self.hist = QTreeWidget()
        self.hist.setHeaderLabels(["Ne zaman", "Değişiklik", "Durum"])
        self.hist.setColumnWidth(0, 150)
        self.hist.setColumnWidth(1, 520)
        self.btn_undo = QPushButton("Seçileni geri al")
        self.btn_undo.clicked.connect(self.undo_selected)
        hist_box = QWidget()
        hl = QVBoxLayout(hist_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.hist)
        hl.addWidget(self.btn_undo)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.report, "Rapor")
        self.tabs.addTab(self.diff, "Önerilen değişiklik")
        self.tabs.addTab(self.tests, "Test sonucu")
        self.tabs.addTab(hist_box, "Geçmiş")

        self.daily = QCheckBox("Her gün açılışta kendini incelesin (hata varsa test edilmiş bir öneri hazırlar, "
                               "uygulamaz)")
        self.daily.setChecked(evolve.settings()["daily"])
        self.daily.toggled.connect(self._save_daily)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.goal)
        lay.addLayout(row)
        lay.addWidget(self.status)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self.daily)

        self.report.setPlainText(evolve.report_text(evolve.self_report(run_tests=False)))
        self.refresh_history()
        waiting = evolve.load_pending()
        if waiting is not None:
            if evolve.check_edits(waiting.edits):          # the files changed since: it can never apply
                evolve.discard_pending()
            else:
                self.show_proposal(waiting)
                self.status.setText("Günlük incelemeden bekleyen bir öneri var. İnceleyip uygulayabilirsiniz"
                                    + (" (önce yeniden test edilecek)." if evolve.needs_retest(waiting) else "."))

    # --- helpers --------------------------------------------------------------------------------------------------
    def _busy(self, on: bool):
        self.running = on
        for b in (self.btn_check, self.btn_propose, self.btn_clear, self.btn_undo):
            b.setEnabled(not on)
        self.btn_apply.setEnabled(not on and bool(self.proposal and self.proposal.edits and self.proposal.tests_ok))
        self.btn_discard.setEnabled(not on and self.proposal is not None)

    def _start(self, fn, ok):
        if getattr(self, "running", False):
            return
        self._busy(True)
        job = _Job(fn, self)
        job.step.connect(self.status.setText)
        job.ok.connect(ok)
        job.failed.connect(self._failed)
        job.finished.connect(lambda: self._busy(False))
        self.job = job
        job.start()

    def _failed(self, msg: str):
        self.status.setText(f"Olmadı: {msg}")

    def _save_daily(self, on: bool):
        s = evolve.settings()
        s["daily"] = bool(on)
        evolve.save_settings(s)

    # --- actions --------------------------------------------------------------------------------------------------
    def examine(self):
        def work(step):
            step("Kendini inceliyor: hata günlüğü ve kendi testleri (birkaç dakika sürebilir)…")
            return evolve.self_report()

        def done(rep):
            self.report.setPlainText(evolve.report_text(rep))
            self.tabs.setCurrentWidget(self.report)
            self.status.setText("İnceleme bitti.")
        self._start(work, done)

    def request(self):
        if self.ask is None and not AIConfig.load().key():
            QMessageBox.information(self, "Kendini Geliştir", "Önce AI Merkezi → Ayarlar'dan Claude API anahtarınızı "
                                                              "girin. Harcama sınırınız burada da geçerlidir.")
            return
        goal = self.goal.text()
        self.proposal = None
        self.diff.clear()
        self.tests.clear()

        def work(step):
            return evolve.propose(goal, on_step=step, ask=self.ask)
        self._start(work, self.show_proposal)

    def show_proposal(self, prop: evolve.Proposal):
        self.proposal = prop
        self.diff.setPlainText(prop.summary + ("\n\n" + prop.diff() if prop.edits else ""))
        self.tests.setPlainText(prop.test_output)
        if not prop.edits:
            self.status.setText(f"Değişiklik önerilmedi: {prop.summary}")
        elif prop.tests_ok:
            self.status.setText(f"Öneri hazır ({prop.rounds}. turda testleri geçti). İnceleyip Uygula'ya basın.")
        else:
            self.status.setText(f"{prop.rounds} turda testleri geçen bir değişiklik bulunamadı; uygulanamaz.")
        self.tabs.setCurrentWidget(self.diff)
        self.btn_apply.setEnabled(bool(prop.edits and prop.tests_ok))
        self.btn_discard.setEnabled(True)

    def apply(self):
        prop = self.proposal
        if not prop or not prop.edits or not prop.tests_ok or getattr(self, "running", False):
            return
        if evolve.needs_retest(prop):              # tested on code that has changed since: test it again first
            def work(step):
                step("Program bu öneriden sonra değişti; öneri yeniden test ediliyor…")
                return evolve.retest(prop)

            def done(p):
                self.show_proposal(p)
                self._busy(False)
                if p.tests_ok:
                    self.apply()
            self._start(work, done)
            return
        files = "\n".join(f"• {e.path}" for e in prop.edits)
        if QMessageBox.question(self, "Değişikliği uygula",
                                f"{prop.summary}\n\nDeğişecek dosyalar:\n{files}\n\nEski hâlleri yedeklenir, "
                                "Geçmiş sekmesinden geri alabilirsiniz. Uygulansın mı?") != QMessageBox.Yes:
            return
        try:
            evolve.apply(prop)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Uygulanamadı", str(exc))
            return
        self.proposal = None
        self.btn_apply.setEnabled(False)
        self.btn_discard.setEnabled(False)
        self.refresh_history()
        self.status.setText("Uygulandı. Değişikliğin etkili olması için programı yeniden başlatın.")

    def refresh_history(self):
        self.hist.clear()
        for h in evolve.history():
            state = "geri alındı" if h.get("undone") else "uygulandı"
            item = QTreeWidgetItem([h.get("applied", ""), h.get("summary", "")[:200], state])
            item.setData(0, 256, h.get("id"))                     # Qt.UserRole
            item.setToolTip(1, "\n".join(e["path"] for e in h.get("edits", [])))
            self.hist.addTopLevelItem(item)

    def undo_selected(self):
        item = self.hist.currentItem()
        if item is None:
            return
        try:
            evolve.undo(item.data(0, 256))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Geri alınamadı", str(exc))
            return
        self.refresh_history()
        self.status.setText("Geri alındı. Etkili olması için programı yeniden başlatın.")

    def discard(self):
        if self.proposal is None or getattr(self, "running", False):
            return
        waiting = evolve.load_pending()
        if waiting and waiting.id == self.proposal.id:
            evolve.discard_pending()
        self.proposal = None
        self.diff.clear()
        self.tests.clear()
        self.btn_apply.setEnabled(False)
        self.btn_discard.setEnabled(False)
        self.status.setText("Öneri silindi.")

    def clear_log(self):
        evolve.error_log().clear()
        self.report.setPlainText(evolve.report_text(evolve.self_report(run_tests=False)))
