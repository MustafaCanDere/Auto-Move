"""'Yapay Zeka ile Akış Kur' window: type what you want, Claude builds the flow."""
from __future__ import annotations

import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout)

from .. import assistant
from ..ai.config import AIConfig
from ..core import Graph


class _Job(QThread):
    done = Signal(object)     # Proposal
    failed = Signal(str)

    def __init__(self, request, current, api_key, model, client=None):
        super().__init__()
        self.args = (request, current, api_key, model)
        self.client = client

    def run(self):
        request, current, api_key, model = self.args
        try:
            self.done.emit(assistant.propose(request, current, client=self.client, model=model, api_key=api_key))
        except assistant.AssistantError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Beklenmeyen hata: {exc}")


class AssistantDialog(QDialog):
    """Emits ``apply_graph(graph, modified)`` when the user accepts a proposal; ``modified`` is True when the
    current flow was edited rather than replaced by a new one."""
    apply_graph = Signal(object, bool)

    def __init__(self, parent=None, current: Graph | None = None, client=None):
        super().__init__(parent)
        self.setWindowTitle("Yapay Zeka ile Akış Kur")
        self.resize(640, 560)
        self.current = current
        self.client = client            # injected in tests
        self.proposal = None
        self.job: _Job | None = None
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("Ne yapmak istiyorsunuz? Kendi cümlelerinizle yazın:"))
        self.request = QPlainTextEdit()
        self.request.setPlaceholderText(assistant.EXAMPLE_REQUESTS[0])
        self.request.setMaximumHeight(110)
        lay.addWidget(self.request)
        ex = QComboBox()
        ex.addItem("Örnek istekler…")
        ex.addItems(assistant.EXAMPLE_REQUESTS)
        ex.activated.connect(lambda i: i and self.request.setPlainText(ex.itemText(i)))
        lay.addWidget(ex)

        self.modify = QCheckBox("Mevcut akışı değiştir (işaretli değilse yeni akış kurulur)")
        self.modify.setEnabled(bool(current and current.nodes))
        lay.addWidget(self.modify)

        row = QHBoxLayout()
        row.addWidget(QLabel("API anahtarı:"))
        self.key = QLineEdit(AIConfig.load().api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("ANTHROPIC_API_KEY ortam değişkeni varsa boş bırakın")
        row.addWidget(self.key, 1)
        self.remember = QCheckBox("Kaydet")
        self.remember.setChecked(bool(self.key.text()))
        row.addWidget(self.remember)
        lay.addLayout(row)
        hint = QLabel('Anahtar: <a href="https://console.anthropic.com/settings/keys">console.anthropic.com</a> '
                      "(ücretli; bir akış kurmak birkaç kuruş tutar). Akış ve blok listesi Anthropic'e gönderilir, "
                      "kamera görüntüsü gönderilmez.")
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa0a6;")
        lay.addWidget(hint)

        self.go = QPushButton("✨  Akışı Kur")
        self.go.clicked.connect(self.generate)
        lay.addWidget(self.go)

        self.result = QPlainTextEdit(readOnly=True)
        self.result.setPlaceholderText("Yapay zekanın açıklaması burada görünür.")
        lay.addWidget(self.result, 1)

        self.buttons = QDialogButtonBox()
        self.apply_btn = self.buttons.addButton("Tuvale Uygula", QDialogButtonBox.AcceptRole)
        self.apply_btn.setEnabled(False)
        self.buttons.addButton(QDialogButtonBox.Close)
        self.buttons.accepted.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

    def _key(self) -> str | None:
        return self.key.text().strip() or None

    def generate(self):
        if not self.request.toPlainText().strip():
            self.result.setPlainText("Önce ne yapmak istediğinizi yazın.")
            return
        if self.client is None and not self._key() and not os.environ.get("ANTHROPIC_API_KEY"):
            self.result.setPlainText("API anahtarı gerekli. Yukarıdaki kutuya yapıştırın.")
            return
        if self.remember.isChecked() and self._key():
            cfg = AIConfig.load()
            if cfg.api_key != self._key():
                cfg.api_key = self._key()
                cfg.save()
        self.go.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.result.setPlainText("Yapay zeka akışı tasarlıyor…")
        current = self.current if self.modify.isChecked() else None
        self.job = _Job(self.request.toPlainText(), current, self._key(), assistant.MODEL, self.client)
        self.job.setParent(self.parent() or self)   # outlive the dialog if it is closed while waiting
        self.job.finished.connect(self.job.deleteLater)
        self.job.done.connect(self._done)
        self.job.failed.connect(self._failed)
        self.job.start()

    def _done(self, proposal):
        self.proposal = proposal
        names = ", ".join(n.title for n in proposal.graph.nodes.values())
        text = f"{proposal.explanation}\n\nBloklar ({len(proposal.graph.nodes)}): {names}"
        if proposal.warnings:
            text += "\n\nDüzeltilemeyen sorunlar (bu kısımlar atlandı):\n- " + "\n- ".join(proposal.warnings)
        self.result.setPlainText(text)
        self.apply_btn.setEnabled(True)
        self.go.setEnabled(True)

    def _failed(self, msg):
        self.result.setPlainText(f"Olmadı: {msg}")
        self.go.setEnabled(True)

    def apply(self):
        if self.proposal:
            self.apply_graph.emit(self.proposal.graph, self.modify.isChecked())
            self.accept()

    def done(self, r):
        if self.job is not None:
            try:
                self.job.done.disconnect(self._done)
                self.job.failed.disconnect(self._failed)
            except (RuntimeError, TypeError):
                pass
        super().done(r)
