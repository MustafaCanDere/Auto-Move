"""AI Merkezi window: chat with your own AI, manage its memory, and set up accounts."""
from __future__ import annotations

import html

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
                               QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..ai import claude
from ..ai.config import AIConfig
from ..ai.memory import DOC_EXT, shared as shared_memory

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
SESSION = "ai-merkezi"


class _Call(QThread):
    ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AICenter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Merkezi")
        self.resize(820, 680)
        self.cfg = AIConfig.load()
        self.jobs: list[_Call] = []
        self.chat_gen = 0                          # bumped by 'Yeni sohbet': late answers of the old chat are dropped
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "Sohbet")
        tabs.addTab(self._memory_tab(), "Hafıza")
        tabs.addTab(self._settings_tab(), "Ayarlar")
        QVBoxLayout(self).addWidget(tabs)
        self.tabs = tabs
        if not self.cfg.key():
            tabs.setCurrentIndex(2)

    @property
    def memory(self):
        return shared_memory(self.cfg.memory_file())

    def _run(self, fn, ok, failed):
        job = _Call(fn, self.parent() or self)     # parent outlives the window if it is closed mid-call
        job.ok.connect(ok)
        job.failed.connect(failed)
        job.finished.connect(lambda: self.jobs.remove(job) if job in self.jobs else None)
        job.finished.connect(job.deleteLater)      # otherwise every question leaves a thread object behind
        self.jobs.append(job)
        job.start()
        return job

    # --- chat ------------------------------------------------------------------------------------------
    def _chat_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        lay.addWidget(self.transcript, 1)
        self.chat_in = QPlainTextEdit()
        self.chat_in.setPlaceholderText("Yapay zekanıza yazın… (Ctrl+Enter ile gönder)")
        self.chat_in.setMaximumHeight(100)
        lay.addWidget(self.chat_in)
        row = QHBoxLayout()
        self.use_mem = QCheckBox("Hafızayı kullan")
        self.use_mem.setChecked(True)
        self.search_web = QCheckBox("İnternette araştır")
        row.addWidget(self.use_mem)
        row.addWidget(self.search_web)
        row.addStretch(1)
        save = QPushButton("Son cevabı hafızaya kaydet")
        save.clicked.connect(self.save_last_answer)
        new = QPushButton("Yeni sohbet")
        new.clicked.connect(self.new_chat)
        self.send_btn = QPushButton("Gönder")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self.send)
        for b in (save, new, self.send_btn):
            row.addWidget(b)
        lay.addLayout(row)
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+Return"), self.chat_in, self.send)
        self.last_answer = ""
        for m in self.memory.history(SESSION, 10):
            self._append(m["role"], m["content"])
        return w

    def _append(self, role: str, text: str):
        who = "Siz" if role == "user" else ("Yapay zeka" if role == "assistant" else role)
        color = "#8ab4f8" if role == "user" else "#81c995" if role == "assistant" else "#f28b82"
        body = html.escape(text).replace("\n", "<br>")
        self.transcript.append(f"<p><b style='color:{color}'>{who}:</b><br>{body}</p>")

    def send(self):
        text = self.chat_in.toPlainText().strip()
        if not text or not self.send_btn.isEnabled():
            return
        self.chat_in.clear()
        self._append("user", text)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Düşünüyor…")
        cfg, use_mem, web = AIConfig.load(), self.use_mem.isChecked(), self.search_web.isChecked()
        mem, gen = self.memory, self.chat_gen

        def work():
            if web:
                report, sources = claude.research(text, cfg=cfg)
                ans = report + ("\n\nKaynaklar:\n" + "\n".join(f"- {s['title']}: {s['url']}" for s in sources)
                                if sources else "")
            else:
                prompt = text
                ctx = mem.context_for(text) if use_mem else ""
                if ctx:
                    prompt = f"{ctx}\n\n---\n{text}"
                ans = claude.ask(prompt, history=mem.history(SESSION), cfg=cfg, max_tokens=4000)
            if gen == self.chat_gen:               # 'Yeni sohbet' meanwhile: do not bring the old talk back
                mem.log(SESSION, "user", text)
                mem.log(SESSION, "assistant", ans)
            return ans
        self._run(work, lambda ans: self._answer(ans, gen), lambda msg: self._chat_failed(msg, gen))

    def _answer(self, ans, gen=None):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Gönder")
        self.refresh_usage()
        if gen is not None and gen != self.chat_gen:
            return
        self.last_answer = ans
        self._append("assistant", ans)

    def _chat_failed(self, msg, gen=None):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Gönder")
        if gen is None or gen == self.chat_gen:
            self._append("hata", msg)

    def save_last_answer(self):
        if self.last_answer:
            self.memory.add_text(self.last_answer, source="sohbetten kaydedilenler")
            self._refresh_sources()
            self._append("bilgi", "Cevap hafızaya kaydedildi.")

    def new_chat(self):
        self.chat_gen += 1
        self.memory.clear_history(SESSION)
        self.transcript.clear()
        self.last_answer = ""

    # --- memory ----------------------------------------------------------------------------------------
    def _memory_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Yapay zekanızın bildikleri. Belgeleriniz bu bilgisayarda kalır; soru sorulunca sadece "
                             "ilgili parçalar Claude'a gönderilir."))
        self.src_tree = QTreeWidget()
        self.src_tree.setHeaderLabels(["Kaynak", "Eklendi", "Parça"])
        self.src_tree.setColumnWidth(0, 420)
        lay.addWidget(self.src_tree, 1)
        row = QHBoxLayout()
        for text, fn in (("Dosya ekle…", self.add_files), ("Klasör ekle…", self.add_folder),
                         ("Not yaz…", self.add_note), ("Seçileni sil", self.remove_selected)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        srow = QHBoxLayout()
        self.mem_q = QLineEdit()
        self.mem_q.setPlaceholderText("Hafızada ara (ör. iade süresi)")
        self.mem_q.returnPressed.connect(self.search_memory)
        srow.addWidget(self.mem_q, 1)
        b = QPushButton("Ara")
        b.clicked.connect(self.search_memory)
        srow.addWidget(b)
        lay.addLayout(srow)
        self.mem_results = QTextBrowser()
        self.mem_results.setMaximumHeight(180)
        lay.addWidget(self.mem_results)
        self._refresh_sources()
        return w

    def _refresh_sources(self):
        self.src_tree.clear()
        for name, added, n in self.memory.sources():
            self.src_tree.addTopLevelItem(QTreeWidgetItem([name, added.replace("T", " "), str(n)]))

    def add_files(self, paths=None):
        if not paths:
            exts = " ".join(f"*{e}" for e in sorted(DOC_EXT))
            paths, _ = QFileDialog.getOpenFileNames(self, "Belge ekle", "", f"Belgeler ({exts})")
        errors = []
        for p in paths or []:
            try:
                self.memory.add_file(p)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{p}: {exc}")
        self._refresh_sources()
        if errors:
            QMessageBox.warning(self, "Bazı dosyalar okunamadı", "\n".join(errors[:10]))

    def add_folder(self, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, "Belge klasörü")
        if folder:
            res = self.memory.add_folder(folder)
            self._refresh_sources()
            bad = [f"{k}: {v}" for k, v in res.items() if not isinstance(v, int)]
            if bad:
                QMessageBox.warning(self, "Bazı dosyalar okunamadı", "\n".join(bad[:10]))

    def add_note(self):
        text, ok = QInputDialog.getMultiLineText(self, "Not", "Yapay zekanın bilmesini istediğiniz bilgi:")
        if ok and text.strip():
            self.memory.add_text(text, source="notlar")
            self._refresh_sources()

    def remove_selected(self):
        for it in self.src_tree.selectedItems():
            self.memory.remove(it.text(0))
        self._refresh_sources()

    def search_memory(self):
        hits = self.memory.search(self.mem_q.text(), 5)
        self.mem_results.setHtml("".join(f"<p><b>{html.escape(s)}</b><br>{html.escape(t)}</p>" for s, t in hits)
                                 or "<i>Bulunamadı</i>")

    # --- settings ----------------------------------------------------------------------------------------
    def _settings_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        c = self.cfg
        self.f_key = QLineEdit(c.api_key)
        self.f_key.setEchoMode(QLineEdit.Password)
        self.f_key.setPlaceholderText("sk-ant-…  (console.anthropic.com → API Keys)")
        self.f_model = QComboBox()
        self.f_model.setEditable(True)
        self.f_model.addItems(MODELS)
        self.f_model.setCurrentText(c.model)
        self.f_persona = QPlainTextEdit(c.persona)
        self.f_persona.setMaximumHeight(90)
        self.f_mail = QLineEdit(c.email_user)
        self.f_pass = QLineEdit(c.email_password)
        self.f_pass.setEchoMode(QLineEdit.Password)
        self.f_pass.setPlaceholderText("Gmail: Google hesabı → Güvenlik → Uygulama şifreleri")
        self.f_imap = QLineEdit(c.imap_host)
        self.f_smtp = QLineEdit(c.smtp_host)
        self.f_port = QSpinBox()
        self.f_port.setRange(1, 65535)
        self.f_port.setValue(c.smtp_port)
        self.f_eleven = QLineEdit(c.elevenlabs_key)
        self.f_eleven.setEchoMode(QLineEdit.Password)
        self.f_eleven.setPlaceholderText("isteğe bağlı (en doğal ses)")
        yt = QHBoxLayout()
        self.f_yt = QLineEdit(c.youtube_client_secret)
        self.f_yt.setPlaceholderText("Google Cloud → OAuth istemcisi (masaüstü) → client_secret.json")
        b = QPushButton("…")
        b.clicked.connect(lambda: self.f_yt.setText(
            QFileDialog.getOpenFileName(self, "client_secret.json", "", "JSON (*.json)")[0] or self.f_yt.text()))
        yt.addWidget(self.f_yt, 1)
        yt.addWidget(b)
        self.f_tg = QLineEdit(c.telegram_token)
        self.f_tg.setEchoMode(QLineEdit.Password)
        self.f_tg.setPlaceholderText("Telegram'da @BotFather → /newbot")
        self.f_tg_allowed = QLineEdit(c.telegram_allowed)
        self.f_tg_allowed.setPlaceholderText("bota yazan herkes için ID günlükte görünür; izin vereceklerinizi ekleyin")
        self.f_limit = QDoubleSpinBox()
        self.f_limit.setRange(0, 10000)
        self.f_limit.setDecimals(2)
        self.f_limit.setPrefix("$ ")
        self.f_limit.setValue(c.daily_limit_usd)
        self.f_limit.setToolTip("0 = sınırsız. Aşılınca o gün yapay zeka blokları durur.")
        self.f_usage = QLabel()
        form.addRow(QLabel("<b>Claude</b>"))
        form.addRow("API anahtarı", self.f_key)
        form.addRow("Model", self.f_model)
        form.addRow("Kişilik / talimat", self.f_persona)
        form.addRow("Günlük harcama sınırı", self.f_limit)
        form.addRow("Bugünkü kullanım", self.f_usage)
        form.addRow(QLabel("<b>E-posta hesabı</b> (E-posta Oku / Cevapla blokları)"))
        form.addRow("E-posta", self.f_mail)
        form.addRow("Şifre", self.f_pass)
        form.addRow("IMAP sunucu", self.f_imap)
        form.addRow("SMTP sunucu", self.f_smtp)
        form.addRow("SMTP port", self.f_port)
        form.addRow(QLabel("<b>Ses ve video</b>"))
        form.addRow("ElevenLabs anahtarı", self.f_eleven)
        form.addRow("YouTube", yt)
        form.addRow(QLabel("<b>Telegram botu</b> (Telegram Mesaj Al / Gönder blokları)"))
        form.addRow("Bot token", self.f_tg)
        form.addRow("İzinli chat ID'ler", self.f_tg_allowed)
        form.addRow("Hafıza dosyası", QLabel(str(c.memory_file())))
        row = QHBoxLayout()
        save = QPushButton("Kaydet")
        save.clicked.connect(self.save_settings)
        test = QPushButton("Claude bağlantısını dene")
        test.clicked.connect(self.test_connection)
        row.addWidget(save)
        row.addWidget(test)
        form.addRow(row)
        self.f_status = QLabel("Ayarlar bu bilgisayarda saklanır: " + str(AIConfig.path()))
        self.f_status.setWordWrap(True)
        form.addRow(self.f_status)
        self.refresh_usage()
        return w

    def refresh_usage(self):
        u = claude.usage_today()
        self.f_usage.setText(f"{u['requests']} istek, {u['input'] + u['output']:,} token, yaklaşık ${u['usd']:.2f}"
                             .replace(",", "."))

    def save_settings(self):
        c = AIConfig.load()
        c.api_key, c.model = self.f_key.text().strip(), self.f_model.currentText().strip() or MODELS[0]
        c.persona = self.f_persona.toPlainText().strip()
        c.email_user, c.email_password = self.f_mail.text().strip(), self.f_pass.text()
        c.imap_host, c.smtp_host, c.smtp_port = self.f_imap.text().strip(), self.f_smtp.text().strip(), self.f_port.value()
        c.elevenlabs_key, c.youtube_client_secret = self.f_eleven.text().strip(), self.f_yt.text().strip()
        c.telegram_token, c.telegram_allowed = self.f_tg.text().strip(), self.f_tg_allowed.text().strip()
        c.daily_limit_usd = self.f_limit.value()
        c.save()
        self.cfg = c
        self.f_status.setText("Kaydedildi.")

    def test_connection(self):
        self.save_settings()
        self.f_status.setText("Deneniyor…")
        self._run(lambda: claude.ask("Sadece 'Bağlantı tamam' yaz.", max_tokens=30, effort="low"),
                  lambda a: self.f_status.setText(f"Claude cevap verdi: {a}"),
                  lambda m: self.f_status.setText(f"Olmadı: {m}"))
