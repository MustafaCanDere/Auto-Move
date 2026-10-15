"""Araçlar → Reçeteler…: product types of the flow (named sets of setting values for chosen blocks)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QListWidget, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..core import recipes as rc
from .scene import NodeItem


class RecipesDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Reçeteler (ürün tipleri)")
        self.resize(820, 480)
        lay = QVBoxLayout(self)
        hint = QLabel("Reçete, seçtiğiniz blokların ayar değerlerinden oluşan bir ürün tipidir (ör. A ürünü: eşik 120, "
                      "tolerans 10-12 mm). Tuvalde blokları seçip 'Seçili blokların değerlerini al' deyin. Akış "
                      "çalışırken reçete değiştirmek için 'Reçete Seç' bloğunu kullanın. Alt akışın içindeki ayarlar "
                      "için o ayarı önce alt akış bloğunda gösterin.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _: self._show_values())
        left.addWidget(self.list)
        row = QHBoxLayout()
        for text, slot in (("Yeni", lambda: self.new_recipe()), ("Yeniden adlandır", lambda: self.rename_recipe()),
                           ("Sil", lambda: self.delete_recipe())):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        left.addLayout(row)
        body.addLayout(left, 1)
        right = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Blok", "Ayar", "Değer"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self._edited)
        right.addWidget(self.table)
        row = QHBoxLayout()
        self.capture_btn = QPushButton("Seçili blokların değerlerini al")
        self.capture_btn.clicked.connect(lambda: self.capture_selected())
        self.remove_btn = QPushButton("Satırları kaldır")
        self.remove_btn.clicked.connect(lambda: self.remove_rows())
        self.apply_btn = QPushButton("▶ Bu reçeteyi uygula")
        self.apply_btn.clicked.connect(lambda: self.apply_current())
        for b in (self.capture_btn, self.remove_btn, self.apply_btn):
            row.addWidget(b)
        right.addLayout(row)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)
        self.status = QLabel(" ")
        self.status.setObjectName("hint")
        lay.addWidget(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self._filling = False
        self._watching = False
        self.reload()

    # data -----------------------------------------------------------------------------------------------------
    @property
    def graph(self):
        return self.win.root

    def current(self) -> dict | None:
        row = self.list.currentRow()
        recipes = self.graph.recipes
        return recipes[row] if 0 <= row < len(recipes) else None

    def _changed(self, msg: str = ""):
        self.win._schedule_snapshot()
        self.win._title()
        self.status.setText(msg or " ")

    def reload(self, select: str | None = None):
        try:
            cur = select or (self.current() or {}).get("name")
        except RuntimeError:          # the dialog was deleted
            return
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.graph.recipes:
            self.list.addItem(r["name"])
        names = [r["name"] for r in self.graph.recipes]
        self.list.setCurrentRow(names.index(cur) if cur in names else (0 if names else -1))
        self.list.blockSignals(False)
        self._show_values()

    def _show_values(self):
        self._filling = True
        try:
            self.table.setRowCount(0)
            r = self.current()
            for b in (self.capture_btn, self.remove_btn, self.apply_btn):
                b.setEnabled(r is not None)
            if r is None:
                return
            for nid, pv in r["values"].items():
                node = self.graph.nodes.get(nid)
                if node is None:
                    continue
                for name, value in pv.items():
                    p = next((p for p in node.params if p.name == name), None)
                    if p is None:
                        continue
                    i = self.table.rowCount()
                    self.table.insertRow(i)
                    a, b, c = QTableWidgetItem(node.title), QTableWidgetItem(p.display), QTableWidgetItem(str(value))
                    for it in (a, b):
                        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    a.setData(Qt.UserRole, (nid, name))
                    self.table.setItem(i, 0, a)
                    self.table.setItem(i, 1, b)
                    self.table.setItem(i, 2, c)
        finally:
            self._filling = False

    def _edited(self, item: QTableWidgetItem):
        """A value typed into the table: checked like a setting (numbers pulled into range)."""
        if self._filling or item.column() != 2:
            return
        r = self.current()
        key = self.table.item(item.row(), 0).data(Qt.UserRole) if self.table.item(item.row(), 0) else None
        if r is None or key is None:
            return
        nid, name = key
        node = self.graph.nodes.get(nid)
        try:
            value = rc.coerce_value(node, name, item.text())
        except (TypeError, ValueError, AttributeError) as exc:
            self.status.setText(f"Geçersiz değer: {exc}")
            self._show_values()
            return
        r["values"].setdefault(nid, {})[name] = value
        self._show_values()
        self._changed(f"'{r['name']}' güncellendi")

    # actions ----------------------------------------------------------------------------------------------------
    def _ask_name(self, title: str, text: str = "") -> str | None:
        name, ok = QInputDialog.getText(self, title, "Reçetenin adı:", text=text)
        return name if ok else None

    def _check_name(self, name, other_than: dict | None = None) -> str | None:
        name = rc.clean_name(name)
        if not name:
            self.status.setText("Reçetenin bir adı olmalı")
            return None
        same = rc.get(self.graph, name)
        if same is not None and same is not other_than:
            self.status.setText(f"'{name}' adlı bir reçete zaten var")
            return None
        if len(self.graph.recipes) >= rc.MAX_RECIPES and other_than is None:
            self.status.setText(f"En fazla {rc.MAX_RECIPES} reçete olabilir")
            return None
        return name

    def new_recipe(self, name: str | None = None) -> dict | None:
        """A new recipe; it starts with the values of the selected blocks when blocks are selected."""
        if name is None:
            name = self._ask_name("Yeni reçete", f"Ürün {len(self.graph.recipes) + 1}")
            if name is None:
                return None
        name = self._check_name(name)
        if name is None:
            return None
        r = {"name": name, "values": rc.capture(self.graph, self._selected_ids()) if not self.win.levels else {}}
        self.graph.recipes.append(r)
        self.reload(name)
        self._changed(f"'{name}' oluşturuldu")
        return r

    def rename_recipe(self, new: str | None = None) -> bool:
        r = self.current()
        if r is None:
            return False
        if new is None:
            new = self._ask_name("Reçeteyi yeniden adlandır", r["name"])
            if new is None:
                return False
        new = self._check_name(new, other_than=r)
        if new is None:
            return False
        old, r["name"] = r["name"], new
        n = rc.rename_selectors(self.graph, old, new)
        self.reload(new)
        self._changed(f"'{old}' → '{new}'" + (f" ({n} 'Reçete Seç' bloğu da güncellendi)" if n else ""))
        if n:
            self.win._values_changed()
        return True

    def delete_recipe(self, confirm: bool = True) -> bool:
        r = self.current()
        if r is None:
            return False
        if confirm and QMessageBox.question(self, "Reçeteyi sil", f"'{r['name']}' reçetesi silinsin mi?") \
                != QMessageBox.Yes:
            return False
        self.graph.recipes.remove(r)
        self.reload()
        self._changed(f"'{r['name']}' silindi (geri almak için Ctrl+Z)")
        return True

    def _selected_ids(self) -> list[str]:
        return [it.node.id for it in self.win.scene.selectedItems() if isinstance(it, NodeItem)]

    def capture_selected(self) -> int:
        """Put the current values of the selected blocks into the recipe (replacing what it had for them)."""
        r = self.current()
        if r is None:
            self.status.setText("Önce bir reçete oluşturun veya seçin")
            return 0
        if self.win.levels:
            self.status.setText("Reçeteler ana akışın bloklarını tutar: ana akışa dönün (alt akışın içindeki bir ayar "
                                "için onu alt akış bloğunda gösterin)")
            return 0
        values = rc.capture(self.graph, self._selected_ids())
        if not values:
            self.status.setText("Tuvalde ayarı olan blokları seçin ('Reçete Seç' ve not blokları alınmaz)")
            return 0
        for nid, pv in values.items():
            r["values"][nid] = pv
        self._show_values()
        n = sum(len(pv) for pv in values.values())
        self._changed(f"{len(values)} bloğun {n} ayarı '{r['name']}' reçetesine alındı")
        return n

    def remove_rows(self, rows: list[int] | None = None) -> int:
        r = self.current()
        if r is None:
            return 0
        if rows is None:
            rows = sorted({i.row() for i in self.table.selectedIndexes()})
        removed = 0
        for row in rows:
            it = self.table.item(row, 0)
            if it is None:
                continue
            nid, name = it.data(Qt.UserRole)
            if name in r["values"].get(nid, {}):
                del r["values"][nid][name]
                removed += 1
            if nid in r["values"] and not r["values"][nid]:
                del r["values"][nid]
        self._show_values()
        if removed:
            self._changed(f"{removed} satır kaldırıldı")
        return removed

    def apply_current(self) -> list[str] | None:
        r = self.current()
        if r is None:
            return None
        errors = self.win.apply_recipe(r["name"])
        self.status.setText(f"'{r['name']}' uygulandı" + (f" ({len(errors)} sorun, günlüğe bakın)" if errors else ""))
        return errors

    # follow the flow only while shown: closing (Esc too) stops it, showing again starts it and reloads
    def showEvent(self, e):
        if not self._watching:
            self.win.scene.structure_changed.connect(self.reload)     # undo / redo / another flow
            self._watching = True
            self.reload()
        super().showEvent(e)

    def hideEvent(self, e):
        self._unwatch()
        super().hideEvent(e)

    def _unwatch(self):
        if self._watching:
            self._watching = False
            try:
                self.win.scene.structure_changed.disconnect(self.reload)
            except (RuntimeError, TypeError):
                pass
