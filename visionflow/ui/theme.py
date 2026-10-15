"""Modern design system, theme stylesheets, and category colors for VisionFlow Studio."""

CATEGORY_COLORS = {
    "1 · Giriş": "#10b981",                     # Emerald
    "2 · Görüntü İşleme": "#0ea5e9",            # Sky Blue / Cyan
    "2b · Kalite Kontrol ve Ölçüm": "#06b6d4",  # Cyan
    "2c · 3B ve Derinlik": "#14b8a6",           # Teal
    "3 · Yapay Zeka": "#8b5cf6",                # Violet / Neon Purple
    "3b · Öğrenme (Kendi Örneklerinle)": "#6366f1", # Indigo
    "4 · Mantık": "#f59e0b",                    # Amber
    "4b · Alt Akışlar": "#ec4899",              # Pink / Rose
    "5 · Çıkış ve Otomasyon": "#f43f5e",        # Rose / Crimson
    "6 · Eklentiler": "#84cc16",                # Lime Green
}
DEFAULT_CATEGORY_COLOR = "#64748b"

# Modern Obsidian (Default Sleek Dark Mode)
STYLESHEET = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", "Inter", Roboto, sans-serif;
    outline: none;
}
QWidget {
    background: #0f1115;
    color: #e2e8f0;
    font-size: 10pt;
}
QMainWindow::separator {
    background: #1e232d;
    width: 2px;
    height: 2px;
}
QDockWidget {
    titlebar-close-icon: none;
    border: none;
}
QDockWidget::title {
    background: #14171f;
    padding: 7px 10px;
    font-weight: 600;
    color: #94a3b8;
    border-bottom: 1px solid #1e232d;
    letter-spacing: 0.5px;
}
QToolBar {
    background: #13151b;
    border-bottom: 1px solid #1e232d;
    spacing: 6px;
    padding: 4px 8px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 9px;
    font-weight: 500;
    color: #cbd5e1;
}
QToolButton:hover {
    background: #1e232d;
    border-color: #2b3342;
    color: #38bdf8;
}
QToolButton:pressed, QToolButton:checked {
    background: #0284c7;
    color: #ffffff;
    border-color: #38bdf8;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTreeWidget {
    background: #161922;
    border: 1px solid #262c3b;
    border-radius: 6px;
    padding: 5px 8px;
    color: #f1f5f9;
    selection-background-color: #0284c7;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #0ea5e9;
    background: #1a1e29;
}
QTreeWidget {
    border: 1px solid #1e232d;
    border-radius: 6px;
    padding: 4px;
}
QTreeWidget::item {
    padding: 5px 6px;
    border-radius: 4px;
}
QTreeWidget::item:hover {
    background: #1a1f2b;
    color: #38bdf8;
}
QTreeWidget::item:selected {
    background: #0284c7;
    color: #ffffff;
}
QPushButton {
    background: #1e232d;
    border: 1px solid #2b3342;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
    color: #f1f5f9;
}
QPushButton:hover {
    background: #28303f;
    border-color: #38bdf8;
    color: #38bdf8;
}
QPushButton:pressed {
    background: #0284c7;
    color: #ffffff;
}
QPushButton:checked {
    background: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
    font-weight: 600;
}
QStatusBar {
    background: #0b0d11;
    border-top: 1px solid #1a1e27;
    color: #94a3b8;
    font-size: 9pt;
}
QMenuBar {
    background: #111318;
    border-bottom: 1px solid #1a1e27;
    padding: 3px 6px;
}
QMenuBar::item {
    padding: 4px 9px;
    border-radius: 4px;
    background: transparent;
    color: #cbd5e1;
}
QMenuBar::item:selected {
    background: #1e232d;
    color: #38bdf8;
}
QMenu {
    background: #14171f;
    border: 1px solid #262c3b;
    border-radius: 8px;
    padding: 5px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 5px;
    color: #e2e8f0;
}
QMenu::item:selected {
    background: #0284c7;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #1e232d;
    margin: 4px 6px;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #1e232d;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #0f1115;
}
QSlider::handle:horizontal:hover {
    background: #7dd3fc;
}
QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #262c3b;
    min-height: 24px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #3b445a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 7px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #262c3b;
    min-width: 24px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal:hover {
    background: #3b445a;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QLabel#hint {
    color: #64748b;
    font-size: 9pt;
}
"""

LIGHT_STYLESHEET = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", "Inter", Roboto, sans-serif;
    outline: none;
}
QWidget { background: #f8fafc; color: #1e293b; font-size: 10pt; }
QMainWindow::separator { background: #e2e8f0; width: 2px; height: 2px; }
QDockWidget::title { background: #f1f5f9; padding: 7px 10px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }
QToolBar { background: #ffffff; border-bottom: 1px solid #e2e8f0; spacing: 6px; padding: 4px 8px; }
QToolButton { padding: 5px 9px; border-radius: 6px; border: 1px solid transparent; color: #334155; }
QToolButton:hover { background: #f1f5f9; border-color: #cbd5e1; color: #0284c7; }
QToolButton:checked { background: #e0f2fe; color: #0284c7; border-color: #38bdf8; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTreeWidget {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 5px 8px;
    selection-background-color: #e0f2fe; selection-color: #0369a1;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #0284c7; }
QTreeWidget::item { padding: 5px 6px; border-radius: 4px; }
QTreeWidget::item:hover { background: #f1f5f9; }
QTreeWidget::item:selected { background: #e0f2fe; color: #0369a1; }
QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 14px; font-weight: 500; }
QPushButton:hover { background: #f1f5f9; border-color: #0284c7; color: #0284c7; }
QStatusBar { background: #f1f5f9; border-top: 1px solid #e2e8f0; color: #64748b; }
QMenuBar { background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 3px 6px; }
QMenuBar::item:selected { background: #f1f5f9; color: #0284c7; }
QMenu { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 5px; }
QMenu::item:selected { background: #e0f2fe; color: #0369a1; border-radius: 5px; }
QSlider::groove:horizontal { height: 5px; background: #e2e8f0; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0284c7; border-radius: 2px; }
QSlider::handle:horizontal { background: #0284c7; width: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#hint { color: #64748b; }
"""

MIDNIGHT_STYLESHEET = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", "Inter", Roboto, sans-serif;
    outline: none;
}
QWidget { background: #070b14; color: #e2e8f0; font-size: 10pt; }
QMainWindow::separator { background: #131d31; width: 2px; height: 2px; }
QDockWidget::title { background: #0d1527; padding: 7px 10px; font-weight: 600; color: #38bdf8; border-bottom: 1px solid #17243c; }
QToolBar { background: #0a1120; border-bottom: 1px solid #17243c; spacing: 6px; padding: 4px 8px; }
QToolButton { padding: 5px 9px; border-radius: 6px; border: 1px solid transparent; color: #94a3b8; }
QToolButton:hover { background: #131d31; border-color: #1e293b; color: #38bdf8; }
QToolButton:checked { background: #0284c7; color: #ffffff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTreeWidget {
    background: #0d1527; border: 1px solid #1a2740; border-radius: 6px; padding: 5px 8px;
    selection-background-color: #0284c7;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #38bdf8; }
QTreeWidget::item { padding: 5px 6px; border-radius: 4px; }
QTreeWidget::item:hover { background: #131d31; color: #38bdf8; }
QPushButton { background: #0f1a2f; border: 1px solid #1d2d4a; border-radius: 6px; padding: 6px 14px; color: #f1f5f9; }
QPushButton:hover { background: #0284c7; border-color: #38bdf8; }
QStatusBar { background: #060910; border-top: 1px solid #131d31; color: #64748b; }
QMenuBar { background: #090e1a; border-bottom: 1px solid #131d31; }
QMenuBar::item:selected { background: #131d31; color: #38bdf8; }
QMenu { background: #0d1527; border: 1px solid #1a2740; border-radius: 8px; padding: 5px; }
QMenu::item:selected { background: #0284c7; color: #ffffff; border-radius: 5px; }
QSlider::groove:horizontal { height: 5px; background: #17243c; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0284c7; border-radius: 2px; }
QSlider::handle:horizontal { background: #38bdf8; width: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#hint { color: #475569; }
"""

EMERALD_STYLESHEET = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", "Inter", Roboto, sans-serif;
    outline: none;
}
QWidget { background: #050e09; color: #e2f4ea; font-size: 10pt; }
QMainWindow::separator { background: #0e2418; width: 2px; height: 2px; }
QDockWidget::title { background: #0a1b12; padding: 7px 10px; font-weight: 600; color: #34d399; border-bottom: 1px solid #123020; }
QToolBar { background: #08170f; border-bottom: 1px solid #123020; spacing: 6px; padding: 4px 8px; }
QToolButton { padding: 5px 9px; border-radius: 6px; border: 1px solid transparent; color: #86efac; }
QToolButton:hover { background: #0e2418; border-color: #17422b; color: #34d399; }
QToolButton:checked { background: #059669; color: #ffffff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTreeWidget {
    background: #0a1b12; border: 1px solid #143523; border-radius: 6px; padding: 5px 8px;
    selection-background-color: #059669;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #34d399; }
QTreeWidget::item { padding: 5px 6px; border-radius: 4px; }
QTreeWidget::item:hover { background: #0e2418; }
QPushButton { background: #0c2015; border: 1px solid #17422b; border-radius: 6px; padding: 6px 14px; color: #e2f4ea; }
QPushButton:hover { background: #059669; border-color: #34d399; }
QStatusBar { background: #040a06; border-top: 1px solid #0e2418; color: #52796f; }
QMenuBar { background: #06130c; border-bottom: 1px solid #0e2418; }
QMenuBar::item:selected { background: #0e2418; color: #34d399; }
QMenu { background: #0a1b12; border: 1px solid #143523; border-radius: 8px; padding: 5px; }
QMenu::item:selected { background: #059669; color: #ffffff; border-radius: 5px; }
QSlider::groove:horizontal { height: 5px; background: #123020; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #059669; border-radius: 2px; }
QSlider::handle:horizontal { background: #34d399; width: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#hint { color: #52796f; }
"""

CANVAS_THEMES = {
    "dark": {
        "bg": "#101217", "grid": "#181b22", "grid_major": "#212530",
        "body": "#191c24", "border": "#272d3a", "selected": "#00d2ff", "port_text": "#cbd5e1",
        "status": "#94a3b8", "thumb_bg": "#0f1115", "socket_pen": "#101217", "edge_selected": "#00d2ff",
        "frame_title": "#f1f5f9", "frame_text": "#94a3b8", "subflow_back": "#2d1b28",
        "minimap_bg": "#12141a", "minimap_border": "#2b3342",
    },
    "midnight": {
        "bg": "#060a12", "grid": "#0d1525", "grid_major": "#15223a",
        "body": "#0c1424", "border": "#192846", "selected": "#38bdf8", "port_text": "#94a3b8",
        "status": "#64748b", "thumb_bg": "#05080f", "socket_pen": "#060a12", "edge_selected": "#38bdf8",
        "frame_title": "#f8fafc", "frame_text": "#94a3b8", "subflow_back": "#131f33",
        "minimap_bg": "#060a12", "minimap_border": "#192846",
    },
    "emerald": {
        "bg": "#040b07", "grid": "#0a1910", "grid_major": "#112b1c",
        "body": "#0a1910", "border": "#143724", "selected": "#34d399", "port_text": "#86efac",
        "status": "#52796f", "thumb_bg": "#030805", "socket_pen": "#040b07", "edge_selected": "#34d399",
        "frame_title": "#ecfdf5", "frame_text": "#a7f3d0", "subflow_back": "#112b1c",
        "minimap_bg": "#040b07", "minimap_border": "#143724",
    },
    "light": {
        "bg": "#f8fafc", "grid": "#edf2f7", "grid_major": "#e2e8f0",
        "body": "#ffffff", "border": "#cbd5e1", "selected": "#0284c7", "port_text": "#334155",
        "status": "#64748b", "thumb_bg": "#f1f5f9", "socket_pen": "#f8fafc", "edge_selected": "#0284c7",
        "frame_title": "#0f172a", "frame_text": "#475569", "subflow_back": "#fce7f3",
        "minimap_bg": "#ffffff", "minimap_border": "#cbd5e1",
    },
}

THEMES = {"dark": "Obsidian Koyu (Varsayılan)", "midnight": "Gece Mavisi (Siber)", "emerald": "Zümrüt Yeşil", "light": "Açık Tema"}
CANVAS = dict(CANVAS_THEMES["dark"])
current_theme = "dark"


def stylesheet(name: str) -> str:
    if name == "light":
        return LIGHT_STYLESHEET
    if name == "midnight":
        return MIDNIGHT_STYLESHEET
    if name == "emerald":
        return EMERALD_STYLESHEET
    return STYLESHEET


def set_theme(name: str, app=None) -> str:
    """Switch canvas colours and application stylesheet. Unknown names default to dark."""
    global current_theme
    name = name if name in CANVAS_THEMES else "dark"
    current_theme = name
    CANVAS.clear()
    CANVAS.update(CANVAS_THEMES[name])
    if app is not None:
        app.setStyleSheet(stylesheet(name))
    return name
