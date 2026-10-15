# PyInstaller tarifi:  pyinstaller visionflow.spec
# Çıktı: dist/VisionFlow/ (Windows'ta VisionFlow.exe, Linux'ta VisionFlow)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = collect_submodules("visionflow")
# AI Merkezi paketleri içeride (fonksiyon içinde) import edildiği için açıkça eklenir
for pkg in ("anthropic", "edge_tts", "imageio_ffmpeg", "pypdf", "docx", "openpyxl",
            "serial", "pycomm3", "pysoem", "scapy", "pypylon", "harvesters", "genicam",
            "paho", "pyttsx3", "asyncua", "pytesseract", "PIL"):          # PLC, endüstriyel kamera, OCR, TTS ve MQTT
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        pass
datas = [("examples", "examples"), ("visionflow/train.py", "."), ("visionflow/resources", "visionflow/resources")] + collect_data_files("cv2")
for pkg in ("imageio_ffmpeg", "docx", "anthropic"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

a = Analysis(
    ["run_visionflow.py"],
    pathex=["."],
    hiddenimports=hidden,
    datas=datas,
    excludes=["torch", "ultralytics", "tkinter", "matplotlib", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VisionFlow", console=False, icon="installer/app.ico")
coll = COLLECT(exe, a.binaries, a.datas, name="VisionFlow")
