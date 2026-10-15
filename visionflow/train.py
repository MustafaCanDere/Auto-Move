"""Train a YOLO model on a labelled dataset and export it to ONNX for the YOLO blocks.

    python -m visionflow.train --data yolo_veri/data.yaml --model yolo11n.pt --epochs 50

Needs ``pip install ultralytics`` (pulls in PyTorch). The editor's "YOLO Modeli Eğit" window runs this
same script in a separate process so the editor stays responsive.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

BASE_MODELS = {
    "yolo11n.pt": "Nesne tespiti · küçük ve hızlı (önerilen)",
    "yolo11s.pt": "Nesne tespiti · daha doğru, daha yavaş",
    "yolo11n-seg.pt": "Segmentasyon · küçük",
    "yolo11n-cls.pt": "Sınıflandırma · küçük (klasör/sınıf yapısı)",
    "yolo11n.yaml": "Nesne tespiti · sıfırdan (internet gerekmez, çok veri ister)",
}


def build_args(data: str, model: str, epochs: int, imgsz: int, device: str, name: str) -> dict:
    return {"data": data, "epochs": epochs, "imgsz": imgsz, "device": device or None, "name": name,
            "project": str(pathlib.Path("runs").resolve()), "exist_ok": True, "verbose": True, "plots": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLO modeli eğit ve ONNX'e aktar")
    ap.add_argument("--data", required=True, help="data.yaml (tespit) veya sınıf klasörü (sınıflandırma)")
    ap.add_argument("--model", default="yolo11n.pt", help=", ".join(BASE_MODELS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="", help="boş = otomatik, cpu, 0 (ilk GPU)")
    ap.add_argument("--name", default="model", help="çıktı adı (models/<ad>.onnx)")
    ap.add_argument("--out", default="models", help="ONNX'in kopyalanacağı klasör")
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("HATA: ultralytics kurulu değil. Kurmak için:  pip install ultralytics", flush=True)
        return 2

    print(f"Eğitim başlıyor: {args.model} · {args.data} · {args.epochs} tur", flush=True)
    model = YOLO(args.model)
    result = model.train(**build_args(args.data, args.model, args.epochs, args.imgsz, args.device, args.name))
    best = pathlib.Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print("HATA: eğitilmiş ağırlık bulunamadı", flush=True)
        return 1
    print("ONNX'e aktarılıyor…", flush=True)
    onnx_path = pathlib.Path(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.name}.onnx"
    shutil.copy2(onnx_path, target)
    print(f"BİTTİ: {target.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
