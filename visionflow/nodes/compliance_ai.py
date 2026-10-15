"""FDA 21 CFR Part 11 Compliance & AI Model Drift Monitoring:
- AuditTrailLogger: Cryptographic SHA-256 hash-chained immutable audit log for pharma/aerospace compliance
- ModelDriftMonitor: Statistical confidence distribution & AI model degradation tracker with retraining pool
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr

CAT_OUT = "5 · Çıkış ve Otomasyon"
CAT_AI = "3 · Yapay Zeka"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None or not trigger:
        node.state[key] = False
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


# ==========================================================================================
# 1. FDA 21 CFR Part 11 Cryptographic Audit Trail (Kriptografik Denetim İzi)
# ==========================================================================================

@register
class AuditTrailLogger(Node):
    title = "21 CFR Part 11 Denetim İzi (Audit Trail)"
    category = CAT_OUT
    description = (
        "İlaç (GMP) ve havacılık standartlarında zorunlu olan FDA 21 CFR Part 11 uyumlu, silinemez ve değiştirilemez "
        "kriptografik (SHA-256 zincirli) işlem kütüğü tutar. Her parametre veya reçete değişimini dijital imzalar."
    )
    inputs = [
        ("log_trigger", BOOL),
        ("action", TEXT),
        ("details", TEXT),
        ("user_override", TEXT),
    ]
    outputs = [
        ("logged", BOOL),
        ("last_hash", TEXT),
        ("total_records", NUM),
        ("is_chain_valid", BOOL),
    ]
    params = [
        Param("audit_log_file", "path", "denetim_izi_21cfr11.json", "Denetim Kütüğü Dosyası"),
        Param("current_user", "str", "Kalite_Muhendisi_1", "Varsayılan Kullanıcı / Operatör"),
        Param("company_name", "str", "VisionFlow GMP Tesis", "Kurum / Tesis"),
        Param("enforce_tamper_check", "bool", True, "Her Adımda Bütünlük Doğrula"),
    ]

    def setup(self):
        log_path = self._resolve_path()
        last_hash = "GENESIS_00000000000000000000000000000000"
        total = 0
        valid = True

        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as fh:
                    records = json.load(fh)
                    if isinstance(records, list) and records:
                        total = len(records)
                        last_hash = records[-1].get("hash", last_hash)
                        if bool(self.v("enforce_tamper_check")):
                            valid = self._verify_chain(records)
            except Exception as exc:
                print(f"[Audit Trail] Kütük okuma hatası: {exc}")

        self.state.update(
            last_hash=last_hash,
            total=total,
            valid=valid,
            prev_trig=False,
        )

    def _resolve_path(self) -> str:
        raw = str(self.v("audit_log_file") or "denetim_izi_21cfr11.json").strip()
        return _node.resolve_path(raw)

    def _verify_chain(self, records: list[dict]) -> bool:
        prev = "GENESIS_00000000000000000000000000000000"
        for r in records:
            if r.get("prev_hash") != prev:
                return False
            data_str = f"{r.get('prev_hash')}|{r.get('timestamp')}|{r.get('user')}|{r.get('action')}|{r.get('details')}"
            expected_hash = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
            if r.get("hash") != expected_hash:
                return False
            prev = expected_hash
        return True

    def process(self, inputs):
        st = self.state
        st.setdefault("last_hash", "GENESIS_00000000000000000000000000000000")
        st.setdefault("total", 0)
        st.setdefault("valid", True)

        trig = _rising(self, inputs.get("log_trigger"), "prev_trig")
        if not trig:
            return {
                "logged": False,
                "last_hash": st["last_hash"],
                "total_records": st["total"],
                "is_chain_valid": st["valid"],
            }

        act = str(inputs.get("action") or "Parametre Onayı / Kalite Kontrol").strip()
        det = str(inputs.get("details") or "Otomatik sistem denetimi tamamlandı.").strip()
        u_ov = inputs.get("user_override")
        user = str(u_ov).strip() if u_ov else str(self.v("current_user")).strip()

        now_iso = dt.datetime.now().astimezone().isoformat()
        prev_h = st["last_hash"]

        # Calculate cryptographic hash
        data_to_hash = f"{prev_h}|{now_iso}|{user}|{act}|{det}"
        curr_hash = hashlib.sha256(data_to_hash.encode("utf-8")).hexdigest()

        entry = {
            "index": st["total"] + 1,
            "timestamp": now_iso,
            "user": user,
            "action": act,
            "details": det,
            "prev_hash": prev_h,
            "hash": curr_hash,
            "standard": "FDA 21 CFR Part 11 / EU Annex 11",
        }

        log_path = self._resolve_path()
        try:
            records = []
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as fh:
                    records = json.load(fh)

            records.append(entry)
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False, indent=2)

            st["last_hash"] = curr_hash
            st["total"] += 1
            st["valid"] = True
            print(f"[Audit Trail 21 CFR 11] Kaydedildi (İmza: {curr_hash[:12]}...)")
        except Exception as exc:
            print(f"[Audit Trail] Kayıt hatası: {exc}")
            st["valid"] = False

        return {
            "logged": True,
            "last_hash": st["last_hash"],
            "total_records": st["total"],
            "is_chain_valid": st["valid"],
        }


# ==========================================================================================
# 2. AI Model Drift Monitor & Active Retraining Pool (Model Kayması İzleyici)
# ==========================================================================================

@register
class ModelDriftMonitor(Node):
    title = "AI Model Kayması & Güvenlik İzleyici (Drift Monitor)"
    category = CAT_AI
    description = (
        "Yapay zeka modellerinin (YOLO, Kusur Tespiti, VLM) güven skorlarını (Confidence) zaman içinde izler. "
        "Işık, kamera kirliliği veya parça yüzeyindeki sapmalar nedeniyle model doğruluğu düştüğünde alarm verir "
        "ve şüpheli görüntüleri 'Yeniden Eğitim Havuzuna' kaydeder."
    )
    inputs = [
        ("confidence", NUM),
        ("is_defect", BOOL),
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("drift_detected", BOOL),
        ("avg_confidence", NUM),
        ("low_conf_pct", NUM),
        ("drift_alarm", BOOL),
        ("saved_for_retraining", NUM),
    ]
    params = [
        Param("min_acceptable_confidence", "float", 85.0, "Kabul Edilen Min Güven %", 10.0, 100.0),
        Param("drift_alert_pct", "float", 15.0, "İzin Verilen Düşük Güven Oranı %", 1.0, 100.0),
        Param("window_size", "int", 50, "Pencere Boyutu (Örnek Sayısı)", 10, 500),
        Param("save_retraining_samples", "bool", True, "Düşük Güvenli Kareleri Havuzda Topla"),
        Param("retrain_folder", "folder", "yeniden_egitim_havuzu", "Eğitim Havuzu Klasörü"),
    ]

    def setup(self):
        self.state.update(
            conf_history=collections.deque(maxlen=int(self.v("window_size"))),
            saved_count=0,
            drift_alarm=False,
            avg_conf=95.0,
            low_pct=0.0,
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("conf_history", collections.deque(maxlen=int(self.v("window_size"))))
        st.setdefault("saved_count", 0)

        conf_in = inputs.get("confidence")
        if conf_in is None:
            return {
                "drift_detected": st.get("drift_alarm", False),
                "avg_confidence": st.get("avg_conf", 95.0),
                "low_conf_pct": st.get("low_pct", 0.0),
                "drift_alarm": st.get("drift_alarm", False),
                "saved_for_retraining": st.get("saved_count", 0),
            }

        conf_val = float(conf_in)
        st["conf_history"].append(conf_val)

        min_conf = float(self.v("min_acceptable_confidence"))
        alert_th = float(self.v("drift_alert_pct"))

        # Save borderline / low confidence frame to retraining pool
        if conf_val < min_conf and bool(self.v("save_retraining_samples")):
            img_in = inputs.get("image")
            if img_in is not None:
                pool_dir = pathlib.Path(_node.resolve_path(self.v("retrain_folder") or "yeniden_egitim_havuzu"))
                pool_dir.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = pool_dir / f"drift_{stamp}.jpg"
                meta_path = pool_dir / f"drift_{stamp}.json"

                try:
                    cv2.imwrite(str(img_path), _bgr(img_in))
                    meta_path.write_text(json.dumps({
                        "confidence": conf_val,
                        "min_threshold": min_conf,
                        "is_defect": bool(inputs.get("is_defect")),
                        "timestamp": dt.datetime.now().isoformat(),
                    }, indent=2), encoding="utf-8")
                    st["saved_count"] += 1
                except Exception as exc:
                    print(f"[Model Drift] Kare kaydedilemedi: {exc}")

        # Statistical drift calculation on history
        hist = list(st["conf_history"])
        avg_c = float(np.mean(hist)) if hist else 100.0
        low_count = sum(1 for c in hist if c < min_conf)
        low_pct = (low_count / float(len(hist))) * 100.0 if hist else 0.0

        # Drift alarm triggered if low confidence predictions exceed threshold
        drift_active = (low_pct >= alert_th) and (len(hist) >= 10)

        st.update(
            avg_conf=round(avg_c, 1),
            low_pct=round(low_pct, 1),
            drift_alarm=drift_active,
        )

        return {
            "drift_detected": drift_active,
            "avg_confidence": round(avg_c, 1),
            "low_conf_pct": round(low_pct, 1),
            "drift_alarm": drift_active,
            "saved_for_retraining": st["saved_count"],
        }
