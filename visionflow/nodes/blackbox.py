"""Black Box Video Recorder (Pre-Trigger & Post-Trigger Ring Buffer):
Maintains a rolling ring buffer of recent camera frames in memory.
When a defect/alarm triggers, it saves an MP4 video clip showing N seconds BEFORE and M seconds AFTER the defect!"""
from __future__ import annotations

import collections
import datetime as dt
import os
import pathlib
import threading
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr

CAT_OUT = "5 · Çıkış ve Otomasyon"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None:
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


@register
class BlackBoxRecorder(Node):
    title = "Kara Kutu Video Kaydedici"
    category = CAT_OUT
    description = (
        "Sürekli olarak son N saniyelik kamera görüntüsünü bellekte (halka tampon) tutar. "
        "Hata veya alarm tetiklendiğinde, arızanın gerçekleştiği andan N saniye öncesini ve M saniye sonrasını "
        "birleştirip otomatik MP4 video klibi oluşturur. WhatsApp ve E-posta ile paylaşılabilir."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("video_path", TEXT),
        ("recorded", BOOL),
        ("total_recordings", NUM),
        ("is_recording", BOOL),
    ]
    params = [
        Param("pre_trigger_sec", "float", 3.0, "Tetik Öncesi Süre (sn)", 0.5, 30.0),
        Param("post_trigger_sec", "float", 3.0, "Tetik Sonrası Süre (sn)", 0.5, 30.0),
        Param("fps", "float", 15.0, "Kayıt FPS", 1.0, 60.0),
        Param("output_folder", "folder", "kara_kutu_videolari", "Kayıt Klasörü"),
        Param("filename_prefix", "str", "hata_klibi", "Dosya Ön Eki"),
        Param("max_width", "int", 640, "Video Genişliği (px)", 160, 3840),
    ]

    def setup(self):
        fps = max(1.0, float(self.v("fps")))
        pre_sec = max(0.5, float(self.v("pre_trigger_sec")))
        buf_size = max(5, int(pre_sec * fps))

        self.state.update(
            ring_buffer=collections.deque(maxlen=buf_size),
            post_buffer=[],
            is_collecting=False,
            frames_needed=0,
            video_path="",
            recorded=False,
            total_recordings=0,
            prev_trig=False,
        )

    def _encode_video(self, all_frames: list[np.ndarray], out_path: str, fps: float):
        if not all_frames:
            return
        try:
            h, w = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                # Fallback to AVI / XVID
                out_path_avi = out_path.rsplit(".", 1)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(out_path_avi, fourcc, fps, (w, h))

            for f in all_frames:
                writer.write(f)
            writer.release()
            print(f"[Kara Kutu] Video kaydedildi: {out_path} ({len(all_frames)} kare)")
        except Exception as exc:
            print(f"[Kara Kutu] Video kodlama hatası: {exc}")

    def process(self, inputs):
        st = self.state
        st.setdefault("ring_buffer", collections.deque(maxlen=45))
        st.setdefault("post_buffer", [])
        st.setdefault("is_collecting", False)
        st.setdefault("frames_needed", 0)
        st.setdefault("video_path", "")
        st.setdefault("total_recordings", 0)

        img_in = inputs.get("image")
        if img_in is None:
            return {
                "video_path": st["video_path"],
                "recorded": False,
                "total_recordings": st["total_recordings"],
                "is_recording": st["is_collecting"],
            }

        bgr = _bgr(img_in)
        max_w = int(self.v("max_width"))
        h, w = bgr.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            bgr = cv2.resize(bgr, (max_w, int(h * scale)))

        fps = max(1.0, float(self.v("fps")))
        post_sec = max(0.5, float(self.v("post_trigger_sec")))

        # Check defect trigger
        trig_fired = _rising(self, inputs.get("trigger"), "prev_trig")
        if trig_fired and not st["is_collecting"]:
            st["is_collecting"] = True
            st["frames_needed"] = max(5, int(post_sec * fps))
            st["post_buffer"] = []

        recorded_flag = False

        if st["is_collecting"]:
            st["post_buffer"].append(bgr.copy())
            if len(st["post_buffer"]) >= st["frames_needed"]:
                # Finished collecting post-trigger frames!
                pre_frames = list(st["ring_buffer"])
                post_frames = list(st["post_buffer"])
                all_frames = pre_frames + post_frames

                out_folder = pathlib.Path(_node.resolve_path(self.v("output_folder") or "kara_kutu_videolari"))
                out_folder.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = str(self.v("filename_prefix") or "hata_klibi")
                out_file = str((out_folder / f"{prefix}_{stamp}.mp4").resolve())

                # Encode in background thread so vision loop isn't stalled
                threading.Thread(
                    target=self._encode_video,
                    args=(all_frames, out_file, fps),
                    daemon=True,
                ).start()

                st["video_path"] = out_file
                st["total_recordings"] += 1
                st["is_collecting"] = False
                st["post_buffer"] = []
                recorded_flag = True
        else:
            # Normal ring buffering
            st["ring_buffer"].append(bgr.copy())

        return {
            "video_path": st["video_path"],
            "recorded": recorded_flag,
            "total_recordings": st["total_recordings"],
            "is_recording": st["is_collecting"],
        }
