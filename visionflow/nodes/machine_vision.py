"""Industrial cameras: GigE Vision and USB3 Vision cameras (Basler, Hikrobot, Daheng, FLIR, Allied Vision, ...).

Two ways in, both optional extra packages:
- Basler cameras through Basler's own pypylon (pip install pypylon, no other download).
- Every other GenICam camera through Harvesters and the GenTL producer (.cti file) that comes with the camera maker's
  software, e.g. Hikrobot MVS. The .cti file is found by itself in the usual install places.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import cv2
import numpy as np

from ..core import NoNewFrame, Node, Param, PortType, register

CAT = "1 · Giriş"
BACKENDS = ["Basler (pypylon)", "GenICam / GigE Vision (Hikrobot, Daheng, FLIR…)"]
RAW_GAIN_DB = 0.0359             # Basler GainRaw step on older GigE cameras (dB per unit)
TRIGGERS = ["Sürekli (serbest akış)", "Yazılım tetiği", "Donanım tetiği (Line0)", "Donanım tetiği (Line1)"]

# GenICam pixel format -> OpenCV conversion. OpenCV names Bayer patterns one row lower than GenICam.
_BAYER = {"BayerRG": cv2.COLOR_BayerBG2BGR, "BayerBG": cv2.COLOR_BayerRG2BGR,
          "BayerGR": cv2.COLOR_BayerGB2BGR, "BayerGB": cv2.COLOR_BayerGR2BGR}


def to_bgr(data: np.ndarray, fmt: str) -> np.ndarray:
    """Camera buffer in GenICam pixel format `fmt` (Mono8, Mono12, BayerRG8, RGB8, BGR8, ...) -> 8-bit BGR image."""
    fmt = fmt or "Mono8"
    m = re.search(r"(\d+)(p|Packed)?$", fmt)
    bits = int(m.group(1)) if m else 8
    if m and m.group(2) and bits != 8:
        raise RuntimeError(f"Sıkıştırılmış piksel biçimi desteklenmiyor: {fmt} (kamerada Mono8, Mono12 veya "
                           "BayerRG8 seçin)")
    if data.dtype != np.uint8:                          # 10/12/16-bit cameras: keep the top 8 bits
        data = (data.astype(np.uint32) >> max(0, bits - 8)).clip(0, 255).astype(np.uint8)
    if fmt.startswith("Mono"):
        img = data.reshape(data.shape[0], data.shape[1]) if data.ndim == 3 else data
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for name, code in _BAYER.items():
        if fmt.startswith(name):
            img = data.reshape(data.shape[0], data.shape[1]) if data.ndim == 3 else data
            return cv2.cvtColor(img, code)
    if fmt.startswith("RGB") and data.ndim == 3:
        return cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
    if fmt.startswith("BGR") and data.ndim == 3:
        return np.ascontiguousarray(data[:, :, :3])
    raise RuntimeError(f"Desteklenmeyen piksel biçimi: {fmt} (kamerada Mono8, BayerRG8 veya RGB8 seçin)")


def cti_files() -> list[str]:
    """GenTL producers (.cti) installed on this computer: GENICAM_GENTL64_PATH plus the usual maker folders."""
    dirs = []
    for var in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    if sys.platform == "win32":
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        dirs += [os.path.join(pf86, r"Common Files\MVS\Runtime\Win64_x64"),          # Hikrobot MVS
                 os.path.join(pf, r"Daheng Imaging\GalaxySDK\GenTL\Win64"),
                 os.path.join(pf, r"Teledyne\Spinnaker\cti64\vs2015")]
    else:
        dirs += ["/opt/MVS/lib/64", "/opt/MVS/lib/aarch64", "/opt/mvIMPACT_Acquire/lib/x86_64",
                 "/opt/spinnaker/lib/spinnaker-gentl", "/usr/lib/ids/cti"]
    found = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.cti"))):
            if f not in found:
                found.append(f)
    return found


# --- backends: one small class per SDK, so the block itself stays the same ------------------------------------------

class _PylonCamera:
    module = None                                    # tests put a fake pypylon.pylon here

    @classmethod
    def _pylon(cls):
        if cls.module is not None:
            return cls.module
        try:
            from pypylon import pylon
        except ImportError as exc:
            raise RuntimeError("Basler kameralar için 'pip install pypylon' kurun") from exc
        return pylon

    @classmethod
    def list(cls, cti: str = "") -> list[dict]:
        pylon = cls._pylon()
        return [{"serial": d.GetSerialNumber(), "model": d.GetModelName(), "vendor": "Basler"}
                for d in pylon.TlFactory.GetInstance().EnumerateDevices()]

    def __init__(self, serial: str, cti: str = ""):
        pylon = self.pylon = self._pylon()
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        dev = _pick(devices, serial, lambda d: d.GetSerialNumber())
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev))
        self.cam.Open()
        self.name = f"{dev.GetModelName()} ({dev.GetSerialNumber()})"
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def feature(self, names, value=None, execute=False):
        """Set (or run) the first of `names` the camera has; returns the name used or None."""
        for name in names:
            try:                     # pypylon raises its own error (not AttributeError) for an unknown feature
                node = getattr(self.cam, name)
                if execute:
                    node.Execute()
                else:
                    node.SetValue(value)
                return name
            except Exception:  # noqa: BLE001  (unknown here, read-only, or the value is out of range)
                continue
        return None

    def raw_gain(self, db: float) -> bool:
        """Older Basler GigE cameras only have GainRaw (camera units, about 0.0359 dB each)."""
        try:
            node = self.cam.GainRaw
            node.SetValue(int(min(max(round(db / RAW_GAIN_DB), node.GetMin()), node.GetMax())))
            return True
        except Exception:  # noqa: BLE001
            return False

    def start(self):
        self.cam.StartGrabbing(self.pylon.GrabStrategy_LatestImageOnly)

    def software_trigger(self, timeout_ms: int):
        for _ in range(4):                       # a picture left over from an earlier, timed-out trigger
            res = self.cam.RetrieveResult(0, self.pylon.TimeoutHandling_Return)
            if res is None or not res.IsValid():
                break
            res.Release()
        try:                                     # a trigger sent while the camera is still busy is ignored
            self.cam.WaitForFrameTriggerReady(int(timeout_ms), self.pylon.TimeoutHandling_Return)
        except Exception:  # noqa: BLE001  (not every camera / pypylon version has it)
            pass
        if self.feature(["TriggerSoftware"], execute=True) is None:
            raise RuntimeError("Kamera yazılım tetiğini kabul etmedi")

    def grab(self, timeout_ms: int):
        res = self.cam.RetrieveResult(int(timeout_ms), self.pylon.TimeoutHandling_Return)
        if res is None or not res.IsValid():
            return None
        try:
            if not res.GrabSucceeded():
                raise RuntimeError(f"Kare bozuk geldi: {res.GetErrorDescription()}")
            return np.array(self.converter.Convert(res).GetArray(), copy=True)
        finally:
            res.Release()

    def close(self):
        try:
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
        finally:
            self.cam.Close()


class _GenICamCamera:
    module = None                                    # tests put a fake harvesters.core here

    @classmethod
    def _harvester(cls, cti: str):
        mod = cls.module
        if mod is None:
            try:
                import harvesters.core as mod
            except ImportError as exc:
                raise RuntimeError("GenICam kameralar için 'pip install harvesters' kurun") from exc
        files = [cti.strip()] if cti.strip() else cti_files()
        if not files:
            raise RuntimeError("GenTL (.cti) dosyası bulunamadı. Kamera üreticisinin yazılımını kurun (Hikrobot "
                               "için MVS) veya ayarlardan .cti dosyasını seçin")
        for f in files:
            if not os.path.isfile(f):
                raise RuntimeError(f".cti dosyası yok: {f}")
        h = mod.Harvester()
        try:
            for f in files:
                h.add_file(f)
            h.update()
        except Exception:
            h.reset()
            raise
        return h

    @classmethod
    def list(cls, cti: str = "") -> list[dict]:
        h = cls._harvester(cti)
        try:
            return [{"serial": str(getattr(d, "serial_number", "")), "model": str(getattr(d, "model", "")),
                     "vendor": str(getattr(d, "vendor", ""))} for d in h.device_info_list]
        finally:
            h.reset()

    def __init__(self, serial: str, cti: str = ""):
        self.h = self._harvester(cti)
        try:
            infos = list(self.h.device_info_list)
            info = _pick(infos, serial, lambda d: str(getattr(d, "serial_number", "")))
            create = getattr(self.h, "create", None) or self.h.create_image_acquirer   # harvesters < 1.3
            self.ia = create(infos.index(info))
        except Exception:
            self.h.reset()
            raise
        self.nm = self.ia.remote_device.node_map
        self.name = f"{getattr(info, 'model', '')} ({getattr(info, 'serial_number', '')})"

    def feature(self, names, value=None, execute=False):
        for name in names:
            node = getattr(self.nm, name, None)
            if node is None:
                continue
            try:
                if execute:
                    node.execute()
                else:
                    node.value = value
                return name
            except Exception:  # noqa: BLE001
                continue
        return None

    def raw_gain(self, db: float) -> bool:
        try:
            node = self.nm.GainRaw
            node.value = int(min(max(round(db / RAW_GAIN_DB), node.min), node.max))
            return True
        except Exception:  # noqa: BLE001
            return False

    def start(self):
        (getattr(self.ia, "start", None) or self.ia.start_acquisition)()

    def software_trigger(self, timeout_ms: int):
        try_fetch = getattr(self.ia, "try_fetch", None)
        for _ in range(4 if try_fetch else 0):   # throw away a picture left over from an earlier trigger
            try:
                buf = try_fetch(timeout=0.001)
            except Exception:  # noqa: BLE001
                break
            if buf is None:
                break
            buf.queue()
        if self.feature(["TriggerSoftware"], execute=True) is None:
            raise RuntimeError("Kamera yazılım tetiğini kabul etmedi")

    def grab(self, timeout_ms: int):
        fetch = getattr(self.ia, "fetch", None) or self.ia.fetch_buffer
        try:
            buf = fetch(timeout=timeout_ms / 1000)
        except Exception as exc:  # noqa: BLE001  (harvesters raises its own TimeoutException)
            if "timeout" in type(exc).__name__.lower():
                return None
            raise
        with buf:
            comp = buf.payload.components[0]
            data = comp.data
            ch = int(getattr(comp, "num_components_per_pixel", 1) or 1)
            data = data.reshape(comp.height, comp.width, ch) if ch > 1 else data.reshape(comp.height, comp.width)
            return to_bgr(np.array(data, copy=True), str(comp.data_format))

    def close(self):
        try:
            try:
                (getattr(self.ia, "stop", None) or self.ia.stop_acquisition)()
            finally:
                self.ia.destroy()
        finally:
            self.h.reset()


def _pick(devices, serial: str, serial_of):
    devices = list(devices)
    if not devices:
        raise RuntimeError("Endüstriyel kamera bulunamadı (kablo, güç/PoE ve kameranın IP adresi bilgisayarla aynı "
                           "ağda mı?)")
    serial = serial.strip()
    if not serial:
        return devices[0]
    if serial.isdigit() and len(serial) <= 2 and int(serial) < len(devices) and \
            not any(serial_of(d) == serial for d in devices):
        return devices[int(serial)]                  # "0", "1": the n-th camera
    for d in devices:
        if serial_of(d) == serial:
            return d
    raise RuntimeError(f"Seri numarası {serial} olan kamera yok. Bulunanlar: "
                       + ", ".join(serial_of(d) for d in devices))


def backend_class(name: str):
    return _PylonCamera if name.startswith("Basler") else _GenICamCamera


def list_cameras(backend: str, cti: str = "") -> list[dict]:
    return backend_class(backend).list(cti)


@register
class IndustrialCamera(Node):
    title = "Endüstriyel Kamera"
    category = CAT
    description = ("GigE Vision / USB3 Vision endüstriyel kameralar: Basler (pypylon), Hikrobot, Daheng, FLIR ve diğer "
                   "GenICam kameralar (harvesters + üreticinin .cti dosyası). Pozlama, kazanç ve tetik ayarlanır. "
                   "Donanım tetiğinde akış her tetikte bir kare işler. Kameraları bulmak için: Araçlar → "
                   "Endüstriyel kameraları bul.")
    outputs = [("image", PortType.IMAGE), ("fps", PortType.NUMBER)]
    inputs = [("trigger", PortType.BOOL)]
    params = [Param("backend", "choice", BACKENDS[0], "Kamera türü", choices=BACKENDS),
              Param("serial", "str", "", "Seri no (boş = ilk kamera)"),
              Param("cti", "path", "", ".cti dosyası (GenICam; boş = otomatik bul)"),
              Param("exposure", "float", 0.0, "Pozlama (µs, 0 = kameranınki)", 0.0, 10_000_000.0, 100.0),
              Param("gain", "float", -1.0, "Kazanç (dB, -1 = kameranınki)", -1.0, 48.0, 0.5),
              Param("trigger_mode", "choice", TRIGGERS[0], "Tetik", choices=TRIGGERS),
              Param("timeout", "int", 1000, "Kare bekleme (ms)", 50, 60000)]
    setup_params = ("backend", "serial", "cti", "trigger_mode")
    is_source = True

    RETRY_SECONDS = 2.0
    HW_POLL_MS = 100                 # hardware trigger: look for a frame this often, so Stop stays quick

    def setup(self):
        self.state.update(cam=None, retry_at=0.0, t=time.perf_counter(), fps=0.0, applied={}, prev_trig=False)
        self._connect()

    def _connect(self):
        cam = backend_class(self.v("backend"))(self.v("serial"), self.v("cti") or "")
        try:
            mode = self.v("trigger_mode")
            cam.feature(["TriggerSelector"], "FrameStart")
            if mode == TRIGGERS[0]:
                cam.feature(["TriggerMode"], "Off")
            else:
                source = "Software" if mode == TRIGGERS[1] else mode[-6:-1]
                # a camera that refuses the trigger would wait forever without saying why
                if cam.feature(["TriggerMode"], "On") is None:
                    raise RuntimeError("Kamera tetik modunu desteklemiyor; 'Sürekli' seçin")
                if cam.feature(["TriggerSource"], source) is None:
                    raise RuntimeError(f"Kamerada '{source}' tetik girişi yok (Basler kameralarda girişler "
                                       "genelde Line1'den başlar)")
            cam.feature(["AcquisitionMode"], "Continuous")
            self.state["applied"] = {}
            self._apply_settings(cam)
            cam.start()
        except Exception:
            cam.close()
            raise
        self.state["cam"] = cam
        print(f"Endüstriyel kamera bağlandı: {cam.name}")

    def _apply_settings(self, cam):
        """Exposure and gain can change while running, without reconnecting."""
        applied = self.state["applied"]
        exp, gain = float(self.v("exposure")), float(self.v("gain"))
        if applied.get("exposure") != exp:
            if exp > 0:
                cam.feature(["ExposureAuto"], "Off")
                if cam.feature(["ExposureTime", "ExposureTimeAbs"], exp) is None:
                    print("Uyarı: kamera pozlama ayarını kabul etmedi")
            applied["exposure"] = exp
        if applied.get("gain") != gain:
            if gain >= 0:
                cam.feature(["GainAuto"], "Off")
                if cam.feature(["Gain", "GainAbs"], gain) is None:
                    if cam.raw_gain(gain):
                        print("Bilgi: bu kamerada kazanç ham birimle (GainRaw) ayarlandı; dB değeri yaklaşıktır")
                    else:
                        print("Uyarı: kamera kazanç ayarını kabul etmedi")
            applied["gain"] = gain

    def process(self, inputs):
        st = self.state
        if st["cam"] is None:                # cable or PoE dropped: reconnect every few seconds
            if time.monotonic() < st["retry_at"]:
                raise RuntimeError("Kamera bağlantısı yok, tekrar deneniyor")
            st["retry_at"] = time.monotonic() + self.RETRY_SECONDS
            self._connect()
        cam = st["cam"]
        self._apply_settings(cam)
        mode = self.v("trigger_mode")
        timeout = int(self.v("timeout"))
        if mode == TRIGGERS[1]:
            trig = inputs.get("trigger")
            if trig is None:                     # nothing connected: one fresh picture every frame
                fire = True
            else:                                # connected: one picture when it turns on
                fire, st["prev_trig"] = bool(trig) and not st["prev_trig"], bool(trig)
            if not fire:
                raise NoNewFrame("tetik bekleniyor")
            try:
                cam.software_trigger(timeout)
            except Exception:
                self._drop()
                raise
        elif mode != TRIGGERS[0]:
            timeout = min(timeout, self.HW_POLL_MS)
        try:
            frame = cam.grab(timeout)
        except Exception:
            self._drop()
            raise
        if frame is None:
            if mode in TRIGGERS[2:]:
                raise NoNewFrame("donanım tetiği bekleniyor")
            st["misses"] = st.get("misses", 0) + 1
            if st["misses"] >= 3:                # several silent time-outs in a row: reconnect
                self._drop()
            raise RuntimeError(f"Kameradan {timeout} ms içinde kare gelmedi")
        st["misses"] = 0
        now = time.perf_counter()
        dt, st["t"] = now - st["t"], now
        st["fps"] = 0.9 * st["fps"] + 0.1 * (1.0 / dt if dt > 0 else 0)
        return {"image": frame, "fps": round(st["fps"], 1)}

    def _drop(self):
        cam, self.state["cam"] = self.state.get("cam"), None
        self.state["retry_at"] = 0.0
        if cam is not None:
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass

    def teardown(self):
        self._drop()
