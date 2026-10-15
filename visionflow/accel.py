"""Inference device (CPU / GPU) for every ONNX and OpenCV-DNN block, decided in one place.

The device comes from, in this order: the block's own "Cihaz" setting, the flow's setting (``graph.accel``, saved
in the flow file) and this computer's setting (``hizlandirma.json`` in the user's VisionFlow folder); "Otomatik"
when none is chosen. A device that is missing or fails to start falls back to the CPU with a warning, so a flow
made on a PC with a graphics card still runs on one without.
"""
from __future__ import annotations

import importlib.util
import json
import time

AUTO, CPU = "Otomatik", "CPU"
CUDA, TENSORRT, OPENVINO, DIRECTML, COREML = ("NVIDIA CUDA", "TensorRT", "Intel OpenVINO", "DirectML (Windows)",
                                             "Apple CoreML")
DEVICES = [AUTO, CPU, CUDA, TENSORRT, OPENVINO, DIRECTML, COREML]
GLOBAL = "Genel ayar"                       # block setting: use the flow's / computer's device
BLOCK_CHOICES = [GLOBAL] + DEVICES

PROVIDERS = {CPU: "CPUExecutionProvider", CUDA: "CUDAExecutionProvider", TENSORRT: "TensorrtExecutionProvider",
             OPENVINO: "OpenVINOExecutionProvider", DIRECTML: "DmlExecutionProvider",
             COREML: "CoreMLExecutionProvider"}
SHORT = {"CPUExecutionProvider": "CPU", "CUDAExecutionProvider": "CUDA", "TensorrtExecutionProvider": "TensorRT",
         "OpenVINOExecutionProvider": "OpenVINO", "DmlExecutionProvider": "DirectML",
         "CoreMLExecutionProvider": "CoreML"}
# "Otomatik" picks the first of these that onnxruntime offers (TensorRT is left out: it spends minutes building an
# engine on the first start, so it is used only when chosen)
AUTO_ORDER = [CUDA, DIRECTML, COREML, OPENVINO]
FALLBACK = "CPU (yedek)"
SETTINGS_FILE = "hizlandirma.json"


def _ort():
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("ONNX modelleri için 'pip install onnxruntime' (NVIDIA için onnxruntime-gpu) "
                           "kurun") from exc
    return onnxruntime


def _warn(msg: str) -> None:
    print(f"UYARI: {msg}")


def clean_device(value, allow_global: bool = False) -> str | None:
    """A known device name, or None."""
    ok = BLOCK_CHOICES if allow_global else DEVICES
    return value if isinstance(value, str) and value in ok else None


# --- settings ------------------------------------------------------------------------------------------------------

def _settings_path():
    from .ai.config import home
    return home() / SETTINGS_FILE


def pc_settings() -> dict:
    """This computer's choice: {"device": ...}. Missing / broken file → {}."""
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    if clean_device(data.get("device")):
        out["device"] = data["device"]
    if isinstance(data.get("fp16"), bool):
        out["fp16"] = data["fp16"]
    return out


def save_pc_settings(device: str | None = None, fp16: bool | None = None) -> None:
    data = pc_settings()
    if device is not None:
        if not clean_device(device):
            raise ValueError(f"Bilinmeyen cihaz: {device}")
        data["device"] = device
    if fp16 is not None:
        data["fp16"] = bool(fp16)
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_graph_accel(value, warnings: list | None = None) -> dict:
    """The flow's saved "accel" key: {"device": name, "fp16": bool}; anything unknown is dropped."""
    if value is None:
        return {}
    out: dict = {}
    if not isinstance(value, dict):
        if warnings is not None:
            warnings.append("Hızlandırma (GPU) ayarı bozuk; bilgisayar ayarı kullanılacak")
        return out
    dev = value.get("device")
    if dev is not None:
        if clean_device(dev):
            out["device"] = dev
        elif warnings is not None:
            warnings.append(f"Hızlandırma: bilinmeyen cihaz '{dev}', bilgisayar ayarı kullanılacak")
    if value.get("fp16") is True:
        out["fp16"] = True
    return out


def default_device(graph=None) -> str:
    """The flow's device if it sets one, else this computer's, else Otomatik."""
    acc = getattr(graph.root if graph is not None else None, "accel", None) or {}
    return clean_device(acc.get("device")) or pc_settings().get("device") or AUTO


def default_fp16(graph=None) -> bool:
    acc = getattr(graph.root if graph is not None else None, "accel", None) or {}
    if "fp16" in acc:
        return bool(acc["fp16"])
    return bool(pc_settings().get("fp16", False))


def resolve(device=None, graph=None) -> str:
    """Block setting → concrete choice (still possibly "Otomatik")."""
    dev = clean_device(device)
    return dev if dev else default_device(graph)


# --- detection -----------------------------------------------------------------------------------------------------

def ort_providers() -> list[str]:
    try:
        return list(_ort().get_available_providers())
    except Exception:  # noqa: BLE001  (not installed, broken install)
        return []


def opencv_cuda_count() -> int:
    try:
        import cv2
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # noqa: BLE001  (no cuda module, no driver)
        return 0


def has_openvino_package() -> bool:
    try:
        return importlib.util.find_spec("openvino") is not None
    except (ImportError, ValueError):
        return False


def available_devices() -> list[dict]:
    """Every device with {"name", "available", "note"} for this computer."""
    provs = ort_providers()
    out = []
    for name in DEVICES[1:]:
        ok = PROVIDERS[name] in provs
        note = ""
        if not provs:
            note = "onnxruntime kurulu değil"
        elif name == CUDA and not ok:
            note = "onnxruntime-gpu ve NVIDIA sürücüsü gerekli"
        elif name == TENSORRT and not ok:
            note = "onnxruntime-gpu + TensorRT gerekli"
        elif name == OPENVINO and not ok:
            note = ("openvino kurulu; onnxruntime-openvino gerekli" if has_openvino_package()
                    else "onnxruntime-openvino gerekli")
        elif name == DIRECTML and not ok:
            note = "onnxruntime-directml gerekli (Windows)"
        elif name == COREML and not ok:
            note = "yalnız macOS"
        out.append({"name": name, "available": ok, "note": note})
    n = opencv_cuda_count()
    for d in out:
        if d["name"] == CUDA and n:
            d["note"] = (d["note"] + "; " if d["note"] else "") + f"OpenCV CUDA: {n} kart"
    return out


def auto_device(providers: list[str] | None = None) -> str:
    provs = ort_providers() if providers is None else providers
    return next((d for d in AUTO_ORDER if PROVIDERS[d] in provs), CPU)


def status_line() -> str:
    """One line for the self-test and the CLI: what this computer can use."""
    devs = available_devices()
    if not ort_providers():
        return "onnxruntime yok: ONNX blokları çalışmaz"
    gpus = [d["name"] for d in devs if d["available"] and d["name"] != CPU]
    cv = opencv_cuda_count()
    extra = f" · OpenCV CUDA: {cv} kart" if cv else ""
    if gpus:
        return f"hızlandırıcı: {', '.join(gpus)} · Otomatik → {auto_device()}{extra}"
    return f"yalnız CPU (GPU sağlayıcısı yok){extra}"


# --- sessions ------------------------------------------------------------------------------------------------------

def _plan(device: str, fp16: bool) -> list:
    """onnxruntime providers list (names or (name, options) pairs) for a concrete device, CPU last."""
    from .ai.config import home
    if device == CUDA:
        first = [("CUDAExecutionProvider", {"device_id": 0})]
    elif device == TENSORRT:
        first = [("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": bool(fp16),
                                                "trt_engine_cache_enable": True,
                                                "trt_engine_cache_path": str(home() / "tensorrt_onbellek")}),
                 ("CUDAExecutionProvider", {"device_id": 0})]
    elif device == OPENVINO:
        opts = {"device_type": "AUTO:GPU,CPU"}
        if fp16:
            opts["precision"] = "FP16"
        first = [("OpenVINOExecutionProvider", opts)]
    elif device == DIRECTML:
        first = [("DmlExecutionProvider", {"device_id": 0})]
    elif device == COREML:
        first = ["CoreMLExecutionProvider"]
    else:
        first = []
    return first + ["CPUExecutionProvider"]


def _name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


def _label(sess, wanted: str) -> str:
    try:
        used = list(sess.get_providers())
    except Exception:  # noqa: BLE001  (stand-in sessions in tests, very old onnxruntime)
        used = []
    first = SHORT.get(used[0], used[0]) if used else SHORT.get(PROVIDERS.get(wanted, ""), "CPU")
    if wanted != CPU and first == "CPU":
        return FALLBACK
    return first


def session(path, device=None, fp16: bool = False, *, graph=None, who: str = ""):
    """An onnxruntime.InferenceSession on the chosen device ("Genel ayar"/None = the flow's / computer's).
    A device that is not installed or fails to start falls back to the CPU with a warning. The device actually
    used is in ``device_of(sess)``: "CUDA", "OpenVINO", ... or "CPU (yedek)" after a fallback."""
    ort = _ort()
    want = resolve(device, graph)
    provs = list(ort.get_available_providers())
    tag = f"[{who}] " if who else ""
    if want == AUTO:
        want = auto_device(provs)
    if want != CPU and PROVIDERS[want] not in provs:
        _warn(f"{tag}{want} bu bilgisayarda kullanılamıyor (onnxruntime sağlayıcısı yok); CPU kullanılıyor")
        want_label, plan = FALLBACK, _plan(CPU, fp16)
    else:
        want_label, plan = None, _plan(want, fp16)
    plan = [p for p in plan if _name(p) in provs or _name(p) == "CPUExecutionProvider"]
    sess = None
    if len(plan) > 1:
        try:
            sess = ort.InferenceSession(path, providers=plan)
        except Exception as exc:  # noqa: BLE001  (driver / CUDA / cuDNN missing, unsupported model ...)
            _warn(f"{tag}{want} başlatılamadı ({type(exc).__name__}: {str(exc)[:200]}); CPU ile devam ediliyor")
            want_label = FALLBACK
    if sess is None:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    label = want_label or _label(sess, want)
    if label == FALLBACK and want_label is None:
        _warn(f"{tag}{want} istenen cihazda açılamadı (sürücü / CUDA kitaplıkları?); CPU kullanılıyor")
    try:
        sess.vf_device = label
    except (AttributeError, TypeError):
        pass
    return sess


def device_of(sess) -> str:
    label = getattr(sess, "vf_device", None)
    return label if isinstance(label, str) else _label(sess, CPU)


def _flow_of(node):
    ref = getattr(node, "_owner", None)
    return ref() if ref is not None else None


def node_session(node, path, fp16: bool | None = None):
    """session() for a block: its "device" setting (if it has one) over the flow's / computer's choice."""
    graph = _flow_of(node)
    dev = node.values.get("device", GLOBAL) if hasattr(node, "values") else GLOBAL
    return session(path, dev, default_fp16(graph) if fp16 is None else fp16, graph=graph,
                   who=getattr(node, "title", ""))


# --- OpenCV DNN ----------------------------------------------------------------------------------------------------

def opencv_backend(net, device=None, *, graph=None, fp16: bool = False, who: str = "") -> str:
    """Set backend/target of a cv2.dnn net for the device. Returns the device used ("CUDA", "OpenVINO", "OpenCL",
    "CPU" or "CPU (yedek)")."""
    import cv2
    dnn = cv2.dnn
    want = resolve(device, graph)
    tag = f"[{who}] " if who else ""
    cuda_ok = opencv_cuda_count() > 0
    if want == AUTO:
        want = CUDA if cuda_ok else CPU
    if want in (CUDA, TENSORRT):
        if cuda_ok and hasattr(dnn, "DNN_BACKEND_CUDA"):
            try:
                net.setPreferableBackend(dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(dnn.DNN_TARGET_CUDA_FP16 if fp16 else dnn.DNN_TARGET_CUDA)
                return "CUDA"
            except Exception as exc:  # noqa: BLE001
                _warn(f"{tag}OpenCV CUDA başlatılamadı ({exc}); CPU kullanılıyor")
        else:
            _warn(f"{tag}Bu OpenCV CUDA desteğiyle derlenmemiş veya kart yok; CPU kullanılıyor")
    elif want == OPENVINO:
        backend = getattr(dnn, "DNN_BACKEND_INFERENCE_ENGINE", None)
        if backend is not None and has_openvino_package():
            try:
                net.setPreferableBackend(backend)
                net.setPreferableTarget(dnn.DNN_TARGET_OPENCL_FP16 if fp16 else dnn.DNN_TARGET_CPU)
                return "OpenVINO"
            except Exception as exc:  # noqa: BLE001
                _warn(f"{tag}OpenVINO başlatılamadı ({exc}); CPU kullanılıyor")
        else:
            _warn(f"{tag}OpenCV için OpenVINO bulunamadı; CPU kullanılıyor")
    elif want == DIRECTML:
        # no DirectML in OpenCV: OpenCL reaches the same graphics cards
        try:
            if cv2.ocl.haveOpenCL():
                net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(dnn.DNN_TARGET_OPENCL_FP16 if fp16 else dnn.DNN_TARGET_OPENCL)
                return "OpenCL"
        except Exception:  # noqa: BLE001
            pass
        _warn(f"{tag}OpenCL bulunamadı; CPU kullanılıyor")
    elif want == COREML:
        _warn(f"{tag}OpenCV CoreML kullanamaz; CPU kullanılıyor")
    net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(dnn.DNN_TARGET_CPU)
    return "CPU" if want == CPU else FALLBACK


# --- benchmark -----------------------------------------------------------------------------------------------------

def tiny_model() -> bytes | str | None:
    """A small convolution network for the speed test: built with the 'onnx' package when installed, else an
    .onnx file found in the program's models folder. None when neither is there."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return _bundled_model()
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.standard_normal((16, 3, 3, 3)).astype(np.float32) * 0.1, "w1")
    w2 = numpy_helper.from_array(rng.standard_normal((32, 16, 3, 3)).astype(np.float32) * 0.1, "w2")
    w3 = numpy_helper.from_array(rng.standard_normal((32, 32, 3, 3)).astype(np.float32) * 0.1, "w3")
    nodes = [helper.make_node("Conv", ["x", "w1"], ["c1"], pads=[1, 1, 1, 1]),
             helper.make_node("Relu", ["c1"], ["r1"]),
             helper.make_node("Conv", ["r1", "w2"], ["c2"], pads=[1, 1, 1, 1], strides=[2, 2]),
             helper.make_node("Relu", ["c2"], ["r2"]),
             helper.make_node("Conv", ["r2", "w3"], ["c3"], pads=[1, 1, 1, 1]),
             helper.make_node("GlobalAveragePool", ["c3"], ["y"])]
    graph = helper.make_graph(nodes, "hiz_testi", [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                                                                 [1, 3, 224, 224])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32, 1, 1])],
                              initializer=[w1, w2, w3])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _bundled_model() -> str | None:
    import pathlib
    from .core.node import search_dirs
    for d in search_dirs():
        for f in sorted((d / "models").glob("*.onnx")) if (d / "models").is_dir() else []:
            return str(f)
    return None


def benchmark(devices: list[str] | None = None, runs: int = 20, model=None, progress=None) -> list[dict]:
    """Time the tiny model on each available device: [{"device", "used", "ms", "error"}]."""
    import numpy as np
    model = tiny_model() if model is None else model
    if model is None:
        raise RuntimeError("Hız testi için model yok: 'pip install onnx' kurun veya models klasörüne bir .onnx koyun")
    if devices is None:
        devices = [d["name"] for d in available_devices() if d["available"]] or [CPU]
    out = []
    for dev in devices:
        if progress:
            progress(dev)
        row = {"device": dev, "used": "", "ms": None, "error": ""}
        try:
            sess = session(model, dev)
            row["used"] = device_of(sess)
            inp = sess.get_inputs()[0]
            shape = [s if isinstance(s, int) and s > 0 else (1 if i < 2 else 224) for i, s in enumerate(inp.shape)]
            x = np.random.default_rng(0).random(shape, dtype=np.float32)
            for _ in range(2):                        # warm-up (kernels, engines)
                sess.run(None, {inp.name: x})
            t0 = time.perf_counter()
            for _ in range(max(1, runs)):
                sess.run(None, {inp.name: x})
            row["ms"] = (time.perf_counter() - t0) * 1000 / max(1, runs)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out.append(row)
    return out


def report_lines(bench: bool = True, runs: int = 20) -> list[str]:
    """Text of `python -m visionflow gpu-test` and of the dialog's copy button."""
    lines = ["VisionFlow hızlandırma (GPU) testi"]
    provs = ort_providers()
    lines.append(f"onnxruntime sağlayıcıları: {', '.join(provs) if provs else 'yok (onnxruntime kurulu değil)'}")
    lines.append(f"OpenCV CUDA kartı: {opencv_cuda_count()}")
    lines.append(f"Seçili varsayılan (bu bilgisayar): {pc_settings().get('device', AUTO)}"
                 f"{' → ' + auto_device() if pc_settings().get('device', AUTO) == AUTO and provs else ''}")
    lines.append("Cihazlar:")
    for d in available_devices():
        lines.append(f"  [{'VAR' if d['available'] else 'YOK'}] {d['name']}{' — ' + d['note'] if d['note'] else ''}")
    if bench and provs:
        lines.append("Hız testi (küçük ağ, 224x224):")
        try:
            for r in benchmark(runs=runs):
                if r["error"]:
                    lines.append(f"  {r['device']}: HATA {r['error']}")
                else:
                    lines.append(f"  {r['device']}: {r['ms']:.2f} ms/kare ({r['used']})")
        except RuntimeError as exc:
            lines.append(f"  {exc}")
    return lines


def main(argv=None) -> int:
    argv = list(argv or [])
    if argv[:1] in (["-h"], ["--help"]):
        print("Kullanım: python -m visionflow gpu-test [--hizli] [--tekrar N]\n"
              "  Bulunan cihazları (CPU, CUDA, OpenVINO, DirectML…) listeler ve küçük bir ağla hız testi yapar.\n"
              "  --hizli    hız testini atla\n  --tekrar N kaç kez çalıştırılsın (varsayılan 20)")
        return 0
    runs = 20
    if "--tekrar" in argv:
        try:
            runs = max(1, int(argv[argv.index("--tekrar") + 1]))
        except (IndexError, ValueError):
            print("--tekrar bir sayı ister")
            return 2
    for line in report_lines(bench="--hizli" not in argv, runs=runs):
        print(line)
    return 0 if ort_providers() else 1
