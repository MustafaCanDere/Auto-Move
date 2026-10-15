"""Industrial / automation outputs: Modbus TCP (PLC), MQTT, serial port, video recording."""
from __future__ import annotations

import datetime as dt
import os
import socket
import struct
import threading

import cv2

from ..core import Node, Param, PortType, register
from .outputs import _rising

CAT = "5 · Çıkış ve Otomasyon"


class ModbusClient:
    """Minimal Modbus TCP client (no extra dependency). Supports coils and holding registers."""

    def __init__(self, host: str, port: int = 502, unit: int = 1, timeout: float = 2.0):
        self.host, self.port, self.unit, self.timeout = host, port, unit, timeout
        self.sock: socket.socket | None = None
        self.tid = 0
        self.lock = threading.Lock()

    def _connect(self):
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), self.timeout)
            self.sock.settimeout(self.timeout)

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC bağlantıyı kapattı")
            buf += chunk
        return buf

    def request(self, pdu: bytes) -> bytes:
        with self.lock:
            try:
                self._connect()
                self.tid = (self.tid + 1) & 0xFFFF
                self.sock.sendall(struct.pack(">HHHB", self.tid, 0, len(pdu) + 1, self.unit) + pdu)
                header = self._recv(7)
                _, _, length, _ = struct.unpack(">HHHB", header)
                if length < 2:
                    self.close()
                    raise RuntimeError(f"Modbus cevabı çok kısa (uzunluk {length})")
                resp = self._recv(length - 1)
            except OSError:
                self.close()
                raise
        if resp[0] & 0x80:
            if len(resp) < 2:
                raise RuntimeError("Modbus hata cevabı eksik")
            raise RuntimeError(f"Modbus hata kodu {resp[1]}")
        return resp

    def write_coil(self, addr: int, value: bool):
        self.request(struct.pack(">BHH", 5, addr, 0xFF00 if value else 0))

    def write_register(self, addr: int, value: int):
        self.request(struct.pack(">BHH", 6, addr, int(value) & 0xFFFF))

    def read_registers(self, addr: int, count: int = 1) -> list[int]:
        resp = self.request(struct.pack(">BHH", 3, addr, count))
        return list(struct.unpack(f">{count}H", resp[2:2 + 2 * count]))

    def read_coils(self, addr: int, count: int = 1) -> list[bool]:
        resp = self.request(struct.pack(">BHH", 1, addr, count))
        bits = int.from_bytes(resp[2:], "little")
        return [bool(bits >> i & 1) for i in range(count)]

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


@register
class ModbusWrite(Node):
    setup_params = ("host", "port", "unit")   # read only when the block starts: changing them restarts it
    title = "PLC Yaz (Modbus TCP)"
    category = CAT
    description = ("Siemens, Delta, Schneider, Mitsubishi vb. PLC'lere Modbus TCP ile bobin (coil) veya "
                   "register yazar. Değer değiştiğinde gönderir.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]
    params = [Param("host", "str", "192.168.0.10", "PLC IP"), Param("port", "int", 502, "Port", 1, 65535),
              Param("unit", "int", 1, "Unit ID", 0, 255),
              Param("kind", "choice", "coil", "Tip", choices=["coil", "register"]),
              Param("address", "int", 0, "Adres", 0, 65535)]

    def setup(self):
        self.state.update(client=ModbusClient(self.v("host"), self.v("port"), self.v("unit")), last=object())

    def process(self, inputs):
        val = inputs["value"]
        if val is None:
            return {"ok": False}
        val = bool(val) if self.v("kind") == "coil" else int(val)
        sent = (self.v("kind"), self.v("address"), val)     # a new address or type must be written again
        if sent == self.state["last"]:
            return {"ok": True}
        c = self.state["client"]
        (c.write_coil if self.v("kind") == "coil" else c.write_register)(self.v("address"), val)
        self.state["last"] = sent
        return {"ok": True}

    def teardown(self):
        if self.state.get("client"):
            self.state["client"].close()


@register
class ModbusRead(Node):
    setup_params = ("host", "port", "unit")   # read only when the block starts: changing them restarts it
    title = "PLC Oku (Modbus TCP)"
    category = "1 · Giriş"
    description = "PLC'den bobin veya register okur (ör. bant çalışıyor mu, ürün tipi kodu)."
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL)]
    params = ModbusWrite.params

    def setup(self):
        self.state["client"] = ModbusClient(self.v("host"), self.v("port"), self.v("unit"))

    def process(self, inputs):
        c = self.state["client"]
        if self.v("kind") == "coil":
            v = c.read_coils(self.v("address"))[0]
        else:
            v = c.read_registers(self.v("address"))[0]
        return {"value": int(v), "bool": bool(v)}

    def teardown(self):
        if self.state.get("client"):
            self.state["client"].close()


@register
class MqttPublish(Node):
    setup_params = ("host", "port")   # read only when the block starts: changing them restarts it
    title = "MQTT Gönder"
    category = CAT
    description = "Değeri bir MQTT konusuna yayınlar (Node-RED, Home Assistant, IoT panelleri). paho-mqtt gerekir."
    inputs = [("value", PortType.ANY), ("trigger", PortType.BOOL)]
    outputs = [("sent", PortType.BOOL)]
    params = [Param("host", "str", "localhost", "Sunucu"), Param("port", "int", 1883, "Port", 1, 65535),
              Param("topic", "str", "visionflow/deger", "Konu"),
              Param("on_change", "bool", True, "Sadece değişince gönder")]

    def setup(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT için 'pip install paho-mqtt' kurun") from exc
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        client.connect_async(self.v("host"), self.v("port"))
        client.loop_start()
        self.state.update(client=client, last=object())

    def process(self, inputs):
        val = inputs["value"]
        if inputs["trigger"] is not None:
            if not _rising(self, inputs["trigger"]):
                return {"sent": False}
        elif self.v("on_change") and val == self.state["last"]:
            return {"sent": False}
        self.state["last"] = val
        self.state["client"].publish(self.v("topic"), str(val))
        return {"sent": True}

    def teardown(self):
        if self.state.get("client"):
            self.state["client"].loop_stop()
            self.state["client"].disconnect()


@register
class SerialWrite(Node):
    setup_params = ("baud", "port")   # read only when the block starts: changing them restarts it
    title = "Seri Port Gönder"
    category = CAT
    description = "Arduino, röle kartı vb. cihazlara seri porttan metin gönderir. {v} yerine değer yazılır. pyserial gerekir."
    inputs = [("value", PortType.ANY), ("trigger", PortType.BOOL)]
    outputs = [("sent", PortType.BOOL)]
    params = [Param("port", "str", "COM3", "Port (COM3, /dev/ttyUSB0)"),
              Param("baud", "choice", "9600", "Baud", choices=["9600", "19200", "38400", "57600", "115200"]),
              Param("template", "str", "{v}\\n", "Mesaj şablonu")]

    def setup(self):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Seri port için 'pip install pyserial' kurun") from exc
        self.state.update(ser=serial.Serial(self.v("port"), int(self.v("baud")), timeout=0.1), last=object())

    def process(self, inputs):
        val = inputs["value"]
        if inputs["trigger"] is not None:
            if not _rising(self, inputs["trigger"]):
                return {"sent": False}
        elif val == self.state["last"]:
            return {"sent": False}
        self.state["last"] = val
        # only the template's own \n \r \t are escapes; the value is sent as it is (Turkish letters, C:\new)
        tpl = self.v("template").replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        msg = tpl.replace("{v}", str(val))
        self.state["ser"].write(msg.encode("utf-8"))
        return {"sent": True}

    def teardown(self):
        if self.state.get("ser"):
            self.state["ser"].close()


@register
class VideoRecorder(Node):
    title = "Video Kaydet"
    category = CAT
    description = ("Görüntüyü MP4 olarak kaydeder. 'record' girişi bağlıysa sadece doğruyken kaydeder; "
                   "her kayıt ayrı dosyaya gider (olay kaydı).")
    inputs = [("image", PortType.IMAGE), ("record", PortType.BOOL)]
    outputs = [("recording", PortType.BOOL), ("path", PortType.TEXT)]
    params = [Param("folder", "str", "videolar", "Klasör"), Param("fps", "int", 20, "FPS", 1, 120)]

    def setup(self):
        self.state.update(writer=None, path="")

    def _close(self):
        if self.state.get("writer") is not None:
            self.state["writer"].release()
            self.state["writer"] = None

    def process(self, inputs):
        img, rec = inputs["image"], inputs["record"]
        active = img is not None and (rec is None or bool(rec))
        if not active:
            self._close()
            return {"recording": False, "path": self.state["path"]}
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.state["writer"] is None:
            os.makedirs(self.v("folder"), exist_ok=True)
            path = os.path.join(self.v("folder"), dt.datetime.now().strftime("kayit_%Y%m%d_%H%M%S_%f.mp4"))
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.v("fps"), (w, h))
            if not writer.isOpened():
                writer.release()
                hint = (" Windows'ta klasör yolunda Türkçe harf (ü, ş, ı…) olmamalı, ör. C:\\Kayitlar."
                        if not path.isascii() else "")
                raise RuntimeError(f"Video dosyası açılamadı: {path}.{hint}")
            self.state["writer"] = writer
            self.state.update(path=path, size=(w, h))
        if (img.shape[1], img.shape[0]) != self.state["size"]:
            img = cv2.resize(img, self.state["size"])
        self.state["writer"].write(img)
        return {"recording": True, "path": self.state["path"]}

    def teardown(self):
        self._close()


S7_TYPES = ["bool", "int", "dint", "real", "byte", "word"]


def _s7_size(kind: str) -> int:
    return {"bool": 1, "byte": 1, "int": 2, "word": 2, "dint": 4, "real": 4}[kind]


class _S7Base(Node):
    setup_params = ("host", "port", "rack", "slot")   # read only when the block starts: changing them restarts it
    params = [Param("host", "str", "192.168.0.1", "PLC IP"),
              Param("rack", "int", 0, "Rack", 0, 7), Param("slot", "int", 1, "Slot (S7-1200/1500: 1)", 0, 31),
              Param("db", "int", 1, "DB numarası", 1, 65535),
              Param("offset", "int", 0, "Bayt adresi", 0, 65535),
              Param("bit", "int", 0, "Bit (bool için)", 0, 7),
              Param("kind", "choice", "bool", "Tip", choices=S7_TYPES),
              Param("port", "int", 102, "Port", 1, 65535)]

    def setup(self):
        try:
            import snap7
        except ImportError as exc:
            raise RuntimeError("Siemens S7 için 'pip install python-snap7' kurun") from exc
        client = snap7.client.Client()
        client.connect(self.v("host"), self.v("rack"), self.v("slot"), self.v("port"))
        self.state.update(client=client, last=object())

    def teardown(self):
        c = self.state.get("client")
        if c is not None:
            try:
                c.disconnect()
            except Exception:  # noqa: BLE001
                pass


@register
class S7Write(_S7Base):
    title = "Siemens S7 Yaz"
    category = CAT
    description = ("S7-300/400/1200/1500 PLC'de bir DB adresine yazar (değer değişince). 1200/1500'de DB için "
                   "'Optimize erişim' kapalı ve PUT/GET izni açık olmalı.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]

    def process(self, inputs):
        from snap7 import util

        val = inputs["value"]
        if val is None:
            return {"ok": False}
        kind, c = self.v("kind"), self.state["client"]
        if val == self.state["last"]:
            return {"ok": True}
        db, off = self.v("db"), self.v("offset")
        buf = c.db_read(db, off, _s7_size(kind))
        if kind == "bool":
            util.set_bool(buf, 0, self.v("bit"), bool(val))
        else:
            {"int": util.set_int, "dint": util.set_dint, "real": util.set_real,
             "byte": util.set_byte, "word": util.set_word}[kind](buf, 0, float(val) if kind == "real" else int(val))
        c.db_write(db, off, buf)
        self.state["last"] = val
        return {"ok": True}


@register
class S7Read(_S7Base):
    title = "Siemens S7 Oku"
    category = "1 · Giriş"
    description = "S7 PLC'de bir DB adresini okur (ürün kodu, bant durumu, reçete numarası)."
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL)]

    def process(self, inputs):
        from snap7 import util

        kind = self.v("kind")
        buf = self.state["client"].db_read(self.v("db"), self.v("offset"), _s7_size(kind))
        if kind == "bool":
            v = util.get_bool(buf, 0, self.v("bit"))
        else:
            v = {"int": util.get_int, "dint": util.get_dint, "real": util.get_real,
                 "byte": util.get_byte, "word": util.get_word}[kind](buf, 0)
        v = round(float(v), 4) if kind == "real" else int(v)
        return {"value": v, "bool": bool(v)}


class _OpcBase(Node):
    # read only when the block starts: changing them restarts it
    setup_params = ("node", "password", "url", "user", "security", "trust")
    params = [Param("url", "str", "opc.tcp://192.168.0.1:4840", "Sunucu adresi"),
              Param("node", "str", "ns=2;s=Makine.Sayac", "Node ID"),
              Param("user", "str", "", "Kullanıcı (opsiyonel)"),
              Param("password", "str", "", "Şifre"),
              Param("security", "choice", "Yok", "Güvenlik (imza / şifreleme)",
                    choices=["Yok", "Basic256Sha256 · İmzala", "Basic256Sha256 · İmzala ve Şifrele"]),
              Param("trust", "choice", "İlk bağlantıda güven", "Sunucu sertifikası",
                    choices=["İlk bağlantıda güven", "Sadece güvenilen listesi"])]

    def setup(self):
        try:
            from asyncua.sync import Client
        except ImportError as exc:
            raise RuntimeError("OPC UA için 'pip install asyncua' kurun") from exc
        from .. import opcua_security as sec
        security = sec.check_choice(self.v("security"))
        client = Client(self.v("url"), timeout=4)
        try:
            if self.v("user"):
                client.set_user(self.v("user"))
                client.set_password(self.v("password"))
            sec.secure_client(client, security, self.v("trust"))
            try:
                client.connect()
            except Exception as exc:  # noqa: BLE001
                msg = sec.friendly_error(exc, security)
                if msg is None:
                    raise
                raise RuntimeError(msg) from None
        except BaseException:
            try:
                client.disconnect()       # stops the client's own thread too
            except Exception:  # noqa: BLE001
                pass
            raise
        self.state.update(client=client, node=client.get_node(self.v("node")), last=object())

    def teardown(self):
        c = self.state.get("client")
        if c is not None:
            try:
                c.disconnect()
            except Exception:  # noqa: BLE001
                pass


@register
class OpcUaWrite(_OpcBase):
    title = "OPC UA Yaz"
    category = CAT
    description = "Bir OPC UA değişkenine değer yazar (değer değişince). Hemen her modern PLC/SCADA destekler."
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]

    def process(self, inputs):
        from asyncua import ua

        val = inputs["value"]
        if val is None:
            return {"ok": False}
        if val == self.state["last"]:
            return {"ok": True}
        node = self.state["node"]
        vtype = node.read_data_type_as_variant_type()
        if vtype == ua.VariantType.Boolean:
            val = bool(val)
        elif vtype in (ua.VariantType.Float, ua.VariantType.Double):
            val = float(val)
        elif vtype == ua.VariantType.String:
            val = str(val)
        else:
            val = int(val)
        node.write_value(ua.DataValue(ua.Variant(val, vtype)))
        self.state["last"] = inputs["value"]
        return {"ok": True}


@register
class OpcUaRead(_OpcBase):
    title = "OPC UA Oku"
    category = "1 · Giriş"
    description = "Bir OPC UA değişkenini okur."
    outputs = [("value", PortType.ANY), ("number", PortType.NUMBER), ("bool", PortType.BOOL)]

    def process(self, inputs):
        v = self.state["node"].read_value()
        try:
            num = float(v)
        except (TypeError, ValueError):
            num = 0.0
        return {"value": v, "number": num, "bool": bool(v)}
