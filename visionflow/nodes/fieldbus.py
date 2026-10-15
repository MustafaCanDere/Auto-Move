"""More PLC connections: Modbus RTU over RS-485/RS-232 (serial port) and Allen-Bradley over EtherNet/IP.

Modbus RTU is written here without extra packages except pyserial for the port itself; several blocks that use
the same port (one RS-485 bus with many devices) share one connection and take turns on it.
"""
from __future__ import annotations

import struct
import threading
import time

from ..core import Node, Param, PortType, register

CAT_IN, CAT_OUT = "1 · Giriş", "5 · Çıkış ve Otomasyon"
BAUDS = ["9600", "19200", "38400", "57600", "115200", "4800", "2400"]


def crc16(data: bytes) -> int:
    """Modbus RTU CRC (poly 0xA001, start 0xFFFF); sent low byte first."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


class _SharedPort:
    """One open serial port shared by every block on that bus, with a lock so requests never interleave."""
    _ports: dict = {}
    _guard = threading.Lock()

    def __init__(self, ser):
        self.ser, self.lock, self.users = ser, threading.Lock(), 0

    @classmethod
    def acquire(cls, port: str, baud: int, parity: str, timeout: float, factory=None) -> "_SharedPort":
        port = port.strip()
        key = port.upper() if port.upper().startswith("COM") else port
        with cls._guard:
            sp = cls._ports.get(key)
            if sp is not None and sp.settings != (baud, parity):
                # one bus has one speed: the newest setting wins (e.g. changed while the flow runs), so the
                # other blocks on this port talk at the new speed too instead of this block failing to start
                with sp.lock:
                    sp.ser.baudrate, sp.ser.parity = baud, parity
                    sp.settings = (baud, parity)
                print(f"Modbus RTU: {port} hızı/paritesi {baud} {parity} olarak değiştirildi "
                      "(bu porttaki bütün bloklar için)")
            if sp is None:
                if factory is None:
                    try:
                        import serial
                    except ImportError as exc:
                        raise RuntimeError("Modbus RTU için 'pip install pyserial' kurun") from exc
                    factory = serial.Serial
                ser = factory(port=port, baudrate=baud, parity=parity, bytesize=8, stopbits=1, timeout=timeout)
                sp = cls._ports[key] = cls(ser)
                sp.settings, sp.key = (baud, parity), key
            sp.users += 1
            return sp

    def release(self):
        with self._guard:
            self.users -= 1
            if self.users <= 0:
                self._ports.pop(self.key, None)
                try:
                    self.ser.close()
                except Exception:  # noqa: BLE001
                    pass


class ModbusRtuClient:
    """Modbus RTU master: same calls as the TCP ModbusClient (read/write coils and holding registers)."""

    def __init__(self, port: str, baud: int = 9600, parity: str = "N", unit: int = 1, timeout: float = 0.5,
                 factory=None):
        self.unit, self.timeout = unit, timeout
        self.port = _SharedPort.acquire(port, baud, parity, timeout, factory)

    def _read_exact(self, n: int) -> bytes:
        ser, buf = self.port.ser, b""
        deadline = time.monotonic() + self.timeout + 0.05
        while len(buf) < n:
            chunk = ser.read(n - len(buf))
            if chunk:
                buf += chunk
            elif time.monotonic() > deadline:
                raise TimeoutError(f"Cihaz {self.unit} cevap vermedi (adres, hız ve kabloyu kontrol edin)")
        return buf

    def _flush(self):
        try:
            self.port.ser.reset_input_buffer()
        except Exception:  # noqa: BLE001
            pass

    def request(self, pdu: bytes) -> bytes:
        frame = bytes([self.unit]) + pdu
        frame += struct.pack("<H", crc16(frame))
        with self.port.lock:
            self._flush()                           # leftovers of an earlier answer
            # RTU needs 3.5 characters of silence between frames
            time.sleep(max(0.00175, 3.5 * 11 / self.port.settings[0]))
            self.port.ser.write(frame)
            try:
                body, crc, fc = self._receive()
            except Exception:
                # a slow device may still answer this request: wait it out and throw that answer away, or it
                # would be taken as the answer to the next request (a wrong value without any error)
                time.sleep(self.timeout)
                self._flush()
                raise
        if struct.unpack("<H", crc)[0] != crc16(body):
            raise RuntimeError("Modbus RTU cevabında CRC hatası (kablo, sonlandırma direnci veya parazit)")
        if fc != pdu[0] and not (fc & 0x80 and fc & 0x7F == pdu[0]):
            raise RuntimeError(f"Beklenmeyen fonksiyon kodu {fc}")
        if fc & 0x80:
            raise RuntimeError(f"Modbus hata kodu {body[2]}")
        if fc in (5, 6) and body[2:6] != pdu[1:5]:
            raise RuntimeError("Cihaz yazılan adresi/değeri farklı onayladı")
        return body[1:]

    def _receive(self):
        head = self._read_exact(2)
        if head[0] != self.unit:
            raise RuntimeError(f"Yanlış cihazdan cevap geldi ({head[0]}, beklenen {self.unit})")
        fc = head[1]
        if fc & 0x80:
            rest = self._read_exact(3)
            body = head + rest[:1]
            crc = rest[1:]
        elif fc in (1, 2, 3, 4):
            count = self._read_exact(1)
            data = self._read_exact(count[0] + 2)
            body, crc = head + count + data[:-2], data[-2:]
        else:                                   # 5, 6, 15, 16: address + value echoed
            data = self._read_exact(6)
            body, crc = head + data[:4], data[4:]
        return body, crc, fc

    def write_coil(self, addr: int, value: bool):
        self.request(struct.pack(">BHH", 5, addr, 0xFF00 if value else 0))

    def write_register(self, addr: int, value: int):
        self.request(struct.pack(">BHH", 6, addr, int(value) & 0xFFFF))

    def read_registers(self, addr: int, count: int = 1) -> list[int]:
        resp = self.request(struct.pack(">BHH", 3, addr, count))
        if resp[1] != 2 * count or len(resp) < 2 + 2 * count:
            raise RuntimeError("Modbus RTU cevabı eksik")
        return list(struct.unpack(f">{count}H", resp[2:2 + 2 * count]))

    def read_coils(self, addr: int, count: int = 1) -> list[bool]:
        resp = self.request(struct.pack(">BHH", 1, addr, count))
        if resp[1] != (count + 7) // 8:
            raise RuntimeError("Modbus RTU cevabı eksik")
        bits = int.from_bytes(resp[2:], "little")
        return [bool(bits >> i & 1) for i in range(count)]

    def close(self):
        if self.port is not None:
            self.port.release()
            self.port = None


RTU_PARAMS = [Param("port", "str", "COM3", "Seri port (COM3, /dev/ttyUSB0)"),
              Param("baud", "choice", "9600", "Hız (baud)", choices=BAUDS),
              Param("parity", "choice", "N", "Parite", choices=["N", "E", "O"]),
              Param("unit", "int", 1, "Cihaz adresi (slave ID)", 1, 247),
              Param("kind", "choice", "register", "Tip", choices=["register", "coil"]),
              Param("address", "int", 0, "Adres", 0, 65535)]


class _RtuBase(Node):
    setup_params = ("port", "baud", "parity", "unit")
    params = RTU_PARAMS
    serial_factory = None               # tests put a fake serial port here

    def setup(self):
        self.state["client"] = ModbusRtuClient(self.v("port"), int(self.v("baud")), self.v("parity"), self.v("unit"),
                                               factory=type(self).serial_factory)

    def teardown(self):
        if self.state.get("client"):
            self.state["client"].close()
            self.state["client"] = None


@register
class ModbusRtuRead(_RtuBase):
    title = "PLC Oku (Modbus RTU)"
    category = CAT_IN
    description = ("RS-485/RS-232 üzerinden Modbus RTU cihazlardan (PLC, sıcaklık kontrol cihazı, sürücü, "
                   "enerji analizörü) register veya bobin okur. Aynı hattaki birden çok cihaz için birden çok blok "
                   "kullanın; portu paylaşırlar. pyserial gerekir.")
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL)]

    def process(self, inputs):
        c = self.state["client"]
        v = c.read_coils(self.v("address"))[0] if self.v("kind") == "coil" else c.read_registers(self.v("address"))[0]
        return {"value": int(v), "bool": bool(v)}


@register
class ModbusRtuWrite(_RtuBase):
    title = "PLC Yaz (Modbus RTU)"
    category = CAT_OUT
    description = ("RS-485/RS-232 üzerinden Modbus RTU cihaza bobin veya register yazar (ör. hatalı ürün → "
                   "ayırıcı rölesi). Değer değiştiğinde gönderir. pyserial gerekir.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]

    def setup(self):
        super().setup()
        self.state["last"] = object()

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


# --- Allen-Bradley (EtherNet/IP, CIP) -----------------------------------------------------------------------------

AB_FAMILIES = ["ControlLogix / CompactLogix", "Micro800", "SLC 500 / MicroLogix"]


class _AbBase(Node):
    setup_params = ("host", "slot", "family")
    params = [Param("host", "str", "192.168.1.10", "PLC IP"),
              Param("family", "choice", AB_FAMILIES[0], "PLC ailesi", choices=AB_FAMILIES),
              Param("slot", "int", 0, "CPU slotu (ControlLogix)", 0, 16),
              Param("tag", "str", "Sayac", "Etiket (tag) adı; SLC/MicroLogix: N7:0, B3:0/1")]
    driver_module = None                # tests put a fake pycomm3 here

    def setup(self):
        mod = type(self).driver_module
        if mod is None:
            try:
                import pycomm3 as mod
            except ImportError as exc:
                raise RuntimeError("Allen-Bradley için 'pip install pycomm3' kurun") from exc
        family, host = self.v("family"), self.v("host").strip()
        if family.startswith("SLC"):
            plc = mod.SLCDriver(host)
        elif family == "Micro800":
            plc = mod.LogixDriver(host, init_tags=True, init_program_tags=False)
        else:
            plc = mod.LogixDriver(f"{host}/{self.v('slot')}")
        try:
            ok = plc.open()
        except Exception as exc:  # noqa: BLE001
            ok, why = False, exc
        else:
            why = None
        if not ok:
            try:                                 # free the PLC's connection slot: a failed start gets no teardown
                plc.close()
            except Exception:  # noqa: BLE001
                pass
            raise ConnectionError(f"{host} adresindeki PLC'ye bağlanılamadı" + (f": {why}" if why else ""))
        self.state["plc"] = plc

    def _check(self, tag):
        err = getattr(tag, "error", None)
        if err:
            raise RuntimeError(f"{self.v('tag')}: {err}")
        return tag

    def teardown(self):
        plc = self.state.pop("plc", None)
        if plc is not None:
            try:
                plc.close()
            except Exception:  # noqa: BLE001
                pass


@register
class AbRead(_AbBase):
    title = "Allen-Bradley Oku"
    category = CAT_IN
    description = ("Rockwell Allen-Bradley PLC'den (ControlLogix, CompactLogix, Micro800, SLC 500, MicroLogix) "
                   "EtherNet/IP ile etiket okur. pycomm3 gerekir.")
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL), ("text", PortType.TEXT)]

    def process(self, inputs):
        v = self._check(self.state["plc"].read(self.v("tag"))).value
        num = float(v) if isinstance(v, (int, float, bool)) else 0.0
        if num.is_integer():
            num = int(num)
        return {"value": num, "bool": bool(v), "text": "" if v is None else str(v)}


@register
class AbWrite(_AbBase):
    title = "Allen-Bradley Yaz"
    category = CAT_OUT
    description = ("Rockwell Allen-Bradley PLC'ye EtherNet/IP ile etiket yazar (ör. hatalı ürün sayısı, ayırıcı "
                   "komutu). Değer değiştiğinde gönderir. pycomm3 gerekir.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]

    def setup(self):
        super().setup()
        self.state["last"] = object()

    def process(self, inputs):
        val = inputs["value"]
        if val is None:
            return {"ok": False}
        if isinstance(val, float) and val.is_integer():
            val = int(val)                       # 3.0 from a counter is 3 for a DINT tag; a REAL tag takes 3 too
        sent = (self.v("tag"), type(val), val)
        if sent == self.state["last"]:
            return {"ok": True}
        self._check(self.state["plc"].write((self.v("tag"), val)))
        self.state["last"] = sent
        return {"ok": True}
