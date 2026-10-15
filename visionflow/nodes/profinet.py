"""Profinet DCP (Discovery and Configuration Protocol): find Profinet devices on the network and, with explicit
confirmation, give one a station name / IP address.

A Profinet IO controller or device (cyclic real-time IO data) cannot be done in pure Python: it needs a GSDML-based
stack with real-time timing. For process data with a Profinet PLC use the Siemens S7 blocks (S7-1200/1500 also talk
S7/PUT-GET over the same cable), OPC UA / Modbus TCP, or a PN/PN coupler / Profinet gateway.

DCP is plain Ethernet (EtherType 0x8892) and needs raw sockets: Linux AF_PACKET with root or CAP_NET_RAW; on
Windows (and macOS) the optional scapy library with Npcap. The frame building and parsing below are pure functions.
"""
from __future__ import annotations

import ipaddress
import os
import queue
import random
import re
import socket
import struct
import sys
import time

from ..core import Node, Param, PortType, register

CAT_IN, CAT_OUT = "1 · Giriş", "5 · Çıkış ve Otomasyon"
ETH_P_PN = 0x8892
ETH_P_VLAN = 0x8100
DCP_MULTICAST = bytes.fromhex("010ecf000000")
FRAME_IDENTIFY_REQ, FRAME_IDENTIFY_RES, FRAME_GETSET = 0xFEFE, 0xFEFF, 0xFEFD
SERVICE_SET, SERVICE_IDENTIFY = 0x04, 0x05
OPT_IP, OPT_DEVICE, OPT_CONTROL, OPT_ALL = 0x01, 0x02, 0x05, 0xFF
ROLES = {1: "IO-Device", 2: "IO-Controller", 4: "IO-Multidevice", 8: "PN-Supervisor"}
SET_ERRORS = {1: "seçenek desteklenmiyor", 2: "alt seçenek desteklenmiyor", 3: "alt seçenek ayarlanmadı",
              4: "kaynak hatası", 5: "cihaz yerel nedenle izin vermiyor", 6: "cihaz çalışırken (bağlantı varken) "
              "değiştirilemez"}
DATA_HELP = ("Profinet PLC ile veri alışverişi için: Siemens S7 Oku/Yaz blokları (S7-1200/1500'de PUT/GET izni), "
             "OPC UA veya Modbus TCP, ya da bir PN/PN coupler / Profinet ağ geçidi kullanın.")


# --- pure helpers ---------------------------------------------------------------------------------------------

def parse_mac(text) -> bytes:
    if isinstance(text, (bytes, bytearray)) and len(text) == 6:
        return bytes(text)
    s = re.sub(r"[\s:\-.]", "", str(text or "")).lower()
    if not re.fullmatch(r"[0-9a-f]{12}", s):
        raise ValueError(f"Geçersiz MAC adresi '{text}' (ör. 00:1B:1B:12:34:56)")
    return bytes.fromhex(s)


def format_mac(mac: bytes) -> str:
    return ":".join(f"{b:02X}" for b in mac)


def check_station_name(name: str) -> str:
    """The name in Profinet form (lower case) or ValueError: labels of a-z, 0-9, '-', separated by '.',
    1…63 characters each, 240 in total, not shaped like an IP address or 'port-xyz'."""
    n = str(name or "").strip().lower()
    if not n:
        raise ValueError("İstasyon adı boş olamaz")
    if len(n) > 240:
        raise ValueError(f"İstasyon adı en çok 240 karakter olabilir ({len(n)})")
    for label in n.split("."):
        if not 1 <= len(label) <= 63:
            raise ValueError("İstasyon adında her bölüm ('.' arası) 1…63 karakter olmalı")
        if not re.fullmatch(r"[a-z0-9-]+", label):
            raise ValueError(f"İstasyon adında yalnız a-z, 0-9, '-' ve '.' olabilir ('{label}'); Türkçe harf ve "
                             "boşluk kullanılamaz")
        if label[0] == "-" or label[-1] == "-":
            raise ValueError("İstasyon adı bölümleri '-' ile başlayamaz veya bitemez")
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", n):
        raise ValueError("İstasyon adı IP adresi biçiminde olamaz")
    if re.match(r"port-\d{3}(-\d{5})?($|\.)", n):
        raise ValueError("İstasyon adı 'port-xyz' biçiminde başlayamaz (port adları için ayrılmış)")
    return n


def _ip4(text, what: str) -> bytes:
    try:
        return ipaddress.IPv4Address(str(text).strip()).packed
    except ValueError:
        raise ValueError(f"Geçersiz {what}: '{text}'") from None


def _header(dst: bytes, src: bytes) -> bytes:
    return parse_mac(dst) + parse_mac(src) + struct.pack(">H", ETH_P_PN)


def _pad(frame: bytes) -> bytes:
    return frame + bytes(max(0, 60 - len(frame)))           # Ethernet minimum frame (without FCS)


def build_identify(src_mac, xid: int, response_delay: int = 1) -> bytes:
    """DCP Identify-All request (multicast 01:0E:CF:00:00:00)."""
    blocks = struct.pack(">BBH", OPT_ALL, OPT_ALL, 0)
    pdu = struct.pack(">HBBIHH", FRAME_IDENTIFY_REQ, SERVICE_IDENTIFY, 0, xid & 0xFFFFFFFF,
                      max(1, min(0x1900, int(response_delay))), len(blocks)) + blocks
    return _pad(_header(DCP_MULTICAST, src_mac) + pdu)


def _block(opt: int, sub: int, payload: bytes) -> bytes:
    b = struct.pack(">BBH", opt, sub, len(payload)) + payload
    return b + (b"\x00" if len(payload) % 2 else b"")


def build_set(src_mac, dst_mac, xid: int, name: str | None = None, ip: str | None = None, mask: str = "255.255.255.0",
              gateway: str = "0.0.0.0", permanent: bool = True) -> bytes:
    """DCP Set request (unicast) giving the device a station name and/or an IP address."""
    if not name and not ip:
        raise ValueError("Atanacak istasyon adı veya IP adresi yok")
    qual = struct.pack(">H", 1 if permanent else 0)
    blocks = b""
    if name:
        blocks += _block(OPT_DEVICE, 0x02, qual + check_station_name(name).encode("ascii"))
    if ip:
        blocks += _block(OPT_IP, 0x02, qual + _ip4(ip, "IP adresi") + _ip4(mask, "alt ağ maskesi")
                         + _ip4(gateway or "0.0.0.0", "ağ geçidi"))
    pdu = struct.pack(">HBBIHH", FRAME_GETSET, SERVICE_SET, 0, xid & 0xFFFFFFFF, 0, len(blocks)) + blocks
    return _pad(_header(dst_mac, src_mac) + pdu)


def _split(frame: bytes):
    """(src MAC, PN payload) of a Profinet frame, VLAN tag skipped; None for anything else."""
    if frame is None or len(frame) < 14:
        return None
    etype = struct.unpack_from(">H", frame, 12)[0]
    off = 14
    if etype == ETH_P_VLAN:
        if len(frame) < 18:
            return None
        etype = struct.unpack_from(">H", frame, 16)[0]
        off = 18
    if etype != ETH_P_PN:
        return None
    return bytes(frame[6:12]), bytes(frame[off:])


def _blocks(data: bytes):
    i = 0
    while i + 4 <= len(data):
        opt, sub, length = struct.unpack_from(">BBH", data, i)
        body = data[i + 4:i + 4 + length]
        if len(body) < length:
            return                                            # truncated block: stop, keep what was parsed
        yield opt, sub, body
        i += 4 + length + (length & 1)


def _text(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("latin-1").strip()


def parse_identify_response(frame: bytes, xid: int | None = None) -> dict | None:
    """The device described by a DCP Identify response, or None when ``frame`` is not one (other traffic, other
    transaction, broken frame)."""
    split = _split(frame)
    if split is None:
        return None
    src, pn = split
    if len(pn) < 12:
        return None
    frame_id, service, stype, rxid, _resv, length = struct.unpack_from(">HBBIHH", pn, 0)
    if frame_id != FRAME_IDENTIFY_RES or service != SERVICE_IDENTIFY or stype & 0x7F != 1:
        return None
    if xid is not None and rxid != xid & 0xFFFFFFFF:
        return None
    dev = {"mac": format_mac(src), "name": "", "type": "", "vendor_id": None, "device_id": None, "role": "",
           "ip": "", "mask": "", "gateway": "", "alias": ""}
    for opt, sub, body in _blocks(pn[12:12 + length]):
        data = body[2:]                                      # after BlockInfo
        if opt == OPT_DEVICE and sub == 0x01:
            dev["type"] = _text(data)
        elif opt == OPT_DEVICE and sub == 0x02:
            dev["name"] = _text(data)
        elif opt == OPT_DEVICE and sub == 0x03 and len(data) >= 4:
            dev["vendor_id"], dev["device_id"] = struct.unpack_from(">HH", data, 0)
        elif opt == OPT_DEVICE and sub == 0x04 and data:
            dev["role"] = ", ".join(v for k, v in ROLES.items() if data[0] & k)
        elif opt == OPT_DEVICE and sub == 0x06:
            dev["alias"] = _text(data)
        elif opt == OPT_IP and sub == 0x02 and len(data) >= 12:
            dev["ip"], dev["mask"], dev["gateway"] = (str(ipaddress.IPv4Address(data[k:k + 4])) for k in (0, 4, 8))
        elif opt == OPT_IP and sub == 0x01 and len(data) >= 6:
            dev["mac"] = format_mac(data[:6])
    return dev


def parse_set_response(frame: bytes, xid: int, device_mac=None):
    """None when ``frame`` is not the answer to Set ``xid``; else a list of (option, suboption, error code)."""
    split = _split(frame)
    if split is None:
        return None
    src, pn = split
    if device_mac is not None and src != parse_mac(device_mac):
        return None
    if len(pn) < 12:
        return None
    frame_id, service, stype, rxid, _resv, length = struct.unpack_from(">HBBIHH", pn, 0)
    if frame_id != FRAME_GETSET or service != SERVICE_SET or rxid != xid & 0xFFFFFFFF:
        return None
    result = []
    for opt, sub, body in _blocks(pn[12:12 + length]):
        if opt == OPT_CONTROL and sub == 0x04 and len(body) >= 3:
            result.append((body[0], body[1], body[2]))
    if stype & 0x7F != 1 and not result:
        result.append((0, 0, 5))
    return result


def format_devices(devs: list[dict]) -> str:
    if not devs:
        return "Profinet cihazı bulunamadı"
    lines = []
    for d in devs:
        ids = (f"üretici 0x{d['vendor_id']:04X} cihaz 0x{d['device_id']:04X}" if d["vendor_id"] is not None
               else "kimlik yok")
        lines.append(f"{d['name'] or '(adsız)'}  IP {d['ip'] or '-'}  MAC {d['mac']}  {ids}"
                     + (f"  {d['type']}" if d["type"] else "") + (f"  [{d['role']}]" if d["role"] else ""))
    return "\n".join(lines)


# --- raw Ethernet access --------------------------------------------------------------------------------------

def list_interfaces() -> list[str]:
    if sys.platform.startswith("linux"):
        try:
            return sorted(n for n in os.listdir("/sys/class/net") if n != "lo")
        except OSError:
            return []
    try:
        from scapy.all import get_if_list
        return list(get_if_list())
    except Exception:  # noqa: BLE001
        return []


def _iface_hint() -> str:
    names = list_interfaces()
    return f" Ağ kartları: {', '.join(names)}" if names else ""


class LinuxTransport:
    def __init__(self, iface: str):
        try:
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_PN))
        except PermissionError as exc:
            raise PermissionError("Profinet keşfi ham Ethernet ister: programı root olarak ya da CAP_NET_RAW "
                                  "yetkisiyle çalıştırın (sudo setcap cap_net_raw+ep <python>)") from exc
        try:
            self.sock.bind((iface, ETH_P_PN))
        except OSError as exc:
            self.sock.close()
            raise ConnectionError(f"Ağ kartı '{iface}' açılamadı ({exc})." + _iface_hint()) from exc
        self.mac = bytes(self.sock.getsockname()[4][:6])

    def send(self, frame: bytes):
        self.sock.send(frame)

    def recv(self, timeout: float):
        self.sock.settimeout(max(0.001, timeout))
        try:
            return self.sock.recv(1600)
        except (socket.timeout, BlockingIOError):
            return None

    def close(self):
        self.sock.close()


class ScapyTransport:
    def __init__(self, iface: str):
        try:
            from scapy.all import AsyncSniffer, get_if_hwaddr, sendp
        except ImportError as exc:
            raise RuntimeError("Bu sistemde Profinet keşfi için Npcap kurun ve 'pip install scapy' yükleyin "
                               "(Linux'ta gerekmez)") from exc
        self._sendp, self.iface, self.q = sendp, iface, queue.Queue()
        try:
            self.mac = parse_mac(get_if_hwaddr(iface))
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(f"Ağ kartı '{iface}' açılamadı ({exc})." + _iface_hint()) from exc
        self.sniffer = AsyncSniffer(iface=iface, filter="ether proto 0x8892", store=False,
                                    prn=lambda p: self.q.put(bytes(p)))
        self.sniffer.start()
        time.sleep(0.2)                                      # the capture must run before the request goes out

    def send(self, frame: bytes):
        from scapy.all import Raw
        self._sendp(Raw(load=frame), iface=self.iface, verbose=False)

    def recv(self, timeout: float):
        try:
            return self.q.get(timeout=max(0.001, timeout))
        except queue.Empty:
            return None

    def close(self):
        try:
            self.sniffer.stop()
        except Exception:  # noqa: BLE001
            pass


def open_transport(iface: str):
    iface = (iface or "").strip()
    if not iface:
        raise ValueError("Ağ kartı adı boş." + _iface_hint())
    if sys.platform.startswith("linux") and hasattr(socket, "AF_PACKET"):
        return LinuxTransport(iface)
    return ScapyTransport(iface)


def discover(iface: str, timeout: float = 2.0, transport=None) -> list[dict]:
    """Every Profinet device that answers an Identify-All within ``timeout`` seconds, sorted by station name."""
    t = transport or open_transport(iface)
    try:
        xid = random.randint(1, 0xFFFFFFFF)
        t.send(build_identify(t.mac, xid, response_delay=int(timeout * 50)))
        found: dict[str, dict] = {}
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            frame = t.recv(left)
            if frame is None:
                continue
            dev = parse_identify_response(frame, xid)
            if dev is not None:
                found[dev["mac"]] = dev
        return sorted(found.values(), key=lambda d: (d["name"], d["mac"]))
    finally:
        if transport is None:
            t.close()


def set_device(iface: str, mac, name: str | None = None, ip: str | None = None, mask: str = "255.255.255.0",
               gateway: str = "0.0.0.0", permanent: bool = True, timeout: float = 2.0, transport=None) -> str:
    """Sends a DCP Set to the device ``mac``; returns a Turkish success message or raises with the device's error."""
    dst = parse_mac(mac)
    xid = random.randint(1, 0xFFFFFFFF)
    t = transport or open_transport(iface)
    try:
        frame = build_set(t.mac, dst, xid, name, ip, mask, gateway, permanent)
        t.send(frame)
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            res = parse_set_response(t.recv(left), xid, dst)
            if res is None:
                continue
            errors = [(o, s, e) for o, s, e in res if e]
            if errors:
                raise RuntimeError("Cihaz ayarı reddetti: " + "; ".join(
                    f"seçenek {o}/{s}: {SET_ERRORS.get(e, f'hata {e}')}" for o, s, e in errors))
            parts = ([f"ad '{check_station_name(name)}'"] if name else []) + ([f"IP {ip}"] if ip else [])
            return f"{format_mac(dst)}: {' ve '.join(parts)} atandı" + ("" if permanent else " (geçici)")
        raise TimeoutError(f"{format_mac(dst)} cevap vermedi (MAC adresini ve kabloyu kontrol edin)")
    finally:
        if transport is None:
            t.close()


# --- blocks ---------------------------------------------------------------------------------------------------

IFACE = Param("iface", "str", "", "Ağ kartı (Linux: eth0, Windows: Ethernet)")


@register
class ProfinetDiscover(Node):
    title = "Profinet Cihazlarını Bul"
    category = CAT_IN
    description = ("Ağdaki Profinet cihazlarını DCP ile bulur (TIA Portal'daki 'Erişilebilir cihazlar' gibi): "
                   "istasyon adı, IP, MAC, üretici/cihaz kimliği, cihaz tipi. Akış başına bir kez tarar; 'tara' girişi "
                   "yükselince yeniden tarar. Linux'ta root/CAP_NET_RAW, Windows'ta Npcap + 'pip install scapy' "
                   "gerekir. Bu blok veri alışverişi yapmaz: " + DATA_HELP)
    inputs = [("scan", PortType.BOOL)]
    outputs = [("text", PortType.TEXT), ("count", PortType.NUMBER), ("names", PortType.TEXT)]
    params = [IFACE, Param("timeout_ms", "int", 2000, "Bekleme süresi (ms)", 200, 10000)]
    transport = None                       # tests put a fake raw-Ethernet transport here

    def setup(self):
        self.state.update(result=None, prev=False)

    def process(self, inputs):
        trig = bool(inputs.get("scan"))
        if self.state.get("result") is None or (trig and not self.state.get("prev")):
            devs = discover(self.v("iface"), self.v("timeout_ms") / 1000.0, transport=type(self).transport)
            self.state["result"] = (format_devices(devs), len(devs), ", ".join(d["name"] for d in devs if d["name"]))
        self.state["prev"] = trig
        text, count, names = self.state["result"]
        return {"text": text, "count": count, "names": names}


@register
class ProfinetSetName(Node):
    title = "Profinet İstasyon Adı Ata"
    category = CAT_OUT
    description = ("Bir Profinet cihazına (MAC adresiyle) DCP ile istasyon adı ve isteğe bağlı IP adresi atar. "
                   "Yalnız 'Onaylıyorum' işaretliyken ve 'uygula' girişi yükseldiğinde bir kez gönderir. DİKKAT: "
                   "çalışan bir hatta yanlış cihazın adını değiştirmek PLC bağlantısını koparır. " + DATA_HELP)
    inputs = [("apply", PortType.BOOL)]
    outputs = [("ok", PortType.BOOL), ("text", PortType.TEXT)]
    params = [IFACE, Param("mac", "str", "", "Cihaz MAC adresi"),
              Param("name", "str", "", "Yeni istasyon adı (boş: değiştirme)"),
              Param("ip", "str", "", "Yeni IP (boş: değiştirme)"),
              Param("mask", "str", "255.255.255.0", "Alt ağ maskesi"),
              Param("gateway", "str", "0.0.0.0", "Ağ geçidi"),
              Param("permanent", "bool", True, "Kalıcı (cihaz yeniden başlayınca da kalsın)"),
              Param("confirm", "bool", False, "Onaylıyorum: bu cihazın ayarını değiştir")]
    transport = None

    def setup(self):
        self.state.update(prev=False, ok=False, text="Bekliyor ('uygula' girişi)")

    def process(self, inputs):
        trig = bool(inputs.get("apply"))
        rising = trig and not self.state.get("prev")
        self.state["prev"] = trig
        if rising:
            if not self.v("confirm"):
                raise ValueError("Güvenlik için 'Onaylıyorum' ayarını işaretleyin; işaretsizken hiçbir şey "
                                 "gönderilmez")
            name, ip = self.v("name").strip(), self.v("ip").strip()
            parse_mac(self.v("mac"))
            if name:
                check_station_name(name)
            if not name and not ip:
                raise ValueError("Yeni istasyon adı veya IP adresi girin")
            self.state["ok"] = False
            self.state["text"] = set_device(self.v("iface"), self.v("mac"), name or None, ip or None,
                                            self.v("mask"), self.v("gateway"), self.v("permanent"),
                                            transport=type(self).transport)
            self.state["ok"] = True
        return {"ok": self.state["ok"], "text": self.state["text"]}
