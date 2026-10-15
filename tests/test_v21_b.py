"""2.1 (b): EtherCAT master (pysoem, faked here), Profinet DCP discovery / set name, byte packing helpers.
No real EtherCAT or Profinet hardware is used: pysoem is replaced by a simulated bus and raw Ethernet by a fake
transport that answers with hand-built DCP frames."""
import math
import struct
import sys
import time
import types

import numpy as np
import pytest

from visionflow.core import create, load_builtin
from visionflow.nodes import ethercat as ec
from visionflow.nodes import profinet as pn
from visionflow.nodes._bytes import coerce_value, dtype_size, pack_value, unpack_value

load_builtin()


# --- byte helpers ---------------------------------------------------------------------------------------------

def test_unpack_every_type_little_and_big():
    buf = bytes([0x01, 0x02, 0x03, 0x04, 0xFF, 0xFF, 0xFF, 0xFF])
    assert unpack_value(buf, 0, "uint8") == 1
    assert unpack_value(buf, 4, "int8") == -1
    assert unpack_value(buf, 0, "uint16") == 0x0201
    assert unpack_value(buf, 0, "uint16", endian="big") == 0x0102
    assert unpack_value(buf, 4, "int16") == -1
    assert unpack_value(buf, 4, "uint16") == 0xFFFF
    assert unpack_value(buf, 0, "uint32") == 0x04030201
    assert unpack_value(buf, 0, "int32", endian="big") == 0x01020304
    assert unpack_value(buf, 4, "int32") == -1
    assert unpack_value(struct.pack("<f", 1.5), 0, "float32") == 1.5
    assert unpack_value(struct.pack(">f", -2.25), 0, "float32", endian="big") == -2.25
    assert unpack_value(bytearray([0b00001000]), 0, "bool", bit=3) is True
    assert unpack_value(bytearray([0b11110111]), 0, "bool", bit=3) is False


def test_pack_roundtrip_and_bits():
    for dtype, val in [("uint8", 255), ("int8", -128), ("uint16", 65535), ("int16", -32768), ("uint32", 2 ** 32 - 1),
                       ("int32", -2 ** 31), ("float32", 3.25)]:
        for endian in ("little", "big"):
            buf = bytearray(8)
            assert pack_value(buf, 2, dtype, val, endian=endian) == val
            assert unpack_value(buf, 2, dtype, endian=endian) == val
            assert buf[:2] == b"\x00\x00"                     # neighbours untouched
    buf = bytearray(b"\x00\xF0")
    pack_value(buf, 1, "bool", True, bit=0)
    pack_value(buf, 1, "bool", 0, bit=7)
    assert buf == bytearray(b"\x00\x71")
    assert pack_value(bytearray(4), 0, "int16", 0x1234, endian="big") == 0x1234
    b2 = bytearray(2)
    pack_value(b2, 0, "int16", 0x1234, endian="big")
    assert b2 == b"\x12\x34"
    pack_value(b2, 0, "int16", 0x1234)
    assert b2 == b"\x34\x12"


@pytest.mark.parametrize("offset, msg", [(-1, "negatif"), (7, "dışında"), (1.5, "tam sayı"), (True, "tam sayı"),
                                         ("2", "tam sayı")])
def test_bad_offsets(offset, msg):
    with pytest.raises(ValueError, match=msg):
        unpack_value(bytes(8), offset, "uint16")
    with pytest.raises(ValueError, match=msg):
        pack_value(bytearray(8), offset, "uint16", 1)


def test_offset_messages_and_bits():
    with pytest.raises(ValueError, match=r"alan 4 bayt \(geçerli adres 0…0\)"):
        unpack_value(bytes(4), 1, "int32")
    with pytest.raises(ValueError, match="hiç veri baytı yok"):
        unpack_value(b"", 0, "uint8")
    for bit in (-1, 8, 2.0, True):
        with pytest.raises(ValueError, match="Bit numarası"):
            unpack_value(bytes(1), 0, "bool", bit=bit)
    unpack_value(bytes(1), 0, "uint8", bit=99)               # bit only matters for bool
    with pytest.raises(ValueError, match="Bilinmeyen veri tipi"):
        unpack_value(bytes(4), 0, "double")
    with pytest.raises(ValueError, match="Bayt sırası"):
        unpack_value(bytes(4), 0, "int16", endian="middle")
    with pytest.raises(ValueError, match="Okunacak veri yok"):
        unpack_value(None, 0, "uint8")
    with pytest.raises(TypeError):
        pack_value(b"\x00\x00", 0, "uint8", 1)
    assert dtype_size("float32") == 4


@pytest.mark.parametrize("dtype, value", [("uint8", 256), ("uint8", -1), ("int8", 128), ("int8", -129),
                                          ("uint16", 65536), ("int16", 32768), ("int16", -32769),
                                          ("uint32", 2 ** 32), ("int32", 2 ** 31), ("int32", -2 ** 31 - 1)])
def test_out_of_range_values(dtype, value):
    buf = bytearray(b"\xAA" * 4)
    with pytest.raises(ValueError, match="aralığının dışında"):
        pack_value(buf, 0, dtype, value)
    assert buf == b"\xAA" * 4                               # nothing half-written


def test_value_coercion():
    assert coerce_value(2.0, "int16") == 2
    assert coerce_value(True, "uint8") == 1
    assert coerce_value("12", "int16") == 12
    assert coerce_value(" 0x10 ", "uint8") == 16
    assert coerce_value("08", "uint8") == 8
    assert coerce_value("1,5", "float32") == 1.5
    assert coerce_value("evet", "bool") is True and coerce_value("0", "bool") is False
    assert coerce_value(np.int64(7), "int32") == 7 and coerce_value(np.float32(0.5), "float32") == 0.5
    assert coerce_value(np.array([3]), "uint8") == 3
    assert coerce_value(float("inf"), "float32") == math.inf
    with pytest.raises(ValueError, match="tam sayı gerekli"):
        coerce_value(2.5, "int16")
    with pytest.raises(ValueError, match="tam sayı gerekli"):
        coerce_value(float("inf"), "int32")
    with pytest.raises(ValueError, match="NaN"):
        coerce_value(float("nan"), "float32")
    with pytest.raises(ValueError, match="float32 aralığının"):
        coerce_value(1e39, "float32")
    with pytest.raises(ValueError, match="bir sayı değil"):
        coerce_value("abc", "uint8")
    with pytest.raises(ValueError, match="değer yok"):
        coerce_value(None, "uint8")
    with pytest.raises(ValueError, match="dizi"):
        coerce_value(np.array([1, 2]), "uint8")
    with pytest.raises(ValueError, match="Sayı bekleniyor"):
        coerce_value([1], "uint8")


# --- fake pysoem ----------------------------------------------------------------------------------------------

class FakeSlave:
    def __init__(self, name, man, pid, nin, nout, loop=False):
        self.name, self.man, self.id, self.rev = name, man, pid, 1
        self._in, self._out = bytearray(nin), bytes(nout)
        self.loop, self.state, self.al_status, self.refuse_op = loop, 1, 0, False
        self.state_writes = []

    @property
    def input(self):
        return bytes(self._in)

    @property
    def output(self):
        return self._out

    @output.setter
    def output(self, value):
        if len(value) != len(self._out):
            raise ValueError("wrong size")
        self._out = bytes(value)

    def write_state(self):
        self.state_writes.append(self.state)
        if self.state & 0x10:                                # ACK clears the error
            self.state &= 0x0F


def make_pysoem():
    mod = types.ModuleType("pysoem")
    mod.NONE_STATE, mod.INIT_STATE, mod.PREOP_STATE, mod.SAFEOP_STATE, mod.OP_STATE = 0, 1, 2, 4, 8
    mod.STATE_ACK = mod.STATE_ERROR = 0x10
    mod.masters, mod.adapters = [], {"eth1": "Intel I210"}
    mod.layout = lambda: [FakeSlave("EL1008", 2, 0x03F03052, 1, 0), FakeSlave("EL2008", 2, 0x07D83052, 0, 1),
                          FakeSlave("EL4102", 2, 0x10063052, 4, 4, loop=True)]

    class Adapter:
        def __init__(self, name, desc):
            self.name, self.desc = name, desc

    class Master:
        def __init__(self):
            self.slaves, self.state, self.closed, self.opened = [], 0, False, None
            self.sent, self.wkc_ok, self.fail_cycle, self.state_log = [], True, False, []
            mod.masters.append(self)

        def open(self, ifname):
            if ifname not in mod.adapters:
                raise ConnectionError("could not open interface")
            self.opened = ifname

        def config_init(self):
            self.slaves = mod.layout()
            for s in self.slaves:
                s.state = mod.PREOP_STATE
            return len(self.slaves)

        def config_map(self):
            for s in self.slaves:
                s.state = mod.SAFEOP_STATE
            return 9

        @property
        def expected_wkc(self):
            return len(self.slaves)

        def state_check(self, expected, timeout=50000):
            # like SOEM: a broadcast read ORs the AL states of all slaves
            st = 0
            for s in self.slaves:
                st |= s.state
            return st & 0x0F

        def read_state(self):
            return min(s.state for s in self.slaves)

        def write_state(self):
            self.state_log.append(self.state)
            for s in self.slaves:
                if self.state == mod.OP_STATE and s.refuse_op:
                    s.state, s.al_status = mod.SAFEOP_STATE + mod.STATE_ERROR, 0x001D
                else:
                    s.state = self.state

        def send_processdata(self):
            if self.fail_cycle:
                raise OSError("adapter gone")
            self.sent.append([s.output for s in self.slaves])
            for s in self.slaves:
                if s.loop:
                    s._in[:] = s._out

        def receive_processdata(self, timeout=2000):
            return len(self.slaves) if self.wkc_ok else 0

        def close(self):
            self.closed = True

    mod.Master = Master
    mod.find_adapters = lambda: [Adapter(k, v) for k, v in mod.adapters.items()]
    return mod


@pytest.fixture
def soem(monkeypatch):
    mod = make_pysoem()
    monkeypatch.setitem(sys.modules, "pysoem", mod)
    monkeypatch.setattr(ec, "OP_TIMEOUT", 0.3)
    yield mod
    for m in list(ec.EcatMaster._masters.values()):       # never leave a thread running after a failed test
        m.users = 1
        m.release()
    assert not ec.EcatMaster._masters


def node(type_name, **values):
    n = create(type_name)
    n.values.update(values)
    assert n.safe_setup(), n.setup_error
    return n


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_missing_pysoem_gives_install_help(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysoem", None)
    n = create("EtherCatRead")
    n.values["adapter"] = "eth1"
    assert not n.safe_setup()
    assert "pip install pysoem" in n.setup_error and "Npcap" in n.setup_error and "CAP_NET_RAW" in n.setup_error
    n.teardown()
    scan = create("EtherCatScan")
    assert scan.safe_setup()
    with pytest.raises(RuntimeError, match="pip install pysoem"):
        scan.process({"scan": None})


def test_shared_master_reaches_op_and_exchanges_data(soem):
    r = node("EtherCatRead", adapter="eth1", slave=3, offset=0, dtype="int32", cycle_ms=2)
    w = node("EtherCatWrite", adapter="eth1", slave=3, offset=0, dtype="int32", cycle_ms=2)
    d = node("EtherCatWrite", adapter="eth1", slave=2, offset=0, dtype="bool", bit=3)   # other cycle: shares
    assert len(soem.masters) == 1
    m = r.state["master"]
    assert m is w.state["master"] is d.state["master"] and m.users == 3
    assert soem.OP_STATE in soem.masters[0].state_log
    assert all(s.state == soem.OP_STATE for s in soem.masters[0].slaves)
    assert w.process({"value": -123456}) == {"ok": True}
    assert d.process({"value": True}) == {"ok": True}
    assert wait_for(lambda: r.process({})["value"] == -123456)
    # the cycle that looped w's value back may have copied the outputs before d wrote: d's bit goes out with the
    # next cycle (asserting it at once failed under load)
    assert wait_for(lambda: soem.masters[0].slaves[1].output == b"\x08")
    d.process({"value": 0})
    assert wait_for(lambda: soem.masters[0].slaves[1].output == b"\x00")
    assert w.process({"value": None}) == {"ok": False}
    # a bad value leaves the output image alone
    with pytest.raises(ValueError, match="Slave 3 çıkışları.*aralığının dışında"):
        node("EtherCatWrite", adapter="eth1", slave=3, dtype="uint8").process({"value": 300})
    assert r.process({})["value"] == -123456
    assert wait_for(lambda: m.cycles > 5)              # a starved cycle thread (busy PC) catches up
    for n in (r, w, d):
        n.teardown()


def test_stop_zeroes_outputs_goes_init_and_restarts_clean(soem):
    w = node("EtherCatWrite", adapter="eth1", slave=2, dtype="uint8")
    w.process({"value": 0x5A})
    master = soem.masters[0]
    assert wait_for(lambda: master.slaves[1].output == b"\x5A")
    thread = w.state["master"].thread
    w.teardown()
    assert not thread.is_alive()
    assert master.sent[-1][1] == b"\x00"                       # last cycle: outputs off
    assert master.state_log[-1] == soem.INIT_STATE and master.closed
    assert not ec.EcatMaster._masters
    w2 = node("EtherCatWrite", adapter="eth1", slave=2, dtype="uint8")   # second run: a fresh master
    assert len(soem.masters) == 2 and w2.state["master"].outs[1] == bytearray(1)
    w2.teardown()


def test_bad_addresses_have_turkish_errors(soem):
    r = node("EtherCatRead", adapter="eth1", slave=5)
    with pytest.raises(ValueError, match=r"ağda 3 slave var \(1…3\)"):
        r.process({})
    r.values.update(slave=1, offset=1)
    with pytest.raises(ValueError, match=r"Slave 1 girişleri \(1 bayt\).*dışında"):
        r.process({})
    r.values.update(slave=2, offset=0)
    with pytest.raises(ValueError, match="hiç veri baytı yok"):
        r.process({})
    r.values.update(slave=1, dtype="bool", bit=7)
    assert r.process({}) == {"value": 0, "bool": False}
    r.teardown()


def test_adapter_and_bus_errors(soem):
    n = create("EtherCatRead")
    assert not n.safe_setup()
    assert "adaptör" in n.setup_error and "eth1 (Intel I210)" in n.setup_error
    n.values["adapter"] = "eth9"
    assert not n.safe_setup()
    assert "'eth9' açılamadı" in n.setup_error and "eth1" in n.setup_error
    soem.layout = lambda: []
    n.values["adapter"] = "eth1"
    assert not n.safe_setup()
    assert "slave bulunamadı" in n.setup_error
    assert soem.masters[-1].closed and not ec.EcatMaster._masters


def test_slave_refusing_op_is_named_and_master_stopped(soem):
    def layout():
        slaves = make_pysoem().layout()
        slaves[2].refuse_op = True
        return slaves
    soem.layout = layout
    n = create("EtherCatRead")
    n.values["adapter"] = "eth1"
    assert not n.safe_setup()
    assert "OP durumuna geçmedi" in n.setup_error and "3:EL4102 SAFE-OP + HATA" in n.setup_error
    assert "0x001D" in n.setup_error
    assert soem.masters[0].closed and not ec.EcatMaster._masters
    import threading
    assert not any(t.name.startswith("EtherCAT") for t in threading.enumerate())


def test_lost_working_counter_fails_reads_and_recovers(soem, monkeypatch):
    monkeypatch.setattr(ec, "STALE_AFTER", 0.05)
    r = node("EtherCatRead", adapter="eth1", slave=3, cycle_ms=2)
    m, fake = r.state["master"], soem.masters[0]
    fake.wkc_ok = False
    fake.slaves[0].state = soem.SAFEOP_STATE + soem.STATE_ERROR     # slave fell out of OP
    assert wait_for(lambda: m.bad_cycles > 3 and m.recoveries > 0)
    time.sleep(0.08)
    with pytest.raises(RuntimeError, match="haberleşmesi kesik"):
        r.process({})
    assert soem.SAFEOP_STATE + soem.STATE_ACK in fake.slaves[0].state_writes
    fake.wkc_ok = True
    assert wait_for(lambda: m.bad_cycles == 0)
    assert r.process({})["value"] == 0
    r.teardown()


def test_cycle_thread_crash_is_reported_and_next_run_reopens(soem):
    r = node("EtherCatRead", adapter="eth1", slave=1)
    soem.masters[0].fail_cycle = True
    assert wait_for(lambda: r.state["master"].error is not None)
    with pytest.raises(RuntimeError, match="çevrimi durdu: adapter gone"):
        r.process({})
    r2 = node("EtherCatRead", adapter="eth1", slave=1)            # dead master is replaced, not reused
    assert r2.state["master"] is not r.state["master"] and len(soem.masters) == 2
    r.teardown()
    r2.teardown()


def test_scan_block_lists_adapters_and_slaves(soem):
    s = node("EtherCatScan")
    out = s.process({"scan": None})
    assert out == {"text": "eth1  Intel I210", "count": 1}
    s.values["adapter"] = "eth1"
    assert s.process({"scan": False})["count"] == 1               # cached until a rising edge
    out = s.process({"scan": True})
    assert out["count"] == 3
    lines = out["text"].splitlines()
    assert lines[0].startswith("1: EL1008  üretici 0x00000002  ürün 0x03F03052  giriş 1 B  çıkış 0 B")
    assert "3: EL4102" in lines[2] and "çıkış 4 B" in lines[2]
    assert soem.masters[-1].closed and len(soem.masters) == 1
    # while a flow runs, the scan reads the running master instead of opening the bus a second time
    r = node("EtherCatRead", adapter="eth1")
    s2 = node("EtherCatScan", adapter="eth1")
    assert "durum OP" in s2.process({})["text"] and len(soem.masters) == 2
    r.teardown()
    assert ec.scan("eth1")[1]["name"] == "EL2008"


# --- Profinet DCP ---------------------------------------------------------------------------------------------

MY_MAC = bytes.fromhex("020000000001")
DEV_MAC = bytes.fromhex("001b1b123456")


def blk(opt, sub, payload, info=True):
    body = (b"\x00\x00" if info else b"") + payload
    return struct.pack(">BBH", opt, sub, len(body)) + body + (b"\x00" if len(body) % 2 else b"")


def identify_response(xid, name=b"plc-1", mac=DEV_MAC, vlan=False, ip=(192, 168, 0, 1), extra=b""):
    blocks = (blk(2, 1, b"S7-1500") + blk(2, 2, name) + blk(2, 3, b"\x00\x2a\x01\x0e") + blk(2, 4, b"\x03\x00")
              + blk(1, 2, bytes(ip) + bytes([255, 255, 255, 0]) + bytes([192, 168, 0, 254])) + extra)
    pdu = struct.pack(">HBBIHH", 0xFEFF, 5, 1, xid, 0, len(blocks)) + blocks
    tag = b"\x81\x00\x00\x00" if vlan else b""
    return MY_MAC + mac + tag + b"\x88\x92" + pdu


def test_build_identify_frame():
    f = pn.build_identify(MY_MAC, 0x12345678, response_delay=100)
    assert len(f) == 60
    assert f[:6] == bytes.fromhex("010ecf000000") and f[6:12] == MY_MAC and f[12:14] == b"\x88\x92"
    assert f[14:16] == b"\xfe\xfe" and f[16] == 5 and f[17] == 0
    assert f[18:22] == bytes.fromhex("12345678") and f[22:24] == b"\x00\x64" and f[24:26] == b"\x00\x04"
    assert f[26:30] == b"\xff\xff\x00\x00" and f[30:] == bytes(30)
    assert pn.build_identify(MY_MAC, 1, response_delay=0)[22:24] == b"\x00\x01"
    assert pn.build_identify(MY_MAC, 1, response_delay=10 ** 6)[22:24] == b"\x19\x00"


def test_parse_identify_response():
    # odd-length name ("plc-1" = 5 bytes + BlockInfo = 7) is padded; the blocks after it must still parse
    dev = pn.parse_identify_response(identify_response(7, extra=blk(2, 6, b"port-001.plc-1")), 7)
    assert dev == {"mac": "00:1B:1B:12:34:56", "name": "plc-1", "type": "S7-1500", "vendor_id": 0x2A,
                   "device_id": 0x010E, "role": "IO-Device, IO-Controller", "ip": "192.168.0.1",
                   "mask": "255.255.255.0", "gateway": "192.168.0.254", "alias": "port-001.plc-1"}
    assert pn.parse_identify_response(identify_response(7, vlan=True), 7)["name"] == "plc-1"
    assert pn.parse_identify_response(identify_response(7), 8) is None          # other transaction
    assert pn.parse_identify_response(identify_response(7), None)["name"] == "plc-1"
    f = identify_response(7)
    assert pn.parse_identify_response(f[:12] + b"\x08\x00" + f[14:], 7) is None  # IPv4, not Profinet
    assert pn.parse_identify_response(f[:20], 7) is None
    assert pn.parse_identify_response(b"", 7) is None and pn.parse_identify_response(None, 7) is None
    req = pn.build_identify(MY_MAC, 7)
    assert pn.parse_identify_response(req, 7) is None                            # our own request echoed back
    cut = pn.parse_identify_response(f[:-10], 7)                                 # truncated: no crash
    assert cut["name"] == "plc-1" and cut["ip"] == ""


class FakeTransport:
    def __init__(self, answer):
        self.mac, self.sent, self.answer, self.queue = MY_MAC, [], answer, []

    def send(self, frame):
        self.sent.append(frame)
        self.queue.extend(self.answer(frame))

    def recv(self, timeout):
        if self.queue:
            return self.queue.pop(0)
        time.sleep(min(timeout, 0.01))
        return None

    def close(self):
        pass


def xid_of(frame):
    return struct.unpack_from(">I", frame, 18)[0]


def test_discover_with_fake_network():
    def answer(req):
        x = xid_of(req)
        return [b"\x00" * 10, identify_response(x, b"b-camera", bytes.fromhex("0000000000bb"), ip=(10, 0, 0, 2)),
                identify_response(x + 1, b"stale"), identify_response(x, b"a-plc"),
                identify_response(x, b"a-plc"), b"garbage" * 3]
    t = FakeTransport(answer)
    devs = pn.discover("eth0", 0.1, transport=t)
    assert [d["name"] for d in devs] == ["a-plc", "b-camera"]
    assert len(t.sent) == 1 and t.sent[0][14:16] == b"\xfe\xfe"
    txt = pn.format_devices(devs)
    assert "a-plc  IP 192.168.0.1  MAC 00:1B:1B:12:34:56  üretici 0x002A cihaz 0x010E  S7-1500" in txt
    assert pn.format_devices([]) == "Profinet cihazı bulunamadı"

    pn.ProfinetDiscover.transport = t
    try:
        b = node("ProfinetDiscover", iface="eth0", timeout_ms=200)
        out = b.process({"scan": None})
        assert out["count"] == 2 and out["names"] == "a-plc, b-camera"
        assert len(t.sent) == 2
        b.process({"scan": False})
        assert len(t.sent) == 2                                  # cached
        b.process({"scan": True})
        b.process({"scan": True})
        assert len(t.sent) == 3                                  # once per rising edge
    finally:
        pn.ProfinetDiscover.transport = None


def test_station_names():
    assert pn.check_station_name(" PLC-1.Line2 ") == "plc-1.line2"
    for bad, msg in [("", "boş"), ("a" * 241, "240"), ("a..b", "1…63"), ("x" * 64, "1…63"), ("ölçüm", "Türkçe"),
                     ("my plc", "a-z"), ("-plc", "'-'"), ("plc-", "'-'"), ("192.168.0.1", "IP adresi"),
                     ("port-001", "port-xyz"), ("port-002-00001.x", "port-xyz")]:
        with pytest.raises(ValueError, match=msg):
            pn.check_station_name(bad)
    assert pn.check_station_name("port-1") == "port-1"
    assert pn.parse_mac("00-1B-1B-12-34-56") == DEV_MAC and pn.parse_mac("001b.1b12.3456") == DEV_MAC
    for bad in ("", "00:1B", "zz:1b:1b:12:34:56", None):
        with pytest.raises(ValueError, match="Geçersiz MAC"):
            pn.parse_mac(bad)


def test_build_set_frame():
    f = pn.build_set(MY_MAC, DEV_MAC, 9, name="plc-1", ip="192.168.0.10", gateway="192.168.0.1", permanent=True)
    assert f[:6] == DEV_MAC and f[14:16] == b"\xfe\xfd" and f[16] == 4 and xid_of(f) == 9
    length = struct.unpack_from(">H", f, 24)[0]
    blocks = f[26:26 + length]
    # name: 02 02 len=7 qualifier 0001 'plc-1' + pad; then IP: 01 02 len=14 qualifier ip mask gw
    assert blocks[:4] == b"\x02\x02\x00\x07" and blocks[4:11] == b"\x00\x01plc-1" and blocks[11] == 0
    assert blocks[12:16] == b"\x01\x02\x00\x0e" and blocks[16:18] == b"\x00\x01"
    assert blocks[18:30] == bytes([192, 168, 0, 10, 255, 255, 255, 0, 192, 168, 0, 1])
    assert length == 30
    temp = pn.build_set(MY_MAC, DEV_MAC, 9, name="ab", permanent=False)
    assert temp[30:32] == b"\x00\x00"
    with pytest.raises(ValueError, match="Geçersiz IP adresi"):
        pn.build_set(MY_MAC, DEV_MAC, 1, ip="300.1.1.1")
    with pytest.raises(ValueError, match="alt ağ maskesi"):
        pn.build_set(MY_MAC, DEV_MAC, 1, ip="10.0.0.1", mask="x")
    with pytest.raises(ValueError, match="yok"):
        pn.build_set(MY_MAC, DEV_MAC, 1)


def set_response(xid, err=0, src=DEV_MAC, opt=2, sub=2):
    blocks = blk(5, 4, bytes([opt, sub, err]), info=False)
    pdu = struct.pack(">HBBIHH", 0xFEFD, 4, 1, xid, 0, len(blocks)) + blocks
    return MY_MAC + src + b"\x88\x92" + pdu


def test_set_device_and_block():
    assert pn.parse_set_response(set_response(5), 5, DEV_MAC) == [(2, 2, 0)]
    assert pn.parse_set_response(set_response(5), 6, DEV_MAC) is None
    assert pn.parse_set_response(set_response(5, src=MY_MAC), 5, DEV_MAC) is None
    ok = FakeTransport(lambda req: [identify_response(1), set_response(xid_of(req))])
    assert pn.set_device("eth0", DEV_MAC, name="Plc-2", transport=ok) == "00:1B:1B:12:34:56: ad 'plc-2' atandı"
    refused = FakeTransport(lambda req: [set_response(xid_of(req), err=6)])
    with pytest.raises(RuntimeError, match="çalışırken"):
        pn.set_device("eth0", DEV_MAC, name="plc-2", transport=refused)
    silent = FakeTransport(lambda req: [])
    with pytest.raises(TimeoutError, match="cevap vermedi"):
        pn.set_device("eth0", DEV_MAC, ip="10.0.0.5", timeout=0.05, transport=silent)

    pn.ProfinetSetName.transport = ok
    try:
        b = node("ProfinetSetName", iface="eth0", mac="00:1b:1b:12:34:56", name="plc-2")
        assert b.process({"apply": None})["ok"] is False and not ok.sent[1:]
        with pytest.raises(ValueError, match="Onaylıyorum"):
            b.process({"apply": True})
        assert len(ok.sent) == 1                                  # nothing sent without confirmation
        b.values["confirm"] = True
        b.process({"apply": False})
        out = b.process({"apply": True})
        assert out == {"ok": True, "text": "00:1B:1B:12:34:56: ad 'plc-2' atandı"}
        b.process({"apply": True})
        assert len(ok.sent) == 2                                  # once per rising edge
        b.values.update(name="kötü ad")
        b.process({"apply": False})
        with pytest.raises(ValueError, match="Türkçe"):
            b.process({"apply": True})
        assert len(ok.sent) == 2
    finally:
        pn.ProfinetSetName.transport = None


def test_transport_errors(monkeypatch):
    with pytest.raises(ValueError, match="Ağ kartı adı boş"):
        pn.open_transport("  ")
    monkeypatch.setattr(pn.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "scapy", None)
    monkeypatch.setitem(sys.modules, "scapy.all", None)
    with pytest.raises(RuntimeError, match="Npcap.*scapy"):
        pn.open_transport("Ethernet")
    b = create("ProfinetDiscover")
    b.values["iface"] = "Ethernet"
    assert b.safe_setup()
    with pytest.raises(RuntimeError, match="Npcap"):
        b.process({})


@pytest.mark.skipif(not hasattr(pn.socket, "AF_PACKET"), reason="Linux only")
def test_linux_transport_bad_interface():
    with pytest.raises((ConnectionError, PermissionError), match="açılamadı|CAP_NET_RAW"):
        pn.open_transport("yok-boyle-kart0")
