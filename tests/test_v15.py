"""1.5: Modbus RTU (RS-485) and Allen-Bradley (EtherNet/IP) blocks, tested against fake devices."""
import struct
import sys
import types

import pytest

from visionflow.core import create, load_builtin
from visionflow.core.registry import NODE_REGISTRY
from visionflow.nodes import fieldbus
from visionflow.nodes.fieldbus import AbRead, AbWrite, ModbusRtuClient, ModbusRtuRead, ModbusRtuWrite, crc16

load_builtin()


# --- a fake RS-485 bus with Modbus slaves on it ------------------------------------------------------------------

class FakeBus:
    def __init__(self):
        self.regs = {1: {0: 1234, 5: 7}, 2: {0: 99}}
        self.coils = {1: {3: True}, 2: {}}
        self.opened, self.closed, self.frames = [], 0, []
        self.out = b""
        self.mangle = None              # fn(reply) -> reply, to fake wire trouble
        self.silent = False

    def factory(self, **kw):
        self.opened.append(kw)
        s = FakeSerial(self)
        s.baudrate, s.parity = kw["baudrate"], kw["parity"]
        return s

    def answer(self, frame: bytes) -> bytes:
        assert struct.unpack("<H", frame[-2:])[0] == crc16(frame[:-2]), "master sent a bad CRC"
        unit, fc = frame[0], frame[1]
        addr, val = struct.unpack(">HH", frame[2:6])
        if unit not in self.regs:
            return b""
        if fc == 3:
            if addr not in self.regs[unit]:
                body = bytes([unit, 0x83, 2])
            else:
                data = b"".join(struct.pack(">H", self.regs[unit].get(addr + i, 0)) for i in range(val))
                body = bytes([unit, 3, len(data)]) + data
        elif fc == 1:
            bits = sum(1 << i for i in range(val) if self.coils[unit].get(addr + i))
            data = bits.to_bytes((val + 7) // 8, "little")
            body = bytes([unit, 1, len(data)]) + data
        elif fc == 6:
            self.regs[unit][addr] = val
            body = frame[:6]
        elif fc == 5:
            self.coils[unit][addr] = val == 0xFF00
            body = frame[:6]
        else:
            body = bytes([unit, fc | 0x80, 1])
        return body + struct.pack("<H", crc16(body))


class FakeSerial:
    def __init__(self, bus):
        self.bus = bus

    def reset_input_buffer(self):
        self.bus.out = b""

    def write(self, frame):
        self.bus.frames.append(bytes(frame))
        if not self.bus.silent:
            reply = self.bus.answer(bytes(frame))
            self.bus.out += self.bus.mangle(reply) if self.bus.mangle else reply

    def read(self, n):
        chunk, self.bus.out = self.bus.out[:n], self.bus.out[n:]
        return chunk

    def close(self):
        self.bus.closed += 1


@pytest.fixture
def bus(monkeypatch):
    b = FakeBus()
    monkeypatch.setattr(ModbusRtuRead, "serial_factory", b.factory)
    monkeypatch.setattr(ModbusRtuWrite, "serial_factory", b.factory)
    monkeypatch.setattr(fieldbus._SharedPort, "_ports", {})
    monkeypatch.setattr(fieldbus.time, "sleep", lambda s: None)
    return b


def rtu(kind="register", address=0, unit=1, write=False, port="COM5", baud="9600"):
    n = create("ModbusRtuWrite" if write else "ModbusRtuRead")
    n.values.update(port=port, baud=baud, parity="E", unit=unit, kind=kind, address=address)
    assert n.safe_setup(), n.setup_error
    return n


def test_block_count_and_titles():
    for name in ("ModbusRtuRead", "ModbusRtuWrite", "AbRead", "AbWrite"):
        assert name in NODE_REGISTRY
    assert len(NODE_REGISTRY) >= 125
    assert ModbusRtuRead.category == "1 · Giriş" and AbWrite.category == "5 · Çıkış ve Otomasyon"


def test_crc_known_vector():
    # the classic example frame "read 1 holding register at 0 from unit 1" ends in 84 0A
    frame = bytes.fromhex("010300000001")
    assert struct.pack("<H", crc16(frame)) == bytes.fromhex("840A")


def test_read_register_and_coil(bus):
    n = rtu(address=0)
    assert n.process({}) == {"value": 1234, "bool": True}
    assert bus.frames[-1] == bytes.fromhex("010300000001840A")
    assert bus.opened[0]["baudrate"] == 9600 and bus.opened[0]["parity"] == "E"
    c = rtu(kind="coil", address=3)
    assert c.process({}) == {"value": 1, "bool": True}
    c.values["address"] = 4
    assert c.process({})["bool"] is False


def test_write_on_change_only(bus):
    w = rtu(write=True, address=10)
    assert w.process({"value": 42.0}) == {"ok": True}
    assert bus.regs[1][10] == 42
    sent = len(bus.frames)
    w.process({"value": 42})
    assert len(bus.frames) == sent                        # unchanged value is not sent again
    w.process({"value": 70000})                           # registers are 16 bit
    assert bus.regs[1][10] == 70000 & 0xFFFF
    assert w.process({"value": None}) == {"ok": False}
    wc = rtu(write=True, kind="coil", address=8, unit=2)
    wc.process({"value": 1})
    assert bus.coils[2][8] is True
    wc.process({"value": 0})
    assert bus.coils[2][8] is False


def test_errors_are_clear(bus):
    n = rtu(address=77)                                   # not a register on the device: exception code 2
    with pytest.raises(RuntimeError, match="hata kodu 2"):
        n.process({})
    n.values["address"] = 0
    bus.mangle = lambda r: r[:-1] + bytes([r[-1] ^ 0xFF])
    with pytest.raises(RuntimeError, match="CRC"):
        n.process({})
    bus.mangle = lambda r: bytes([9]) + r[1:]
    with pytest.raises(RuntimeError, match="Yanlış cihaz"):
        n.process({})
    bus.mangle = None
    assert n.process({})["value"] == 1234                  # the bus recovers after bad answers


def test_timeout(bus, monkeypatch):
    client = ModbusRtuClient("COM9", 9600, "N", 1, timeout=0.01, factory=bus.factory)
    bus.silent = True
    with pytest.raises(TimeoutError, match="cevap vermedi"):
        client.read_registers(0)
    bus.silent = False
    assert client.read_registers(0, 2) == [1234, 0]
    client.close()
    client.close()                                         # closing twice is harmless
    assert bus.closed == 1


def test_missing_device_times_out(bus):
    client = ModbusRtuClient("COM9", 9600, "N", 30, timeout=0.01, factory=bus.factory)
    with pytest.raises(TimeoutError):
        client.read_registers(0)
    client.close()


def test_blocks_share_one_port(bus):
    a = rtu(unit=1, port="com5")
    b = rtu(unit=2, port="COM5", write=True)
    assert len(bus.opened) == 1                            # one bus, one open port (COM names ignore case)
    assert a.process({})["value"] == 1234
    b.process({"value": 5})
    assert bus.regs[2][0] == 5
    a.teardown()
    assert bus.closed == 0                                 # still used by the other block
    b.teardown()
    assert bus.closed == 1
    a.teardown()                                           # teardown twice does not close twice
    assert bus.closed == 1
    rtu()                                                  # a new run opens the port again
    assert len(bus.opened) == 2


def test_changing_the_bus_speed_while_running(bus, capsys):
    a = rtu(baud="9600")
    b = rtu(baud="9600", unit=2)
    b.teardown()                                           # the user changes the speed on b: it restarts
    b = rtu(baud="19200", unit=2)
    ser = a.state["client"].port.ser
    assert (ser.baudrate, len(bus.opened)) == (19200, 1)   # the shared port follows, a keeps working
    assert a.process({})["value"] == 1234 and b.process({})["value"] == 99
    assert "19200" in capsys.readouterr().out


def test_late_answer_is_not_taken_for_the_next_request(bus, monkeypatch):
    client = ModbusRtuClient(" COM9 ", 9600, "N", 1, timeout=0.01, factory=bus.factory)
    assert bus.opened[0]["port"] == "COM9"                 # spaces around the port name are ignored
    late = []
    bus.silent = True
    real_write = FakeSerial.write

    def slow_write(self, frame):
        real_write(self, frame)
        late.append(bus.answer(bytes(frame)))
    monkeypatch.setattr(FakeSerial, "write", slow_write)
    # the answer arrives while the client is waiting after the time-out
    def arrive(_s):
        bus.out += b"".join(late)
        late.clear()
    monkeypatch.setattr(fieldbus.time, "sleep", arrive)
    with pytest.raises(TimeoutError):
        client.read_registers(0)
    bus.silent = False
    monkeypatch.setattr(FakeSerial, "write", real_write)
    bus.regs[1][5] = 555
    assert client.read_registers(5) == [555]               # not 1234 from the late answer
    client.close()


def test_write_again_after_address_or_type_change(bus):
    w = rtu(write=True, address=0)
    w.process({"value": 7})
    w.values["address"] = 5
    w.process({"value": 7})
    assert bus.regs[1][5] == 7
    c = rtu(write=True, kind="coil", address=9)
    c.process({"value": 1})
    c.values["kind"] = "register"
    c.process({"value": 1})
    assert bus.regs[1][9] == 1


def test_short_or_wrong_echo_is_an_error(bus):
    n = rtu(address=0)
    bus.mangle = lambda r: (lambda b: b + struct.pack("<H", crc16(b)))(bytes([1, 3, 0]))
    with pytest.raises(RuntimeError, match="eksik"):
        n.process({})
    w = rtu(write=True, address=3)
    bus.mangle = lambda r: (lambda b: b + struct.pack("<H", crc16(b)))(r[:5] + bytes([r[5] ^ 1]))
    with pytest.raises(RuntimeError, match="farklı onayladı"):
        w.process({"value": 8})


def test_missing_pyserial(monkeypatch):
    monkeypatch.setattr(ModbusRtuRead, "serial_factory", None)
    monkeypatch.setattr(fieldbus._SharedPort, "_ports", {})
    monkeypatch.setitem(sys.modules, "serial", None)
    n = create("ModbusRtuRead")
    assert not n.safe_setup()
    assert "pip install pyserial" in n.setup_error


# --- a fake pycomm3 ---------------------------------------------------------------------------------------------

def fake_pycomm3(tags, connect=True):
    log = []

    class Tag:
        def __init__(self, tag, value, error=None):
            self.tag, self.value, self.error = tag, value, error

    class _Driver:
        def __init__(self, path, **kw):
            log.append((type(self).__name__, path, kw))

        def open(self):
            return connect

        def read(self, tag):
            if tag not in tags:
                return Tag(tag, None, "Tag doesn't exist")
            return Tag(tag, tags[tag])

        def write(self, item):
            tag, value = item
            log.append(("write", tag, value))
            tags[tag] = value
            return Tag(tag, value)

        def close(self):
            log.append(("close",))

    mod = types.ModuleType("pycomm3")
    mod.LogixDriver = type("LogixDriver", (_Driver,), {})
    mod.SLCDriver = type("SLCDriver", (_Driver,), {})
    return mod, log


def ab(kind, family=AbRead.params[1].default, tag="Sayac", slot=0):
    n = create(kind)
    n.values.update(host=" 10.0.0.5 ", family=family, slot=slot, tag=tag)
    return n


def test_ab_families_pick_the_right_driver(monkeypatch):
    mod, log = fake_pycomm3({"Sayac": 5})
    monkeypatch.setattr(AbRead, "driver_module", mod)
    for family, slot, want in (("ControlLogix / CompactLogix", 2, ("LogixDriver", "10.0.0.5/2", {})),
                               ("Micro800", 0, ("LogixDriver", "10.0.0.5",
                                                {"init_tags": True, "init_program_tags": False})),
                               ("SLC 500 / MicroLogix", 0, ("SLCDriver", "10.0.0.5", {}))):
        n = ab("AbRead", family, slot=slot)
        assert n.safe_setup(), n.setup_error
        assert log[-1] == want
        n.teardown()
        assert log[-1] == ("close",)


def test_ab_read_values_and_errors(monkeypatch):
    mod, _ = fake_pycomm3({"Sayac": 7, "Sicaklik": 21.5, "Hazir": True, "Mesaj": "OK"})
    monkeypatch.setattr(AbRead, "driver_module", mod)
    n = ab("AbRead")
    assert n.safe_setup()
    assert n.process({}) == {"value": 7, "bool": True, "text": "7"}
    n.values["tag"] = "Sicaklik"
    assert n.process({})["value"] == 21.5
    n.values["tag"] = "Hazir"
    assert n.process({}) == {"value": 1, "bool": True, "text": "True"}
    n.values["tag"] = "Mesaj"
    assert n.process({}) == {"value": 0, "bool": True, "text": "OK"}
    n.values["tag"] = "Yok"
    with pytest.raises(RuntimeError, match="Yok: Tag doesn't exist"):
        n.process({})


def test_ab_write_on_change(monkeypatch):
    tags = {}
    mod, log = fake_pycomm3(tags)
    monkeypatch.setattr(AbWrite, "driver_module", mod)
    n = ab("AbWrite", tag="Hatali")
    assert n.safe_setup()
    assert n.process({"value": 3.0}) == {"ok": True}
    assert tags["Hatali"] == 3 and type(tags["Hatali"]) is int      # DINT tag gets an integer
    writes = sum(1 for e in log if e[0] == "write")
    n.process({"value": 3})
    assert sum(1 for e in log if e[0] == "write") == writes
    n.process({"value": True})                                        # True == 1 but a different type: send it
    assert tags["Hatali"] is True
    assert n.process({"value": None}) == {"ok": False}
    f = ab("AbWrite", tag="Fault_Code")
    assert f.safe_setup()
    f.process({"value": 2.0})
    assert type(tags["Fault_Code"]) is int                            # a name starting with F is not special
    f.values["tag"] = "Fault_Old"
    f.process({"value": 2})
    assert tags["Fault_Old"] == 2                                     # new tag, same value: written again


def test_ab_connection_failure_and_missing_package(monkeypatch):
    mod, log = fake_pycomm3({}, connect=False)
    monkeypatch.setattr(AbRead, "driver_module", mod)
    n = ab("AbRead")
    assert not n.safe_setup()
    assert "bağlanılamadı" in n.setup_error and log[-1] == ("close",)

    def refuse(self):
        raise OSError("timed out")
    monkeypatch.setattr(mod.LogixDriver, "open", refuse)               # pycomm3 raises instead of returning False
    log.clear()
    n = ab("AbRead")
    assert not n.safe_setup()
    assert "bağlanılamadı: timed out" in n.setup_error and log[-1] == ("close",)
    monkeypatch.setattr(AbRead, "driver_module", None)
    monkeypatch.setitem(sys.modules, "pycomm3", None)
    n = ab("AbRead")
    assert not n.safe_setup()
    assert "pip install pycomm3" in n.setup_error
