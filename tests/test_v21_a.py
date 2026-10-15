"""2.1 (a): VisionFlow as a server for PLCs / SCADA / HMIs — OPC UA server and Modbus TCP server blocks."""
import asyncio
import json
import pathlib
import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visionflow.core import Graph, load_builtin  # noqa: E402
from visionflow.nodes import plc_server  # noqa: E402
from visionflow.nodes.industrial import ModbusClient  # noqa: E402

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def server_threads():
    return [t for t in threading.enumerate() if t.name.startswith(("OPC UA sunucusu", "Modbus sunucusu"))]


@pytest.fixture(autouse=True)
def no_leaks():
    yield
    assert not plc_server._OpcServer._servers and not plc_server._ModbusServer._servers
    deadline = time.time() + 5
    while server_threads() and time.time() < deadline:
        time.sleep(0.05)
    assert not server_threads()


def opc(port, coro_fn, user=None, password=None):
    """Run ``coro_fn(client, ns)`` with an asyncua client connected to the VisionFlow server (10 s limit)."""
    from asyncua import Client

    async def main():
        c = Client(f"opc.tcp://127.0.0.1:{port}/visionflow/", timeout=4)
        if user:
            c.set_user(user)
            c.set_password(password)
        async with c:
            ns = await c.get_namespace_index("urn:visionflow")
            return await coro_fn(c, ns)
    return asyncio.run(asyncio.wait_for(main(), 10))


def outputs(results, node):
    r = results[node.id]
    assert r.error is None, r.error
    return r.outputs


# ------------------------------------------------------------------------------------------------- OPC UA
def test_opc_publish_values_status_and_read_only():
    port = free_port()
    g = Graph()
    num = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="Adet", folder="Hat1")
    flag = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="Iyi", vtype="Mantıksal (Boolean)")
    text = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="Sonuc")
    assert g.setup() == {}
    try:
        assert len(plc_server._OpcServer._servers) == 1 and plc_server._OpcServer._servers[port].users == 3
        srv = plc_server._OpcServer._servers[port]
        frames = [(7, 1, "İYİ"), (np.int64(12), "evet", "RED"), ([{"x": 1}] * 3, 0, 3.5)]
        expected = [(7, True, "İYİ"), (12, True, "RED"), (3, False, "3.5")]
        for (a, b, c), exp in zip(frames, expected):
            for node, val in ((num, a), (flag, b), (text, c)):
                out = node.process({"value": val})
                assert out["ok"] is True and out["url"].endswith(f":{port}/visionflow/")

            async def check(cl, ns):
                vals = [await cl.get_node(f"ns={ns};s=VisionFlow.{k}").read_value()
                        for k in ("Hat1.Adet", "Iyi", "Sonuc")]
                status = [await cl.get_node(f"ns={ns};s=VisionFlow.Durum.{k}").read_value()
                          for k in ("Calisiyor", "KareSayisi", "Surum")]
                vtype = await cl.get_node(f"ns={ns};s=VisionFlow.Hat1.Adet").read_data_type_as_variant_type()
                return vals, status, vtype
            vals, status, vtype = opc(port, check)
            assert tuple(vals) == exp
            assert status[0] is True and status[1] >= 1 and status[2]
            assert vtype.name == "Int32"                  # automatic: the first value (int) decided
        assert srv.frames == 3

        async def write_published(cl, ns):
            from asyncua import ua
            await cl.get_node(f"ns={ns};s=VisionFlow.Hat1.Adet").write_value(
                ua.DataValue(ua.Variant(99, ua.VariantType.Int32)))
        with pytest.raises(Exception, match="BadUserAccessDenied|AccessDenied|NotWritable"):
            opc(port, write_published)                    # published values are read-only for clients
        assert num.process({"value": None})["ok"] is False
    finally:
        g.teardown()
    assert port_is_free(port)


def test_opc_read_block_sees_client_writes_and_changed_flag():
    port = free_port()
    g = Graph()
    cmd = g.add_new("OpcUaServerRead", (0, 0), port=port, name="Komut", initial="4")
    recipe = g.add_new("OpcUaServerRead", (0, 0), port=port, name="Recete", folder="Hat1",
                       vtype="Metin (String)", initial="A")
    start = g.add_new("OpcUaServerRead", (0, 0), port=port, name="Basla", vtype="Mantıksal (Boolean)",
                      initial="hayır")
    assert g.setup() == {}
    try:
        r = g.run_once()
        assert outputs(r, cmd) == {"value": 4, "number": 4.0, "bool": True, "changed": False}
        assert outputs(r, recipe)["value"] == "A" and outputs(r, start)["bool"] is False

        async def write(cl, ns):
            from asyncua import ua
            await cl.get_node(f"ns={ns};s=VisionFlow.Komut").write_value(
                ua.DataValue(ua.Variant(17, ua.VariantType.Int32)))
            await cl.get_node(f"ns={ns};s=VisionFlow.Hat1.Recete").write_value(
                ua.DataValue(ua.Variant("B2", ua.VariantType.String)))
            await cl.get_node(f"ns={ns};s=VisionFlow.Basla").write_value(
                ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))
        opc(port, write)
        r = g.run_once()
        assert outputs(r, cmd) == {"value": 17, "number": 17.0, "bool": True, "changed": True}
        assert outputs(r, recipe)["value"] == "B2" and outputs(r, recipe)["changed"] is True
        assert outputs(r, start)["bool"] is True and outputs(r, start)["changed"] is True
        r = g.run_once()                                   # nothing new: changed drops back
        assert outputs(r, cmd)["changed"] is False and outputs(r, start)["changed"] is False
    finally:
        g.teardown()


def test_opc_publish_and_read_same_variable_and_restart_cleanly():
    port = free_port()
    g = Graph()
    pub = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="Ortak")
    rd = g.add_new("OpcUaServerRead", (0, 0), port=port, name="Ortak", vtype="Ondalık (Double)")
    g.connect(rd.id, "number", pub.id, "value")
    for round_ in range(2):                                # stop and start again on the same port
        assert g.setup() == {}
        try:
            r = g.run_once()
            assert outputs(r, rd)["value"] == 0.0

            async def write(cl, ns):
                from asyncua import ua
                await cl.get_node(f"ns={ns};s=VisionFlow.Ortak").write_value(
                    ua.DataValue(ua.Variant(2.5 + round_, ua.VariantType.Double)))
            opc(port, write)                               # the variable is writable (a Read block asked for it)
            r = g.run_once()
            assert outputs(r, rd)["value"] == 2.5 + round_ and outputs(r, pub)["ok"] is True
        finally:
            g.teardown()
        assert not plc_server._OpcServer._servers and port_is_free(port)


def test_opc_bad_inputs_give_turkish_errors():
    port = free_port()
    g = Graph()
    pub = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="Deger", vtype="Tamsayı (Int32)")
    assert g.setup() == {}
    try:
        for bad, msg in ((np.zeros((10, 10, 3), np.uint8), "Görüntü"), (np.zeros((4, 4), np.uint16), "Görüntü"),
                         ({"a": 1}, "Sözlük"), ("abc", "sayı değil"), (2 ** 40, "aralığı dışında"),
                         (float("nan"), "tamsayı")):
            with pytest.raises(ValueError, match=msg):
                pub.process({"value": bad})
        assert pub.process({"value": "12,4"})["ok"] is True     # numeric text with a Turkish comma

        async def read(cl, ns):
            return await cl.get_node(f"ns={ns};s=VisionFlow.Deger").read_value()
        assert opc(port, read) == 12
    finally:
        g.teardown()

    for values, msg in (({"name": ""}, "boş olamaz"), ({"name": "a.b"}, "karakter"),
                        ({"name": "x", "folder": "Durum"}, "Durum")):
        g = Graph()
        n = g.add_new("OpcUaServerPublish", (0, 0), port=port, **values)
        errors = g.setup()
        g.teardown()
        assert msg in errors[n.id]
    g = Graph()
    n = g.add_new("OpcUaServerRead", (0, 0), port=port, name="x", initial="yazı")
    assert "sayı değil" in g.setup()[n.id]                 # bad initial value for an Int32 variable
    g.teardown()


def test_opc_user_password_and_conflicts():
    port = free_port()
    g = Graph()
    rd = g.add_new("OpcUaServerRead", (0, 0), port=port, name="K", user="operator", password="1234")
    other = g.add_new("OpcUaServerPublish", (0, 0), port=port, name="P", user="baska", password="x")
    errors = g.setup()
    try:
        assert rd.id not in errors and "aynı kullanıcı" in errors[other.id]

        async def read(cl, ns):
            return await cl.get_node(f"ns={ns};s=VisionFlow.K").read_value()
        assert opc(port, read, "operator", "1234") == 0
        with pytest.raises(Exception):
            opc(port, read)                                # anonymous refused
        with pytest.raises(Exception):
            opc(port, read, "operator", "yanlis")
    finally:
        g.teardown()


def test_opc_port_in_use_and_missing_asyncua(monkeypatch):
    with socket.socket() as blocker:
        blocker.bind(("0.0.0.0", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        g = Graph()
        n = g.add_new("OpcUaServerPublish", (0, 0), port=port)
        err = g.setup()[n.id]
        g.teardown()
        assert "kullanımda" in err or "açılamadı" in err
    monkeypatch.setitem(sys.modules, "asyncua", None)
    g = Graph()
    n = g.add_new("OpcUaServerPublish", (0, 0), port=free_port())
    assert "pip install asyncua" in g.setup()[n.id]
    g.teardown()


# --------------------------------------------------------------------------------------------- Modbus TCP
def test_modbus_encode_decode():
    f, o = plc_server.MB_FORMATS, plc_server.MB_ORDERS
    assert plc_server.mb_encode(-2, f[0], o[0]) == [0xFFFE]
    assert plc_server.mb_decode([0xFFFE], f[0], o[0]) == -2 and plc_server.mb_decode([0xFFFE], f[1], o[0]) == 65534
    assert plc_server.mb_encode(1.5, f[2], o[0]) == [0x3FC0, 0] and plc_server.mb_encode(1.5, f[2], o[1]) == [0, 0x3FC0]
    for fmt in f[2:]:
        for order in o:
            assert plc_server.mb_decode(plc_server.mb_encode(-123456, fmt, order), fmt, order) == -123456
    for v, fmt in ((40000, f[0]), (-1, f[1]), (2 ** 31, f[3]), (float("inf"), f[0])):
        with pytest.raises(ValueError):
            plc_server.mb_encode(v, fmt, o[0])


def raw(port, pdu, unit=1):
    with socket.create_connection(("127.0.0.1", port), 3) as s:
        s.settimeout(3)
        s.sendall(struct.pack(">HHHB", 9, 0, len(pdu) + 1, unit) + pdu)
        head = s.recv(7)
        tid, _, length, u = struct.unpack(">HHHB", head)
        assert tid == 9 and u == unit
        body = b""
        while len(body) < length - 1:
            body += s.recv(length - 1 - len(body))
        return body


def test_modbus_write_blocks_read_by_plc_client():
    port = free_port()
    g = Graph()
    f = plc_server.MB_FORMATS
    w16 = g.add_new("ModbusServerWrite", (0, 0), port=port, address=0)
    wf = g.add_new("ModbusServerWrite", (0, 0), port=port, address=10, format=f[2])
    wf2 = g.add_new("ModbusServerWrite", (0, 0), port=port, address=20, format=f[2], order=plc_server.MB_ORDERS[1])
    w32 = g.add_new("ModbusServerWrite", (0, 0), port=port, address=30, format=f[3])
    coil = g.add_new("ModbusServerWrite", (0, 0), port=port, area="Bobin / coil (0x)", address=5)
    di = g.add_new("ModbusServerWrite", (0, 0), port=port, area="Discrete input (1x)", address=3)
    ir = g.add_new("ModbusServerWrite", (0, 0), port=port, area="Input register (3x)", address=7)
    assert g.setup() == {}
    try:
        assert plc_server._ModbusServer._servers[port].users == 7
        for node, val in ((w16, -5), (wf, 3.25), (wf2, 3.25), (w32, 100000), (coil, True), (di, "evet"),
                          (ir, np.float32(42.0))):
            assert node.process({"value": val}) == {"ok": True}
        c = ModbusClient("127.0.0.1", port, unit=3, timeout=3)
        try:
            assert c.read_registers(0) == [0xFFFB]
            assert struct.unpack(">f", struct.pack(">HH", *c.read_registers(10, 2)))[0] == 3.25
            lo, hi = c.read_registers(20, 2)
            assert struct.unpack(">f", struct.pack(">HH", hi, lo))[0] == 3.25
            assert struct.unpack(">i", struct.pack(">HH", *c.read_registers(30, 2)))[0] == 100000
            assert c.read_coils(0, 8) == [False] * 5 + [True, False, False]
        finally:
            c.close()
        assert raw(port, struct.pack(">BHH", 2, 0, 4)) == bytes([2, 1, 0b1000])          # discrete inputs
        assert raw(port, struct.pack(">BHH", 4, 7, 1)) == bytes([4, 2, 0, 42])           # input register
        assert coil.process({"value": 0}) == {"ok": True}
        assert raw(port, struct.pack(">BHH", 1, 5, 1)) == bytes([1, 1, 0])
        assert w16.process({"value": None}) == {"ok": False}
    finally:
        g.teardown()


def test_modbus_read_blocks_see_plc_writes():
    port = free_port()
    g = Graph()
    f, o = plc_server.MB_FORMATS, plc_server.MB_ORDERS
    r16 = g.add_new("ModbusServerRead", (0, 0), port=port, address=100)
    rf = g.add_new("ModbusServerRead", (0, 0), port=port, address=200, format=f[2], order=o[1])
    rc = g.add_new("ModbusServerRead", (0, 0), port=port, area="Bobin / coil (0x)", address=9)
    assert g.setup() == {}
    try:
        r = g.run_once()
        assert outputs(r, r16) == {"value": 0, "bool": False, "changed": False}
        c = ModbusClient("127.0.0.1", port, timeout=3)
        try:
            c.write_register(100, 0xFFFF)                  # -1 as a signed 16-bit number
            c.write_coil(9, True)
            hi, lo = struct.unpack(">HH", struct.pack(">f", -7.5))
            c.request(struct.pack(">BHHB2H", 16, 200, 2, 4, lo, hi))   # low word first (CDAB)
        finally:
            c.close()
        r = g.run_once()
        assert outputs(r, r16) == {"value": -1, "bool": True, "changed": True}
        assert outputs(r, rf) == {"value": -7.5, "bool": True, "changed": True}
        assert outputs(r, rc) == {"value": 1, "bool": True, "changed": True}
        r = g.run_once()
        assert not any(outputs(r, n)["changed"] for n in (r16, rf, rc))
        assert raw(port, bytes([15]) + struct.pack(">HHB", 8, 3, 1) + bytes([0b101])) == \
            bytes([15]) + struct.pack(">HH", 8, 3)
        r = g.run_once()
        assert outputs(r, rc) == {"value": 0, "bool": False, "changed": True}   # coil 9 = bit 1 = 0
    finally:
        g.teardown()


def test_modbus_exception_responses():
    port = free_port()
    g = Graph()
    g.add_new("ModbusServerRead", (0, 0), port=port)
    assert g.setup() == {}
    try:
        assert raw(port, struct.pack(">BHH", 3, 9999, 2)) == bytes([0x83, 2])     # past the last address
        assert raw(port, struct.pack(">BHH", 3, 0, 126)) == bytes([0x83, 3])      # too many registers
        assert raw(port, struct.pack(">BHH", 1, 0, 0)) == bytes([0x81, 3])
        assert raw(port, struct.pack(">BHH", 6, 10000, 1)) == bytes([0x86, 2])
        assert raw(port, struct.pack(">BHH", 5, 0, 0x1234)) == bytes([0x85, 3])   # coil value must be FF00/0000
        assert raw(port, bytes([7])) == bytes([0x87, 1])                            # unsupported function
        assert raw(port, bytes([16]) + struct.pack(">HHB", 0, 2, 3) + b"\0\0\0") == bytes([0x90, 3])
        assert raw(port, bytes([3, 0])) == bytes([0x83, 3])                         # short request
        c = ModbusClient("127.0.0.1", port, timeout=3)
        try:
            with pytest.raises(RuntimeError, match="hata kodu 2"):
                c.read_registers(9999, 5)
            assert c.read_registers(9999) == [0]           # the connection still works after an exception
        finally:
            c.close()
    finally:
        g.teardown()


def test_modbus_restart_drops_clients_and_frees_port():
    port = free_port()
    g = Graph()
    w = g.add_new("ModbusServerWrite", (0, 0), port=port, address=1)
    for value in (11, 22):
        assert g.setup() == {}
        c = ModbusClient("127.0.0.1", port, timeout=3)
        try:
            w.process({"value": value})
            assert c.read_registers(1) == [value]          # a fresh server: nothing left from the last run
        finally:
            g.teardown()
        with pytest.raises((OSError, ConnectionError)):
            c.read_registers(1)                            # the PLC connection was closed with the server
        c.close()
        assert port_is_free(port)


def test_modbus_bad_inputs():
    port = free_port()
    g = Graph()
    w = g.add_new("ModbusServerWrite", (0, 0), port=port, address=9999, format=plc_server.MB_FORMATS[2])
    w16 = g.add_new("ModbusServerWrite", (0, 0), port=port, address=0)
    assert g.setup() == {}
    try:
        r = g.run_once()                                   # unconnected input: nothing written, no error
        assert outputs(r, w) == {"ok": False}
        with pytest.raises(ValueError, match="iki register"):
            w.process({"value": 1.0})
        for bad, msg in ((np.zeros((8, 8), np.float32), "Görüntü"), ("on iki", "sayı değil"),
                         (70000, "aralığı dışında"), (np.zeros((2, 2, 4), np.uint8), "Görüntü"),
                         (np.array([1, 2]), "dizi")):
            with pytest.raises(ValueError, match=msg):
                w16.process({"value": bad})
        assert w16.process({"value": np.array([7])}) == {"ok": True}
    finally:
        g.teardown()
    with socket.socket() as blocker:
        blocker.bind(("0.0.0.0", 0))
        blocker.listen(1)
        g = Graph()
        n = g.add_new("ModbusServerWrite", (0, 0), port=blocker.getsockname()[1])
        assert "kullanımda" in g.setup()[n.id]
        g.teardown()


# --------------------------------------------------------------------------------------- example / headless
def test_example_flow_runs_headless_and_serves_the_count(tmp_path):
    from visionflow import headless
    data = json.loads((ROOT / "examples" / "opcua_sunucu.json").read_text(encoding="utf-8"))
    opc_port, mb_port = free_port(), free_port()
    for n in data["nodes"]:
        if n["type"].startswith("OpcUaServer"):
            n["values"]["port"] = opc_port
        elif n["type"].startswith("ModbusServer"):
            n["values"]["port"] = mb_port
    flow = tmp_path / "akis.json"
    flow.write_text(json.dumps(data), encoding="utf-8")
    stop = threading.Event()
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("rc", headless.run_flow(
        str(flow), fps=20, stats=999, should_stop=stop.is_set)), daemon=True)
    t.start()
    try:
        deadline = time.time() + 20
        got = None
        while time.time() < deadline:
            try:
                async def read(cl, ns):
                    return await cl.get_node(f"ns={ns};s=VisionFlow.Hat1.NesneSayisi").read_value()
                got = opc(opc_port, read)
                c = ModbusClient("127.0.0.1", mb_port, timeout=3)
                try:
                    mb = c.read_registers(0)[0]
                finally:
                    c.close()
                if got and mb:
                    break
            except Exception:  # noqa: BLE001  (server still starting / no frame yet)
                pass
            time.sleep(0.2)
        assert got and got > 0 and mb == got               # the test pattern has red objects
    finally:
        stop.set()
        t.join(20)
    assert not t.is_alive() and result["rc"] == 0
