"""VisionFlow as a SERVER that PLCs, SCADA and HMIs connect to: an OPC UA server and a Modbus TCP server (slave).

Every block that names the same port shares one server, which starts when the first of them starts and stops when
the last one stops (flow stopped), so a stopped and restarted flow leaves no thread or port behind. No Qt here: the
blocks work the same in the editor and in ``python -m visionflow --headless``.

OPC UA: endpoint ``opc.tcp://<ip>:<port>/visionflow/``, namespace ``urn:visionflow`` (normally ns=2); variables live
under Objects/VisionFlow[/<folder>] with string node ids ``ns=2;s=VisionFlow.<folder>.<name>``.
Modbus TCP: 10000 coils, discrete inputs, input registers and holding registers each (addresses 0…9999), function
codes 1, 2, 3, 4, 5, 6, 15 and 16; any unit id is answered.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import math
import socket
import socketserver
import struct
import threading

import numpy as np

from ..core import Node, Param, PortType, register

CAT_IN, CAT_OUT = "1 · Giriş", "5 · Çıkış ve Otomasyon"
OPC_NS = "urn:visionflow"
OPC_PATH = "/visionflow/"


# ---------------------------------------------------------------------------------------------- shared helpers
def _plain(value, what: str):
    """An input value as a plain Python bool / int / float / str. Images are refused with a clear message; a list
    (detections, contours) becomes its length. None stays None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim >= 2:
            raise ValueError(f"Görüntü {what} yazılamaz: önce sayıya veya metne çevirin "
                             "(ör. Nesne Say → adet, Ortalama Parlaklık → sayı)")
        if value.size != 1:
            raise ValueError(f"{value.size} elemanlı dizi {what} yazılamaz: tek bir sayı bağlayın")
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return len(value)                         # detections / contours: their count
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        raise ValueError(f"Sözlük {what} yazılamaz: tek bir sayı, doğru/yanlış veya metin bağlayın")
    return str(value)


def _number(value, what: str) -> float:
    """bool / number / numeric text → float (Turkish error otherwise)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"'{value}' sayı değil; {what} sayı veya doğru/yanlış bekliyor") from None


def _truth(value) -> bool:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "evet", "yes", "on", "doğru", "dogru", "açık", "acik"):
            return True
        if s in ("0", "false", "hayır", "hayir", "no", "off", "yanlış", "yanlis", "kapalı", "kapali", ""):
            return False
        try:
            return float(s.replace(",", ".")) != 0
        except ValueError:
            raise ValueError(f"'{value}' doğru/yanlış olarak anlaşılamadı") from None
    return bool(value)


def local_ip() -> str:
    from .web import local_ip as _ip
    return _ip()


def _port_error(port: int, exc: OSError, what: str) -> RuntimeError:
    if (isinstance(exc, PermissionError) or getattr(exc, "errno", None) == 13) and port < 1024:
        return RuntimeError(f"{what}: {port} portu için yetki yok. Linux'ta 1024'ün altındaki portlar (ör. 502) "
                            "yönetici (root) yetkisi ister; 1024 üstü bir port (ör. 5020) seçin")
    if (getattr(exc, "errno", None) in (98, 10048, 48) or getattr(exc, "winerror", None) == 10013
            or (isinstance(exc, PermissionError) and port >= 1024)):
        return RuntimeError(f"{what}: {port} portu kullanımda (başka bir program veya ikinci bir VisionFlow). "
                            "Başka bir port seçin")
    return RuntimeError(f"{what}: {port} portu açılamadı ({exc})")


# ================================================================================================ OPC UA server
class _OpcServer:
    """One asyncua server on its own thread + event loop; blocks talk to it through call()."""
    _servers: dict[int, "_OpcServer"] = {}
    _guard = threading.Lock()

    def __init__(self, port: int, user: str, password: str, security: str = "Yok",
                 trust: str = "İlk bağlantıda güven"):
        self.port, self.auth = port, (user, password)
        self.security, self.trust = security, trust
        self.users = 0
        self.vars: dict[str, list] = {}           # "Folder.Name" → [node, VariantType, last written value]
        self.folders: dict[str, object] = {}
        self.frames = 0
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name=f"OPC UA sunucusu :{port}", daemon=True)
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # lifecycle --------------------------------------------------------------------------------------------
    @classmethod
    def acquire(cls, port: int, user: str = "", password: str = "", security: str = "Yok",
                trust: str = "İlk bağlantıda güven") -> "_OpcServer":
        from .. import opcua_security as sec
        if not sec.is_secure(security):
            trust = ""                    # client certificates are not checked without security: any choice fits
        with cls._guard:
            srv = cls._servers.get(port)
            if srv is not None and srv.auth != (user, password):
                raise ValueError(f"{port} portundaki OPC UA sunucusu başka kullanıcı/şifre ile çalışıyor: aynı "
                                 "porttaki bütün OPC UA Sunucusu bloklarına aynı kullanıcı ve şifreyi girin")
            if srv is not None and (srv.security, srv.trust) != (security, trust):
                raise ValueError(f"{port} portundaki OPC UA sunucusu başka güvenlik ayarıyla çalışıyor: aynı porttaki "
                                 "bütün OPC UA Sunucusu bloklarına aynı güvenlik ve sertifika ayarını girin")
            if srv is None:
                srv = cls(port, user, password, security, trust)
                srv._start()
                cls._servers[port] = srv
            srv.users += 1
            return srv

    def release(self) -> None:
        with self._guard:
            self.users -= 1
            if self.users > 0:
                return
            if self._servers.get(self.port) is self:
                del self._servers[self.port]
        self._stop()

    def _start(self) -> None:
        self.thread.start()
        if not self._ready.wait(30):
            self._stop()
            raise RuntimeError(f"OPC UA sunucusu {self.port} portunda 30 sn içinde başlamadı")
        if self._error is not None:
            self.thread.join(5)
            exc = self._error
            if isinstance(exc, OSError):
                raise _port_error(self.port, exc, "OPC UA sunucusu") from exc
            raise RuntimeError(f"OPC UA sunucusu başlatılamadı: {exc}") from exc

    def _run(self) -> None:
        loop = self.loop
        asyncio.set_event_loop(loop)
        try:
            try:
                loop.run_until_complete(self._init())
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                server = getattr(self, "server", None)
                if server is not None:
                    try:
                        loop.run_until_complete(server.stop())
                    except Exception:  # noqa: BLE001
                        pass
                return
            finally:
                self._ready.set()
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:  # noqa: BLE001
                pass
            loop.close()

    async def _init(self) -> None:
        from asyncua import Server, ua
        from asyncua.crypto.permission_rules import User, UserRole

        auth_user, auth_password = self.auth

        class _Users:
            def get_user(self, iserver, username=None, password=None, certificate=None):
                if not auth_user or (username == auth_user and password == auth_password):
                    return User(role=UserRole.User)
                return None             # wrong user name or password: the session is refused

        server = Server(user_manager=_Users())
        self.server = server
        await server.init()
        server.set_endpoint(f"opc.tcp://0.0.0.0:{self.port}{OPC_PATH}")
        server.set_server_name("VisionFlow OPC UA Sunucusu")
        from .. import opcua_security as sec
        server.set_security_policy([sec.server_policy(self.security)])
        if sec.is_secure(self.security):
            sec.need_crypto()
            await server.set_application_uri(sec.app_uri("sunucu"))
            cert, key = await sec.own_certificate("sunucu")
            await server.load_certificate(str(cert))
            await server.load_private_key(str(key))
            server.set_certificate_validator(sec.validator(self.trust, "İstemci"))
        server.set_identity_tokens([ua.UserNameIdentityToken] if auth_user else [ua.AnonymousIdentityToken])
        self.ns = await server.register_namespace(OPC_NS)
        root = await server.nodes.objects.add_folder(ua.NodeId("VisionFlow", self.ns),
                                                     ua.QualifiedName("VisionFlow", self.ns))
        self.folders[""] = root
        from .. import __version__
        status = await self._folder("Durum")
        self.status = {}
        for name, value, vt in (("Calisiyor", True, ua.VariantType.Boolean),
                                ("SonGuncelleme", dt.datetime.now(dt.timezone.utc), ua.VariantType.DateTime),
                                ("KareSayisi", 0, ua.VariantType.Int64),
                                ("Surum", str(__version__), ua.VariantType.String)):
            self.status[name] = await status.add_variable(ua.NodeId(f"VisionFlow.Durum.{name}", self.ns),
                                                          ua.QualifiedName(name, self.ns), ua.Variant(value, vt))
        await server.start()

    async def _folder(self, folder: str):
        from asyncua import ua
        if folder not in self.folders:
            self.folders[folder] = await self.folders[""].add_folder(
                ua.NodeId(f"VisionFlow.{folder}", self.ns), ua.QualifiedName(folder, self.ns))
        return self.folders[folder]

    def _stop(self) -> None:
        loop = self.loop
        if self.thread.is_alive() and not loop.is_closed() and loop.is_running():
            async def stop():
                try:
                    await self.status["Calisiyor"].write_value(False)
                except Exception:  # noqa: BLE001
                    pass
                await self.server.stop()
            try:
                asyncio.run_coroutine_threadsafe(stop(), loop).result(10)
            except Exception:  # noqa: BLE001
                pass
            loop.call_soon_threadsafe(loop.stop)
        self.thread.join(10)

    def call(self, coro, timeout: float = 5.0):
        if not self.thread.is_alive():
            coro.close()
            raise RuntimeError("OPC UA sunucusu durmuş")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    @property
    def url(self) -> str:
        if not hasattr(self, "_url"):
            self._url = f"opc.tcp://{local_ip()}:{self.port}{OPC_PATH}"
        return self._url

    # variables ---------------------------------------------------------------------------------------------
    def variable(self, folder: str, name: str, vtype, initial=None, writable: bool = False) -> str:
        """Create the variable (or reuse it); returns its key. ``vtype`` None = decided by the first value."""
        key = f"{folder}.{name}" if folder else name

        async def make():
            entry = self.vars.setdefault(key, {"node": None, "type": None, "last": _UNSET, "folder": folder,
                                               "name": name, "writable": False})
            entry["writable"] = entry["writable"] or writable
            if entry["node"] is None and vtype is not None:
                await self._create(entry, vtype, initial)
            elif writable and entry["node"] is not None:
                await entry["node"].set_writable()
            return key
        return self.call(make())

    async def _create(self, entry: dict, vtype, value) -> None:
        from asyncua import ua
        key = f"{entry['folder']}.{entry['name']}" if entry["folder"] else entry["name"]
        node = await (await self._folder(entry["folder"])).add_variable(
            ua.NodeId(f"VisionFlow.{key}", self.ns), ua.QualifiedName(entry["name"], self.ns),
            ua.Variant(_opc_convert(value, vtype), vtype))
        if entry["writable"]:
            await node.set_writable()
        entry["node"], entry["type"] = node, vtype

    def publish(self, key: str, value, frame: int) -> None:
        async def write():
            from asyncua import ua
            entry = self.vars[key]
            if entry["node"] is None:             # automatic type: the first value decides
                await self._create(entry, _auto_type(value), value)
            vtype = entry["type"]
            val = _opc_convert(value, vtype)
            if val != entry["last"]:
                await entry["node"].write_value(ua.Variant(val, vtype))
                entry["last"] = val
            await self._status(frame)
        self.call(write())

    def read(self, key: str, frame: int):
        async def get():
            v = await self.vars[key]["node"].read_value()
            await self._status(frame)
            return v
        return self.call(get())

    async def _status(self, frame: int) -> None:
        from asyncua import ua
        if frame > self.frames:
            self.frames = frame
            await self.status["KareSayisi"].write_value(ua.Variant(frame, ua.VariantType.Int64))
        await self.status["SonGuncelleme"].write_value(
            ua.Variant(dt.datetime.now(dt.timezone.utc), ua.VariantType.DateTime))


_UNSET = object()
OPC_TYPES = ["Otomatik", "Ondalık (Double)", "Tamsayı (Int32)", "Mantıksal (Boolean)", "Metin (String)"]


def _opc_vtype(choice: str):
    from asyncua import ua
    return {"Ondalık (Double)": ua.VariantType.Double, "Tamsayı (Int32)": ua.VariantType.Int32,
            "Mantıksal (Boolean)": ua.VariantType.Boolean, "Metin (String)": ua.VariantType.String}.get(choice)


def _auto_type(value):
    from asyncua import ua
    if isinstance(value, bool):
        return ua.VariantType.Boolean
    if isinstance(value, int):
        return ua.VariantType.Int32 if -2 ** 31 <= value < 2 ** 31 else ua.VariantType.Double
    if isinstance(value, float):
        return ua.VariantType.Double
    return ua.VariantType.String


def _opc_convert(value, vtype):
    """A plain value as the variable's OPC UA type (Turkish error when it does not fit)."""
    from asyncua import ua
    if vtype == ua.VariantType.Boolean:
        return False if value is None else _truth(value)
    if vtype == ua.VariantType.String:
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return str(value)
    if value is None or value == "":
        num = 0.0
    else:
        num = _number(value, "OPC UA değişkeni")
    if vtype == ua.VariantType.Double:
        return num
    if not math.isfinite(num):
        raise ValueError(f"{value} tamsayı değişkene yazılamaz")
    num = int(round(num))
    lim = 2 ** 31 if vtype == ua.VariantType.Int32 else 2 ** 63
    if not -lim <= num < lim:
        raise ValueError(f"{num} değişkenin aralığı dışında (Int32: −2147483648…2147483647); "
                         "Veri tipi'ni Ondalık (Double) seçin")
    return num


def _need_asyncua():
    try:
        import asyncua  # noqa: F401  (only checks that it is installed)
    except ImportError as exc:
        raise RuntimeError("OPC UA sunucusu için 'asyncua' paketi gerekli: pip install asyncua") from exc


def _clean_name(text: str, what: str, allow_empty: bool = False) -> str:
    s = str(text or "").strip()
    if not s:
        if allow_empty:
            return ""
        raise ValueError(f"{what} boş olamaz")
    if any(c in s for c in ".;/\\"):
        raise ValueError(f"{what} '{s}' içinde . ; / \\ karakterleri olamaz")
    return s


_OPC_PARAMS = [Param("port", "int", 4840, "Port", 1, 65535),
               Param("name", "str", "Adet", "Değişken adı"),
               Param("folder", "str", "", "Klasör (opsiyonel)"),
               Param("vtype", "choice", "Otomatik", "Veri tipi", choices=OPC_TYPES),
               Param("user", "str", "", "Kullanıcı (boş = anonim)"),
               Param("password", "str", "", "Şifre"),
               Param("security", "choice", "Yok", "Güvenlik (imza / şifreleme)",
                     choices=["Yok", "Basic256Sha256 · İmzala", "Basic256Sha256 · İmzala ve Şifrele"]),
               Param("trust", "choice", "İlk bağlantıda güven", "İstemci sertifikaları",
                     choices=["İlk bağlantıda güven", "Sadece güvenilen listesi"])]


class _OpcServerNode(Node):
    setup_params = ("port", "name", "folder", "vtype", "user", "password", "security", "trust")
    writable = False

    def setup(self):
        _need_asyncua()
        name = _clean_name(self.v("name"), "Değişken adı")
        folder = _clean_name(self.v("folder"), "Klasör adı", allow_empty=True)
        if folder == "Durum":
            raise ValueError("'Durum' klasörü VisionFlow'un kendi durum değişkenlerine ayrılmış; başka bir ad seçin")
        vtype = _opc_vtype(self.v("vtype"))
        if vtype is None and self.writable:
            raise ValueError("Okunacak değişkenin tipini seçin (Otomatik olamaz: PLC hangi tipi yazacağını bilmeli)")
        initial = self._initial() if self.writable else None
        if vtype is not None and initial is not None:
            _opc_convert(initial, vtype)          # a bad initial value fails here, before the server starts
        from .. import opcua_security as sec
        security = sec.check_choice(self.v("security"))
        if sec.is_secure(security):
            sec.need_crypto()
        server = _OpcServer.acquire(self.v("port"), self.v("user").strip(), self.v("password"), security,
                                    self.v("trust"))
        try:
            key = server.variable(folder, name, vtype, initial, writable=self.writable)
        except BaseException:
            server.release()
            raise
        self.state.update(server=server, key=key, frame=0)
        print(f"[{self.title}] {server.url} → VisionFlow.{key} (ns={server.ns})")

    def _initial(self):
        return None

    def teardown(self):
        server = self.state.pop("server", None)
        if server is not None:
            server.release()


@register
class OpcUaServerPublish(_OpcServerNode):
    title = "OPC UA Sunucusu: Yayınla"
    category = CAT_OUT
    description = ("VisionFlow'u bir OPC UA sunucusu yapar: girişteki değeri (sayı, doğru/yanlış, metin) PLC, "
                   "SCADA veya HMI'nin okuyabileceği bir değişkene yazar. Adres: opc.tcp://<bu-bilgisayarın-ip>:4840"
                   "/visionflow/ — değişken: Objects/VisionFlow[/Klasör]/Ad (ns=2;s=VisionFlow.Klasör.Ad). "
                   "Durum klasöründe Calisiyor, SonGuncelleme, KareSayisi da yayınlanır. Görüntü yazılamaz; "
                   "tespit listesi adedi olarak yazılır.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL), ("url", PortType.TEXT)]
    params = _OPC_PARAMS

    def process(self, inputs):
        val = _plain(inputs["value"], "OPC UA'ya")
        self.state["frame"] += 1
        if val is None:
            return {"ok": False, "url": self.state["server"].url}
        self.state["server"].publish(self.state["key"], val, self.state["frame"])
        return {"ok": True, "url": self.state["server"].url}


@register
class OpcUaServerRead(_OpcServerNode):
    title = "OPC UA Sunucusu: Oku"
    category = CAT_IN
    description = ("VisionFlow'un OPC UA sunucusunda PLC/SCADA'nın yazabileceği bir değişken açar ve okur "
                   "(ör. reçete no, başlat komutu). 'değişti' değer değiştiği karede doğru olur. Adres: "
                   "opc.tcp://<ip>:4840/visionflow/, değişken: ns=2;s=VisionFlow.Klasör.Ad.")
    outputs = [("value", PortType.ANY), ("number", PortType.NUMBER), ("bool", PortType.BOOL),
               ("changed", PortType.BOOL)]
    params = [_OPC_PARAMS[0], Param("name", "str", "Komut", "Değişken adı"), _OPC_PARAMS[2],
              Param("vtype", "choice", "Tamsayı (Int32)", "Veri tipi", choices=OPC_TYPES[1:]),
              Param("initial", "str", "0", "Başlangıç değeri")] + _OPC_PARAMS[4:]
    setup_params = _OpcServerNode.setup_params + ("initial",)
    writable = True

    def _initial(self):
        return self.v("initial")

    def process(self, inputs):
        self.state["frame"] += 1
        v = self.state["server"].read(self.state["key"], self.state["frame"])
        if isinstance(v, str):
            try:
                num, flag = _number(v, ""), _truth(v)
            except ValueError:
                num, flag = 0.0, bool(v.strip())
        else:
            num = 0.0 if v is None else float(v)
            flag = bool(v)
        prev = self.state.get("prev", _UNSET)
        self.state["prev"] = v
        return {"value": v, "number": num, "bool": flag, "changed": prev is not _UNSET and prev != v}


# ============================================================================================ Modbus TCP server
MB_SIZE = 10000                                    # addresses 0…9999 in each of the four tables


class _MbTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _MbHandler(socketserver.BaseRequestHandler):
    def handle(self):
        srv: _ModbusServer = self.server.owner
        sock: socket.socket = self.request
        with srv.lock:
            srv.conns.add(sock)
        try:
            while True:
                head = _recv_exact(sock, 7)
                if head is None:
                    return
                tid, proto, length, unit = struct.unpack(">HHHB", head)
                if proto != 0 or not 2 <= length <= 254:
                    return                        # not Modbus TCP: drop the connection
                pdu = _recv_exact(sock, length - 1)
                if pdu is None:
                    return
                resp = srv.handle(pdu)
                sock.sendall(struct.pack(">HHHB", tid, 0, len(resp) + 1, unit) + resp)
        except OSError:
            return
        finally:
            with srv.lock:
                srv.conns.discard(sock)


def _recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _ModbusServer:
    """Modbus TCP slave: four tables in memory, shared by every block on the same port."""
    _servers: dict[int, "_ModbusServer"] = {}
    _guard = threading.Lock()

    def __init__(self, port: int):
        self.port, self.users = port, 0
        self.coils = bytearray(MB_SIZE)
        self.discrete = bytearray(MB_SIZE)
        self.holding = [0] * MB_SIZE
        self.input_regs = [0] * MB_SIZE
        self.lock = threading.Lock()
        self.conns: set = set()
        self.requests = 0
        try:
            self.tcp = _MbTCPServer(("0.0.0.0", port), _MbHandler)
        except OSError as exc:
            raise _port_error(port, exc, "Modbus TCP sunucusu") from exc
        self.tcp.owner = self
        self.thread = threading.Thread(target=self.tcp.serve_forever, kwargs={"poll_interval": 0.1},
                                       name=f"Modbus sunucusu :{port}", daemon=True)
        self.thread.start()

    @classmethod
    def acquire(cls, port: int) -> "_ModbusServer":
        with cls._guard:
            srv = cls._servers.get(port)
            if srv is None:
                srv = cls._servers[port] = cls(port)
            srv.users += 1
            return srv

    def release(self) -> None:
        with self._guard:
            self.users -= 1
            if self.users > 0:
                return
            if self._servers.get(self.port) is self:
                del self._servers[self.port]
        self.tcp.shutdown()
        with self.lock:
            conns = list(self.conns)
        for s in conns:                           # connected PLCs are dropped, their threads end
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.tcp.server_close()
        self.thread.join(5)

    @property
    def url(self) -> str:
        if not hasattr(self, "_url"):
            self._url = f"{local_ip()}:{self.port}"
        return self._url

    # protocol ----------------------------------------------------------------------------------------------
    def handle(self, pdu: bytes) -> bytes:
        fc = pdu[0]

        def exc(code):
            return bytes([fc | 0x80, code])
        with self.lock:
            self.requests += 1
            if fc in (1, 2, 3, 4):
                if len(pdu) != 5:
                    return exc(3)
                addr, count = struct.unpack(">HH", pdu[1:5])
                limit = 2000 if fc in (1, 2) else 125
                if not 1 <= count <= limit:
                    return exc(3)
                if addr + count > MB_SIZE:
                    return exc(2)
                if fc in (1, 2):
                    table = self.coils if fc == 1 else self.discrete
                    bits = table[addr:addr + count]
                    data = bytearray((count + 7) // 8)
                    for i, b in enumerate(bits):
                        if b:
                            data[i // 8] |= 1 << (i % 8)
                    return bytes([fc, len(data)]) + bytes(data)
                table = self.holding if fc == 3 else self.input_regs
                return bytes([fc, 2 * count]) + struct.pack(f">{count}H", *table[addr:addr + count])
            if fc in (5, 6):
                if len(pdu) != 5:
                    return exc(3)
                addr, value = struct.unpack(">HH", pdu[1:5])
                if addr >= MB_SIZE:
                    return exc(2)
                if fc == 5:
                    if value not in (0xFF00, 0x0000):
                        return exc(3)
                    self.coils[addr] = 1 if value else 0
                else:
                    self.holding[addr] = value
                return pdu
            if fc in (15, 16):
                if len(pdu) < 6:
                    return exc(3)
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                data = pdu[6:]
                if fc == 15:
                    if not 1 <= count <= 1968 or nbytes != (count + 7) // 8 or len(data) != nbytes:
                        return exc(3)
                    if addr + count > MB_SIZE:
                        return exc(2)
                    for i in range(count):
                        self.coils[addr + i] = data[i // 8] >> (i % 8) & 1
                else:
                    if not 1 <= count <= 123 or nbytes != 2 * count or len(data) != nbytes:
                        return exc(3)
                    if addr + count > MB_SIZE:
                        return exc(2)
                    self.holding[addr:addr + count] = struct.unpack(f">{count}H", data)
                return pdu[:5]
            return exc(1)                         # function not supported

    # VisionFlow side ---------------------------------------------------------------------------------------
    def set_bit(self, table: str, addr: int, value: bool) -> None:
        with self.lock:
            getattr(self, table)[addr] = 1 if value else 0

    def get_bit(self, table: str, addr: int) -> bool:
        with self.lock:
            return bool(getattr(self, table)[addr])

    def set_regs(self, table: str, addr: int, words: list[int]) -> None:
        with self.lock:
            getattr(self, table)[addr:addr + len(words)] = words

    def get_regs(self, table: str, addr: int, count: int) -> list[int]:
        with self.lock:
            return list(getattr(self, table)[addr:addr + count])


MB_AREAS = {"Holding register (4x)": "holding", "Bobin / coil (0x)": "coils",
            "Input register (3x)": "input_regs", "Discrete input (1x)": "discrete"}
MB_FORMATS = ["16-bit işaretli tamsayı", "16-bit işaretsiz tamsayı", "32-bit ondalık (float)",
              "32-bit işaretli tamsayı"]
MB_ORDERS = ["Yüksek kelime önce (ABCD)", "Düşük kelime önce (CDAB)"]


def mb_encode(value: float, fmt: str, order: str) -> list[int]:
    """A number as 1 or 2 register words."""
    if fmt == "32-bit ondalık (float)":
        raw = struct.pack(">f", value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{value} tamsayı register'a yazılamaz")
        n = int(round(value))
        lo, hi = {"16-bit işaretli tamsayı": (-32768, 32767), "16-bit işaretsiz tamsayı": (0, 65535),
                  "32-bit işaretli tamsayı": (-2 ** 31, 2 ** 31 - 1)}[fmt]
        if not lo <= n <= hi:
            raise ValueError(f"{n} '{fmt}' aralığı dışında ({lo}…{hi}); 32-bit veya float biçimini seçin")
        if fmt.startswith("16"):
            return [n & 0xFFFF]
        raw = struct.pack(">i", n)
    words = list(struct.unpack(">HH", raw))
    return words[::-1] if order == MB_ORDERS[1] else words


def mb_decode(words: list[int], fmt: str, order: str) -> float:
    if fmt.startswith("16"):
        w = words[0]
        return float(w - 0x10000 if fmt == MB_FORMATS[0] and w >= 0x8000 else w)
    if order == MB_ORDERS[1]:
        words = words[::-1]
    raw = struct.pack(">HH", *words)
    if fmt == "32-bit ondalık (float)":
        return float(struct.unpack(">f", raw)[0])
    return float(struct.unpack(">i", raw)[0])


class _MbServerNode(Node):
    setup_params = ("port",)

    def setup(self):
        self.state["server"] = _ModbusServer.acquire(self.v("port"))
        print(f"[{self.title}] Modbus TCP sunucusu: {self.state['server'].url}")

    def teardown(self):
        server = self.state.pop("server", None)
        if server is not None:
            server.release()

    def _where(self):
        table = MB_AREAS[self.v("area")]
        addr = self.v("address")
        width = 2 if table in ("holding", "input_regs") and self.v("format").startswith("32") else 1
        if addr + width > MB_SIZE:
            raise ValueError(f"Adres {addr} aralık dışında: 32-bit değer iki register kullanır (0…{MB_SIZE - 2})")
        return table, addr, width


_MB_PARAMS = [Param("port", "int", 5020, "Port (502 Linux'ta yönetici ister)", 1, 65535),
              Param("area", "choice", "Holding register (4x)", "Alan", choices=list(MB_AREAS)),
              Param("address", "int", 0, "Adres (0 tabanlı)", 0, MB_SIZE - 1),
              Param("format", "choice", MB_FORMATS[0], "Sayı biçimi", choices=MB_FORMATS),
              Param("order", "choice", MB_ORDERS[0], "32-bit kelime sırası", choices=MB_ORDERS)]


@register
class ModbusServerWrite(_MbServerNode):
    title = "Modbus Sunucusu: Yaz"
    category = CAT_OUT
    description = ("VisionFlow'u Modbus TCP sunucusu (slave) yapar; PLC/HMI bu bilgisayara bağlanıp değeri okur. "
                   "Girişteki sayıyı holding/input register'a (16-bit tamsayı veya iki register'da 32-bit "
                   "float/tamsayı) ya da doğru/yanlışı bobine yazar. Varsayılan port 5020 (502 Linux'ta yönetici "
                   "yetkisi ister). Adresler 0 tabanlı, 0…9999.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]
    params = _MB_PARAMS

    def process(self, inputs):
        val = _plain(inputs["value"], "Modbus'a")
        if val is None:
            return {"ok": False}
        table, addr, _ = self._where()
        server: _ModbusServer = self.state["server"]
        if table in ("coils", "discrete"):
            server.set_bit(table, addr, _truth(val))
        else:
            server.set_regs(table, addr, mb_encode(_number(val, "Modbus register'ı"), self.v("format"),
                                                   self.v("order")))
        return {"ok": True}


@register
class ModbusServerRead(_MbServerNode):
    title = "Modbus Sunucusu: Oku"
    category = CAT_IN
    description = ("VisionFlow'un Modbus TCP sunucusuna PLC'nin yazdığı holding register'ı veya bobini okur "
                   "(ör. reçete no, başlat komutu). 'değişti' değer değiştiği karede doğru olur. Port ve adresler "
                   "'Modbus Sunucusu: Yaz' ile aynı.")
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL), ("changed", PortType.BOOL)]
    params = [_MB_PARAMS[0], Param("area", "choice", "Holding register (4x)", "Alan",
                                   choices=["Holding register (4x)", "Bobin / coil (0x)"])] + _MB_PARAMS[2:]

    def process(self, inputs):
        table, addr, width = self._where()
        server: _ModbusServer = self.state["server"]
        if table == "coils":
            v = 1.0 if server.get_bit(table, addr) else 0.0
        else:
            v = mb_decode(server.get_regs(table, addr, width), self.v("format"), self.v("order"))
        if v.is_integer() and abs(v) < 2 ** 53:
            v = int(v)
        prev = self.state.get("prev", _UNSET)
        self.state["prev"] = v
        return {"value": v, "bool": bool(v), "changed": prev is not _UNSET and prev != v}
