"""EtherCAT master through the optional ``pysoem`` library (SOEM): read slave inputs and write slave outputs.

Every block of a flow that uses the same network adapter shares ONE master (an EtherCAT segment has exactly one
master). The master brings the slaves to OP and exchanges the process data in a background thread every
``cycle_ms``; the blocks only read the latest input image and change the output image, so a slow vision step never
stalls the bus. When the last block stops, the outputs are zeroed for one cycle and the slaves are put back to INIT.

pysoem needs raw Ethernet access: on Windows Npcap (WinPcap-compatible mode), on Linux root or CAP_NET_RAW
(``sudo setcap cap_net_raw,cap_net_admin+ep $(readlink -f $(which python3))``).
"""
from __future__ import annotations

import threading
import time

from ..core import Node, Param, PortType, register
from ._bytes import DTYPE_CHOICES, pack_value, unpack_value

CAT_IN, CAT_OUT = "1 · Giriş", "5 · Çıkış ve Otomasyon"
INSTALL_HELP = ("EtherCAT için 'pip install pysoem' kurun. Windows'ta ayrıca Npcap gerekir (kurulumda 'WinPcap API "
                "uyumlu mod' seçili olmalı); Linux'ta program root olarak ya da CAP_NET_RAW yetkisiyle çalışmalı.")
STATE_NAMES = {0: "YOK", 1: "INIT", 2: "PRE-OP", 3: "BOOT", 4: "SAFE-OP", 8: "OP"}
OP_TIMEOUT = 3.0          # seconds the slaves get to reach OP
STALE_AFTER = 1.0         # seconds without a complete process-data exchange before reads fail


def load_pysoem():
    try:
        import pysoem
    except ImportError as exc:
        raise RuntimeError(INSTALL_HELP) from exc
    return pysoem


def state_name(state: int) -> str:
    base = STATE_NAMES.get(int(state) & 0x0F, str(int(state) & 0x0F))
    return base + (" + HATA" if int(state) & 0x10 else "")


def list_adapters() -> list[tuple[str, str]]:
    """(name, description) of every network adapter pysoem can open."""
    mod = load_pysoem()
    return [(str(a.name), str(getattr(a, "desc", "") or "")) for a in mod.find_adapters()]


def _adapters_hint() -> str:
    try:
        ads = list_adapters()
    except Exception:  # noqa: BLE001
        return ""
    if not ads:
        return " Hiç ağ kartı bulunamadı."
    return " Ağ kartları: " + "; ".join(f"{n} ({d})" if d else n for n, d in ads)


def _slave_info(no: int, s) -> dict:
    def size(attr):
        try:
            return len(getattr(s, attr))
        except Exception:  # noqa: BLE001
            return 0
    return {"no": no, "name": str(getattr(s, "name", "") or "?"), "vendor": int(getattr(s, "man", 0) or 0),
            "product": int(getattr(s, "id", 0) or 0), "rev": int(getattr(s, "rev", 0) or 0),
            "in_bytes": size("input"), "out_bytes": size("output"), "state": state_name(getattr(s, "state", 0) or 0)}


def format_slaves(slaves: list[dict]) -> str:
    if not slaves:
        return "Slave bulunamadı"
    return "\n".join(f"{s['no']}: {s['name']}  üretici 0x{s['vendor']:08X}  ürün 0x{s['product']:08X}  "
                     f"giriş {s['in_bytes']} B  çıkış {s['out_bytes']} B  durum {s['state']}" for s in slaves)


class EcatMaster:
    """One running EtherCAT master per adapter, shared by reference count."""
    _masters: dict = {}
    _guard = threading.Lock()

    def __init__(self, adapter: str, cycle_ms: float):
        self.adapter, self.cycle_ms = adapter, float(cycle_ms)
        self.mod = load_pysoem()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.users = 0
        self.error: str | None = None
        self.cycles = 0
        self.bad_cycles = 0
        self.last_good = 0.0
        self.recoveries = 0
        self.master = None
        self.slaves: list = []
        self.outs: list[bytearray] = []
        self.ins: list[bytes] = []

    # --- sharing ----------------------------------------------------------------------------------------------
    @classmethod
    def acquire(cls, adapter: str, cycle_ms: float) -> "EcatMaster":
        adapter = (adapter or "").strip()
        if not adapter:
            raise ValueError("EtherCAT ağ kartı (adaptör) adı boş." + _adapters_hint())
        with cls._guard:
            m = cls._masters.get(adapter)
            if m is not None and m.error:                 # a dead master is not handed out again
                m._close()
                cls._masters.pop(adapter, None)
                m = None
            if m is None:
                m = cls(adapter, cycle_ms)
                m._open()
                cls._masters[adapter] = m
            elif abs(m.cycle_ms - float(cycle_ms)) > 1e-9:
                print(f"EtherCAT: {adapter} zaten {m.cycle_ms:g} ms çevrimle çalışıyor; bu blok da onu kullanıyor")
            m.users += 1
            return m

    @classmethod
    def active(cls, adapter: str) -> "EcatMaster | None":
        with cls._guard:
            return cls._masters.get((adapter or "").strip())

    def release(self):
        with self._guard:
            self.users -= 1
            if self.users > 0:
                return
            if self._masters.get(self.adapter) is self:
                self._masters.pop(self.adapter, None)
        self._close()

    # --- start / stop -----------------------------------------------------------------------------------------
    def _open(self):
        mod = self.mod
        m = mod.Master()
        try:
            m.open(self.adapter)
        except PermissionError as exc:
            raise PermissionError(f"{self.adapter} açılamadı: yetki yok. " + INSTALL_HELP) from exc
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(f"EtherCAT ağ kartı '{self.adapter}' açılamadı ({exc})." + _adapters_hint()) \
                from exc
        self.master = m
        try:
            n = m.config_init()
            if not n or n <= 0:
                raise ConnectionError(f"{self.adapter} üzerinde EtherCAT slave bulunamadı (kablo, besleme ve doğru "
                                      "ağ kartını kontrol edin)")
            self.slaves = list(m.slaves)
            m.config_map()
            if m.state_check(mod.SAFEOP_STATE, 50000) != mod.SAFEOP_STATE:
                raise ConnectionError("Slave'ler SAFE-OP durumuna geçmedi: " + self._not_in(mod.SAFEOP_STATE))
            self.outs = [bytearray(len(s.output)) for s in self.slaves]
            self.ins = [bytes(s.input) for s in self.slaves]
            self.last_good = time.monotonic()
            self.thread = threading.Thread(target=self._cycle, name=f"EtherCAT {self.adapter}", daemon=True)
            self.thread.start()
            m.state = mod.OP_STATE
            m.write_state()
            deadline = time.monotonic() + OP_TIMEOUT
            while m.state_check(mod.OP_STATE, 50000) != mod.OP_STATE:
                if self.error:
                    raise ConnectionError(self.error)
                if time.monotonic() > deadline:
                    raise ConnectionError(f"Slave'ler {OP_TIMEOUT:g} s içinde OP durumuna geçmedi: "
                                          + self._not_in(mod.OP_STATE))
                time.sleep(0.01)
        except Exception:
            self._close()
            raise

    def _not_in(self, wanted: int) -> str:
        try:
            self.master.read_state()
        except Exception:  # noqa: BLE001
            pass
        bad = [f"{i + 1}:{getattr(s, 'name', '?')} {state_name(getattr(s, 'state', 0) or 0)}"
               + (f" (AL durum kodu 0x{s.al_status:04X})" if getattr(s, "al_status", 0) else "")
               for i, s in enumerate(self.slaves) if (getattr(s, "state", 0) or 0) != wanted]
        return ", ".join(bad) or "bilinmeyen neden"

    def _close(self):
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None
        m, self.master = self.master, None
        if m is None:
            return
        mod = self.mod
        try:                                     # safe stop: one cycle with every output at 0, then INIT
            if self.slaves and self.outs and not self.error:
                for s, buf in zip(self.slaves, self.outs):
                    if len(buf):
                        s.output = bytes(len(buf))
                m.send_processdata()
                m.receive_processdata(2000)
        except Exception:  # noqa: BLE001
            pass
        try:
            m.state = mod.INIT_STATE
            m.write_state()
        except Exception:  # noqa: BLE001
            pass
        try:
            m.close()
        except Exception:  # noqa: BLE001
            pass

    # --- process-data cycle (background thread) ---------------------------------------------------------------
    def _expected_wkc(self):
        exp = getattr(self.master, "expected_wkc", None)
        return exp if isinstance(exp, int) and exp > 0 else None

    def cycle_once(self) -> bool:
        """One exchange: outputs out, inputs in. True when every slave took part (working counter complete)."""
        m = self.master
        with self.lock:
            for s, buf in zip(self.slaves, self.outs):
                if len(buf):
                    s.output = bytes(buf)
        m.send_processdata()
        wkc = m.receive_processdata(max(1000, int(self.cycle_ms * 1000)))
        exp = self._expected_wkc()
        good = wkc >= exp if exp else wkc > 0
        with self.lock:
            self.cycles += 1
            if good:
                self.ins = [bytes(s.input) for s in self.slaves]
                self.bad_cycles = 0
                self.last_good = time.monotonic()
            else:
                self.bad_cycles += 1
        return good

    def recover(self):
        """Slaves that dropped out of OP (cable pulled, watchdog) are acknowledged and asked for OP again."""
        mod, m = self.mod, self.master
        m.read_state()
        for s in self.slaves:
            st = getattr(s, "state", 0) or 0
            if st == mod.SAFEOP_STATE + mod.STATE_ERROR:
                s.state = mod.SAFEOP_STATE + mod.STATE_ACK
                s.write_state()
            elif st == mod.SAFEOP_STATE:
                s.state = mod.OP_STATE
                s.write_state()
            elif st == getattr(mod, "NONE_STATE", 0) and hasattr(s, "recover"):
                try:
                    s.recover()
                except Exception:  # noqa: BLE001
                    pass
        self.recoveries += 1

    def _cycle(self):
        period = max(0.0005, self.cycle_ms / 1000.0)
        next_t = time.monotonic()
        last_recover = 0.0
        try:
            while not self.stop_event.is_set():
                good = self.cycle_once()
                now = time.monotonic()
                if not good and now - last_recover > 0.5:
                    last_recover = now
                    try:
                        self.recover()
                    except Exception:  # noqa: BLE001
                        pass
                next_t += period
                delay = next_t - time.monotonic()
                if delay < -period:                    # far behind (PC busy): do not try to catch up in a burst
                    next_t = time.monotonic()
                elif delay > 0:
                    self.stop_event.wait(delay)
        except Exception as exc:  # noqa: BLE001
            self.error = f"EtherCAT çevrimi durdu: {exc}"

    # --- block access -----------------------------------------------------------------------------------------
    def _slave(self, no) -> int:
        n = len(self.slaves)
        if isinstance(no, bool) or not isinstance(no, int) or not 1 <= no <= n:
            raise ValueError(f"Slave numarası {no} geçersiz: ağda {n} slave var (1…{n})")
        return no - 1

    def _check_alive(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.bad_cycles and time.monotonic() - self.last_good > STALE_AFTER:
            raise RuntimeError(f"EtherCAT haberleşmesi kesik: {time.monotonic() - self.last_good:.1f} s'dir eksik "
                               "cevap (kablo, slave beslemesi veya OP dışına düşen slave)")

    def read(self, no: int, offset: int, dtype: str, bit: int = 0):
        self._check_alive()
        i = self._slave(no)
        with self.lock:
            data = self.ins[i]
        try:
            return unpack_value(data, offset, dtype, bit)
        except ValueError as exc:
            raise ValueError(f"Slave {no} girişleri ({len(data)} bayt): {exc}") from None

    def write(self, no: int, offset: int, dtype: str, value, bit: int = 0):
        self._check_alive()
        i = self._slave(no)
        with self.lock:
            buf = bytearray(self.outs[i])            # changed on a copy: a bad value leaves the image untouched
            try:
                v = pack_value(buf, offset, dtype, value, bit)
            except ValueError as exc:
                raise ValueError(f"Slave {no} çıkışları ({len(buf)} bayt): {exc}") from None
            self.outs[i][:] = buf
        return v

    def slave_list(self) -> list[dict]:
        return [_slave_info(i + 1, s) for i, s in enumerate(self.slaves)]


def scan(adapter: str) -> list[dict]:
    """The slaves on ``adapter``: of the running master when one is active, else of a short look at the bus."""
    adapter = (adapter or "").strip()
    if not adapter:
        raise ValueError("EtherCAT ağ kartı (adaptör) adı boş." + _adapters_hint())
    live = EcatMaster.active(adapter)
    if live is not None:
        return live.slave_list()
    mod = load_pysoem()
    m = mod.Master()
    try:
        m.open(adapter)
    except Exception as exc:  # noqa: BLE001
        raise ConnectionError(f"EtherCAT ağ kartı '{adapter}' açılamadı ({exc})." + _adapters_hint()) from exc
    try:
        n = m.config_init()
        if not n or n <= 0:
            return []
        try:
            m.config_map()                          # fills the input/output sizes
        except Exception:  # noqa: BLE001
            pass
        return [_slave_info(i + 1, s) for i, s in enumerate(m.slaves)]
    finally:
        try:
            m.close()
        except Exception:  # noqa: BLE001
            pass


# --- blocks ---------------------------------------------------------------------------------------------------

ADAPTER = Param("adapter", "str", "", "Ağ kartı (Linux: eth0, Windows: \\Device\\NPF_{...})")
CYCLE = Param("cycle_ms", "int", 4, "Çevrim süresi (ms)", 1, 1000)
DATA_PARAMS = [Param("slave", "int", 1, "Slave no (1'den başlar)", 1, 65535),
               Param("offset", "int", 0, "Bayt adresi", 0, 65535),
               Param("dtype", "choice", "uint8", "Veri tipi", choices=DTYPE_CHOICES),
               Param("bit", "int", 0, "Bit (bool için)", 0, 7)]


class _EcatBase(Node):
    setup_params = ("adapter", "cycle_ms")
    params = [ADAPTER, CYCLE] + DATA_PARAMS

    def setup(self):
        self.state["master"] = EcatMaster.acquire(self.v("adapter"), self.v("cycle_ms"))

    def teardown(self):
        m = self.state.pop("master", None)
        if m is not None:
            m.release()


@register
class EtherCatRead(_EcatBase):
    title = "EtherCAT Oku"
    category = CAT_IN
    description = ("EtherCAT slave'inin giriş süreç verisinden (TxPDO) bir değer okur: bit, uint8, int16, uint16, "
                   "int32, float32 (EtherCAT bayt sırası: little-endian). Aynı ağ kartını kullanan bütün EtherCAT "
                   "blokları tek bir master paylaşır; veri arka planda her çevrimde yenilenir. Adresleri "
                   "'EtherCAT Ağını Tara' bloğu ve slave'in ESI/PDO dökümanından bulun. 'pip install pysoem' gerekir; "
                   "Windows'ta Npcap, Linux'ta root/CAP_NET_RAW.")
    outputs = [("value", PortType.NUMBER), ("bool", PortType.BOOL)]

    def process(self, inputs):
        v = self.state["master"].read(self.v("slave"), self.v("offset"), self.v("dtype"), self.v("bit"))
        num = round(v, 6) if isinstance(v, float) else int(v)
        return {"value": num, "bool": bool(v)}


@register
class EtherCatWrite(_EcatBase):
    title = "EtherCAT Yaz"
    category = CAT_OUT
    description = ("EtherCAT slave'inin çıkış süreç verisine (RxPDO) bir değer yazar (ör. dijital çıkış modülünde "
                   "ayırıcı valfi). Değer, master'ın bir sonraki çevriminde gönderilir; akış durunca bütün çıkışlar "
                   "0 yapılıp slave'ler INIT'e alınır. Aralık dışı değer (uint8'e 300 gibi) yazılmaz, hata verir. "
                   "'pip install pysoem' gerekir.")
    inputs = [("value", PortType.ANY)]
    outputs = [("ok", PortType.BOOL)]

    def process(self, inputs):
        val = inputs.get("value")
        if val is None:
            return {"ok": False}
        self.state["master"].write(self.v("slave"), self.v("offset"), self.v("dtype"), val, self.v("bit"))
        return {"ok": True}


@register
class EtherCatScan(Node):
    title = "EtherCAT Ağını Tara"
    category = CAT_IN
    description = ("Ağ kartlarını ve seçilen karttaki EtherCAT slave'lerini listeler: sıra no, ad, üretici (vendor) "
                   "kimliği, ürün kodu, giriş/çıkış bayt sayısı, durum. Ağ kartı boşsa yalnızca kartları listeler. "
                   "Akış başına bir kez tarar; 'tara' girişi yükselince yeniden tarar. 'pip install pysoem' gerekir.")
    inputs = [("scan", PortType.BOOL)]
    outputs = [("text", PortType.TEXT), ("count", PortType.NUMBER)]
    params = [ADAPTER]

    def setup(self):
        self.state.update(result=None, prev=False)

    def process(self, inputs):
        trig = bool(inputs.get("scan"))
        if self.state.get("result") is None or (trig and not self.state.get("prev")):
            adapter = self.v("adapter").strip()
            if not adapter:
                ads = list_adapters()
                text = "\n".join(f"{n}  {d}" for n, d in ads) or "Ağ kartı bulunamadı"
                self.state["result"] = (text, len(ads))
            else:
                slaves = scan(adapter)
                self.state["result"] = (format_slaves(slaves), len(slaves))
        self.state["prev"] = trig
        text, count = self.state["result"]
        return {"text": text, "count": count}
