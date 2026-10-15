"""Reading and writing typed values in a process-data byte image (EtherCAT PDO, Profinet IO data, S7 DB...).

All checks raise ValueError with a Turkish message, so a block shows the user exactly what is wrong (offset past the
end of the slave's data, 300 written to a uint8, 2.5 written to an int16...) instead of writing garbage.
"""
from __future__ import annotations

import math
import numbers
import struct

# data type → (struct code, size in bytes, min, max); "bool" is one bit of one byte
DTYPES = {
    "bool": ("B", 1, 0, 1),
    "uint8": ("B", 1, 0, 0xFF),
    "int8": ("b", 1, -0x80, 0x7F),
    "uint16": ("H", 2, 0, 0xFFFF),
    "int16": ("h", 2, -0x8000, 0x7FFF),
    "uint32": ("I", 4, 0, 0xFFFFFFFF),
    "int32": ("i", 4, -0x80000000, 0x7FFFFFFF),
    "float32": ("f", 4, None, None),
}
DTYPE_CHOICES = list(DTYPES)
FLOAT32_MAX = 3.4028234663852886e38


def dtype_size(dtype: str) -> int:
    if dtype not in DTYPES:
        raise ValueError(f"Bilinmeyen veri tipi '{dtype}' (seçenekler: {', '.join(DTYPES)})")
    return DTYPES[dtype][1]


def _endian(endian: str) -> str:
    if endian in ("little", "<"):
        return "<"
    if endian in ("big", ">"):
        return ">"
    raise ValueError(f"Bayt sırası 'little' veya 'big' olmalı, '{endian}' verildi")


def _check_place(length: int, offset, dtype: str, bit) -> tuple[int, int]:
    size = dtype_size(dtype)
    if isinstance(offset, bool) or not isinstance(offset, numbers.Integral):
        raise ValueError(f"Bayt adresi tam sayı olmalı, '{offset}' verildi")
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"Bayt adresi negatif olamaz ({offset})")
    if offset + size > length:
        if length == 0:
            raise ValueError("Bu alanda hiç veri baytı yok (cihazın bu yönde süreç verisi tanımlı değil)")
        raise ValueError(f"Bayt adresi {offset} + {size} bayt ({dtype}) veri alanının dışında: "
                         f"alan {length} bayt (geçerli adres 0…{length - size})")
    if dtype == "bool":
        if isinstance(bit, bool) or not isinstance(bit, numbers.Integral) or not 0 <= int(bit) <= 7:
            raise ValueError(f"Bit numarası 0…7 olmalı, '{bit}' verildi")
        bit = int(bit)
    return offset, bit


def unpack_value(buf, offset: int, dtype: str, bit: int = 0, endian: str = "little"):
    """The value at ``offset`` of ``buf`` (bytes/bytearray): bool for "bool", int for integers, float for float32."""
    if buf is None:
        raise ValueError("Okunacak veri yok")
    data = bytes(buf)
    offset, bit = _check_place(len(data), offset, dtype, bit)
    if dtype == "bool":
        return bool(data[offset] >> bit & 1)
    code = DTYPES[dtype][0]
    value = struct.unpack_from(_endian(endian) + code, data, offset)[0]
    return float(value) if dtype == "float32" else int(value)


def coerce_value(value, dtype: str):
    """``value`` checked and converted for ``dtype``: an int in range, a float that fits float32, or a bool."""
    dtype_size(dtype)
    if value is None:
        raise ValueError("Yazılacak değer yok (giriş bağlı değil veya boş)")
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):   # numpy scalar / 1-element array
        try:
            value = value.item()
        except (ValueError, AttributeError):
            raise ValueError("Tek bir sayı bekleniyor, dizi verildi") from None
    if isinstance(value, str):
        s = value.strip().lower().replace(",", ".")
        if dtype == "bool" and s in ("true", "evet", "on", "doğru", "false", "hayır", "off", "yanlış"):
            return s in ("true", "evet", "on", "doğru")
        try:
            value = int(s, 0)
        except ValueError:
            try:
                value = float(s)
            except ValueError:
                raise ValueError(f"'{value}' bir sayı değil") from None
    if not isinstance(value, (numbers.Real, bool)):
        raise ValueError(f"Sayı bekleniyor, {type(value).__name__} verildi")
    if dtype == "bool":
        return bool(value)
    if dtype == "float32":
        value = float(value)
        if math.isnan(value):
            raise ValueError("NaN (sayı değil) yazılamaz")
        if math.isfinite(value) and abs(value) > FLOAT32_MAX:
            raise ValueError(f"{value} float32 aralığının dışında (±{FLOAT32_MAX:.3g})")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{dtype} için tam sayı gerekli, {value} verildi")
    value = int(value)
    lo, hi = DTYPES[dtype][2], DTYPES[dtype][3]
    if not lo <= value <= hi:
        raise ValueError(f"{value} değeri {dtype} aralığının dışında ({lo}…{hi})")
    return value


def pack_value(buf: bytearray, offset: int, dtype: str, value, bit: int = 0, endian: str = "little"):
    """Writes ``value`` into ``buf`` at ``offset`` in place (only the addressed bit for "bool"); returns the value
    actually written."""
    if not isinstance(buf, bytearray):
        raise TypeError("pack_value bir bytearray ister")
    offset, bit = _check_place(len(buf), offset, dtype, bit)
    value = coerce_value(value, dtype)
    if dtype == "bool":
        if value:
            buf[offset] |= 1 << bit
        else:
            buf[offset] &= ~(1 << bit) & 0xFF
        return value
    struct.pack_into(_endian(endian) + DTYPES[dtype][0], buf, offset, value)
    return value
