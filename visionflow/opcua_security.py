"""OPC UA security for the OPC UA client and server blocks: policy choice, the program's own self-signed application
certificates (in <VisionFlow folder>/pki) and trust of the other side's certificate.

Layout of <home>/pki:
  own/      client_cert.der, client_key.pem, server_cert.der, server_key.pem (made on first use, valid 1 year,
            renewed automatically when expired or the computer's name changes)
  trusted/  certificates of servers/clients that are trusted (.der); copy one here to trust it
  rejected/ certificates that were refused (move one to trusted/ to accept it)
"""
from __future__ import annotations

import asyncio
import hashlib
import pathlib
import re
import socket
import threading

SEC_NONE = "Yok"
SEC_SIGN = "Basic256Sha256 · İmzala"
SEC_ENCRYPT = "Basic256Sha256 · İmzala ve Şifrele"
SECURITY_CHOICES = [SEC_NONE, SEC_SIGN, SEC_ENCRYPT]

TRUST_FIRST = "İlk bağlantıda güven"
TRUST_STRICT = "Sadece güvenilen listesi"
TRUST_CHOICES = [TRUST_FIRST, TRUST_STRICT]

CLIENT_URI = "urn:visionflow:{host}:istemci"
SERVER_URI = "urn:visionflow:{host}:sunucu"

_LOCK = threading.Lock()


def pki_dir() -> pathlib.Path:
    from .ai.config import home
    return home() / "pki"


def is_secure(choice: str) -> bool:
    return choice in (SEC_SIGN, SEC_ENCRYPT)


def check_choice(choice: str) -> str:
    if choice not in SECURITY_CHOICES:
        raise ValueError(f"Bilinmeyen OPC UA güvenlik ayarı: {choice!r} (seçenekler: {', '.join(SECURITY_CHOICES)})")
    return choice


def need_crypto() -> None:
    """Security needs the 'cryptography' package (asyncua uses it for certificates and encryption)."""
    try:
        import cryptography  # noqa: F401
        from asyncua.crypto import cert_gen  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("OPC UA güvenliği (imza/şifreleme) için 'cryptography' paketi gerekli: "
                           "pip install cryptography") from exc


def hostname() -> str:
    return socket.gethostname() or "localhost"


def app_uri(role: str) -> str:
    return (SERVER_URI if role == "sunucu" else CLIENT_URI).format(host=re.sub(r"[^\w.-]+", "-", hostname()))


def server_policy(choice: str):
    """asyncua SecurityPolicyType for a server endpoint."""
    from asyncua import ua
    return {SEC_NONE: ua.SecurityPolicyType.NoSecurity,
            SEC_SIGN: ua.SecurityPolicyType.Basic256Sha256_Sign,
            SEC_ENCRYPT: ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt}[check_choice(choice)]


def client_mode(choice: str):
    from asyncua import ua
    return ua.MessageSecurityMode.Sign if choice == SEC_SIGN else ua.MessageSecurityMode.SignAndEncrypt


async def own_certificate(role: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(certificate .der, private key .pem) of this program as OPC UA 'istemci' or 'sunucu', made when missing."""
    need_crypto()
    from asyncua.crypto import cert_gen
    from cryptography.x509.oid import ExtendedKeyUsageOID
    name = "server" if role == "sunucu" else "client"
    own = pki_dir() / "own"
    for sub in ("own", "trusted", "rejected"):          # where the user puts / finds certificates
        (pki_dir() / sub).mkdir(parents=True, exist_ok=True)
    cert, key = own / f"{name}_cert.der", own / f"{name}_key.pem"
    use = [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH] if role == "sunucu" \
        else [ExtendedKeyUsageOID.CLIENT_AUTH]
    await cert_gen.setup_self_signed_certificate(key, cert, app_uri(role), hostname(), use,
                                                 {"organizationName": "VisionFlow"})
    try:
        key.chmod(0o600)                      # the private key: only this user
    except OSError:
        pass
    return cert, key


def own_certificate_sync(role: str) -> tuple[pathlib.Path, pathlib.Path]:
    with _LOCK:
        return asyncio.run(own_certificate(role))


# ----------------------------------------------------------------------------------------------- trust store
def fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def _uri_of(cert) -> str:
    from cryptography import x509
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        uris = san.value.get_values_for_type(x509.UniformResourceIdentifier)
        if uris:
            return uris[0]
    except x509.ExtensionNotFound:
        pass
    return cert.subject.rfc4514_string()


def _safe(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text)[:60] or "sertifika"


def _trusted() -> dict[str, tuple[str, pathlib.Path]]:
    """fingerprint → (application URI, file) of every trusted certificate."""
    from cryptography import x509
    out = {}
    folder = pki_dir() / "trusted"
    for f in sorted(folder.glob("*")) if folder.is_dir() else []:
        if f.suffix.lower() not in (".der", ".pem", ".crt", ".cer"):
            continue
        try:
            raw = f.read_bytes()
            cert = x509.load_pem_x509_certificate(raw) if raw.lstrip().startswith(b"-----") \
                else x509.load_der_x509_certificate(raw)
        except (OSError, ValueError):
            continue
        from cryptography.hazmat.primitives.serialization import Encoding
        out[fingerprint(cert.public_bytes(Encoding.DER))] = (_uri_of(cert), f)
    return out


def check_trust(cert, uri: str, mode: str, who: str) -> str:
    """Accept or refuse the other side's certificate. ``mode`` TRUST_FIRST trusts a peer's first certificate
    (and refuses a different one later for the same application URI); TRUST_STRICT only accepts certificates in
    pki/trusted. Returns 'güvenilen' or 'yeni güvenildi'; raises PermissionError with a Turkish message."""
    from cryptography.hazmat.primitives.serialization import Encoding
    der = cert.public_bytes(Encoding.DER)
    fp = fingerprint(der)
    uri = _uri_of(cert) or uri       # the URI in the certificate itself: what the trusted files are compared by
    with _LOCK:
        trusted = _trusted()
        if fp in trusted:
            return "güvenilen"
        name = f"{_safe(uri)}_{fp[:16]}.der"
        same_uri = [f for t_uri, f in trusted.values() if t_uri == uri]
        if mode == TRUST_FIRST and not same_uri:
            folder = pki_dir() / "trusted"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(der)
            return "yeni güvenildi"
        rejected = pki_dir() / "rejected"
        rejected.mkdir(parents=True, exist_ok=True)
        (rejected / name).write_bytes(der)
    where = f"{pki_dir() / 'rejected' / name}"
    if same_uri:
        raise PermissionError(f"{who} sertifikası değişmiş ({uri}); güvenilen sertifikayla aynı değil. Bilinçli bir "
                              f"değişiklikse eski sertifikayı pki/trusted klasöründen silip yenisini ({where}) "
                              "oraya taşıyın.")
    raise PermissionError(f"{who} sertifikası güvenilenler listesinde yok ({uri}). Güvenmek için {where} dosyasını "
                          f"{pki_dir() / 'trusted'} klasörüne taşıyın.")


def validator(mode: str, who: str):
    """An asyncua certificate validator (client: the server's certificate; server: the client's)."""
    async def validate(cert, app_description):
        from asyncua import ua
        from asyncua.common.utils import ServiceError
        uri = getattr(app_description, "ApplicationUri", "") or ""
        try:
            check_trust(cert, uri, mode, who)
        except PermissionError as exc:
            print(f"[OPC UA] {exc}")
            raise ServiceError(ua.StatusCodes.BadCertificateUntrusted) from None
    return validate


# ----------------------------------------------------------------------------------------------- client
def secure_client(client, choice: str, trust: str) -> None:
    """Configure an asyncua.sync.Client for the chosen security (before connect()). Nothing for 'Yok'."""
    if not is_secure(check_choice(choice)):
        return
    need_crypto()
    if trust not in TRUST_CHOICES:
        raise ValueError(f"Bilinmeyen güven ayarı: {trust!r}")
    from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
    aio = getattr(client, "aio_obj", client)
    aio.application_uri = app_uri("istemci")
    cert, key = own_certificate_sync("istemci")
    try:
        client.set_security(SecurityPolicyBasic256Sha256, str(cert), str(key), mode=client_mode(choice))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OPC UA sunucusu '{choice}' güvenliğini sunmuyor ya da ulaşılamadı: "
                           f"{type(exc).__name__}: {exc}") from None
    aio.certificate_validator = validator(trust, "Sunucu")


def friendly_error(exc: BaseException, choice: str) -> str | None:
    """A Turkish explanation of a failed secure/insecure connection; None when there is nothing to add."""
    text = f"{type(exc).__name__}: {exc}"
    if "BadCertificateUntrusted" in text or "Untrusted" in text:
        return ("Sertifika güvenilmedi: karşı taraf VisionFlow'un sertifikasını (ya da VisionFlow onunkini) kabul "
                f"etmedi. pki/rejected klasörüne bakın ({pki_dir()}). Ayrıntı: {text}")
    if "BadSecurityPolicyRejected" in text or "BadSecurityModeRejected" in text or "No matching endpoints" in text:
        hint = ("sunucu imza/şifreleme istiyor olabilir: 'Güvenlik' ayarını sunucununkiyle aynı yapın"
                if not is_secure(choice) else "sunucunun güvenlik ayarlarına bakın")
        return f"Sunucu '{choice}' güvenlik ayarını sunmuyor ({hint}). Ayrıntı: {text}"
    return None
