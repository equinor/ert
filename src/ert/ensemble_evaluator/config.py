import ipaddress
import logging
import os
import pathlib
import socket
import ssl
import tempfile
import typing
import warnings
from base64 import b64encode
from datetime import datetime, timedelta
from typing import Optional

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ert.shared import find_available_socket
from ert.shared import get_machine_name as ert_shared_get_machine_name

from .evaluator_connection_info import EvaluatorConnectionInfo

logger = logging.getLogger(__name__)


def get_machine_name() -> str:
    warnings.warn(
        "get_machine_name has been moved from ert.ensemble_evaluator.config to ert.shared",
        DeprecationWarning,
        stacklevel=2,
    )
    return ert_shared_get_machine_name()


def _generate_authentication() -> str:
    n_bytes = 128
    random_bytes = bytes(os.urandom(n_bytes))
    token = b64encode(random_bytes).decode("utf-8")
    return token


def _generate_certificate(
    ip_address: str,
) -> typing.Tuple[str, bytes, bytes]:
    """Generate a private key and a certificate signed with it
    The key is encrypted before being stored.
    Returns the certificate as a string, the key as bytes (encrypted), and
    the password used for encrypting the key
    """
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=4096, backend=default_backend()
    )

    # Generate the certificate and sign it with the private key
    cert_name = ert_shared_get_machine_name()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NO"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bergen"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sandsli"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Equinor"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"{cert_name}"),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))  # 1 year
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(f"{cert_name}"),
                    x509.DNSName(ip_address),
                    x509.IPAddress(ipaddress.ip_address(ip_address)),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    cert_str = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    pw = bytes(os.urandom(28))
    key_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.BestAvailableEncryption(pw),
    )
    return cert_str, key_bytes, pw


class EvaluatorServerConfig:
    """
    This class is responsible for identifying a host:port-combo and then provide
    low-level sockets bound to said combo. The problem is that these sockets may
    be closed by underlying code, while the EvaluatorServerConfig-instance is
    still alive and expected to provide a bound low-level socket. Thus we risk
    that the host:port is hijacked by another process in the meantime.

    To prevent this, we keep a handle to the bound socket and every time
    a socket is requested we return a duplicate of this. The duplicate will be
    bound similarly to the handle, but when closed the handle stays open and
    holds the port.

    In particular, the websocket-server closes the websocket when exiting a
    context:

       https://github.com/aaugustin/websockets/blob/c439f1d52aafc05064cc11702d1c3014046799b0/src/websockets/legacy/server.py#L890

    and digging into the cpython-implementation of asyncio, we see that causes
    the asyncio code to also close the underlying socket:

       https://github.com/python/cpython/blob/b34dd58fee707b8044beaf878962a6fa12b304dc/Lib/asyncio/selector_events.py#L607-L611

    """

    def __init__(
        self,
        custom_port_range: typing.Optional[range] = None,
        use_token: bool = True,
        generate_cert: bool = True,
        custom_host: typing.Optional[str] = None,
    ) -> None:
        self._socket_handle = find_available_socket(
            custom_range=custom_port_range, custom_host=custom_host
        )
        host, port = self._socket_handle.getsockname()
        self.protocol = "wss" if generate_cert else "ws"
        self.url = f"{self.protocol}://{host}:{port}"
        self.client_uri = f"{self.url}/client"
        self.dispatch_uri = f"{self.url}/dispatch"
        if generate_cert:
            cert, key, pw = _generate_certificate(host)
        else:
            cert, key, pw = None, None, None
        self.cert = cert
        self._key: Optional[bytes] = key
        self._key_pw = pw

        self.token = _generate_authentication() if use_token else None

    def get_socket(self) -> socket.socket:
        return self._socket_handle.dup()

    def get_connection_info(self) -> EvaluatorConnectionInfo:
        return EvaluatorConnectionInfo(
            self.url,
            self.cert,
            self.token,
        )

    def get_server_ssl_context(
        self, protocol: int = ssl.PROTOCOL_TLS_SERVER
    ) -> typing.Optional[ssl.SSLContext]:
        if self.cert is None:
            return None
        backup_default_tmp = tempfile.tempdir
        try:
            tempfile.tempdir = os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = pathlib.Path(tmp_dir)
                cert_path = tmp_path / "ee.crt"
                with open(cert_path, "w", encoding="utf-8") as filehandle_1:
                    filehandle_1.write(self.cert)

                key_path = tmp_path / "ee.key"
                if self._key is not None:
                    with open(key_path, "wb") as filehandle_2:
                        filehandle_2.write(self._key)
                context = ssl.SSLContext(protocol=protocol)
                context.load_cert_chain(cert_path, key_path, self._key_pw)
                return context
        finally:
            tempfile.tempdir = backup_default_tmp
