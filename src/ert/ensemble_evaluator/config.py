import ipaddress
import logging
import os
import socket
import uuid
import warnings
from base64 import b64encode
from datetime import datetime, timedelta

import zmq
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
) -> tuple[str, bytes, bytes]:
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
        .not_valid_before(datetime.now(UTC))
        .not_valid_after(datetime.now(UTC) + timedelta(days=365))  # 1 year
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
    def __init__(
        self,
        custom_port_range: range | None = None,
        use_token: bool = True,
        custom_host: str | None = None,
        use_ipc_protocol: bool = True,
    ) -> None:
        self.host: str | None = None
        self.router_port: int | None = None
        self.url = f"ipc:///tmp/socket-{uuid.uuid4().hex[:8]}"
        self.token: str | None = None

        self.server_public_key: bytes | None = None
        self.server_secret_key: bytes | None = None
        if not use_ipc_protocol:
            self._socket_handle = find_available_socket(
                custom_range=custom_port_range,
                custom_host=custom_host,
                will_close_then_reopen_socket=True,
            )
            self.host, self.router_port = self._socket_handle.getsockname()
            self.url = f"tcp://{self.host}:{self.router_port}"

        if use_token:
            self.server_public_key, self.server_secret_key = zmq.curve_keypair()
            self.token = self.server_public_key.decode("utf-8")

    def get_socket(self) -> socket.socket:
        return self._socket_handle.dup()

    def get_connection_info(self) -> EvaluatorConnectionInfo:
        return EvaluatorConnectionInfo(
            self.url,
            self.token,
        )
