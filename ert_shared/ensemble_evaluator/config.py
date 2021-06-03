import yaml
import ipaddress
import logging
import socket
from ert_shared.storage.main import bind_socket
import tempfile

import os
import pathlib
import socket
import ssl
import typing
from base64 import b64encode
from datetime import datetime, timedelta

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dns import resolver, reversename, exception

logger = logging.getLogger(__name__)


def _get_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def find_open_port(lower: int = 51820, upper: int = 51840) -> int:
    host = _get_ip_address()
    for port in range(lower, upper):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            return port
        except socket.error:
            pass
    msg = f"No open port for host {host} in the range {lower}-{upper}"
    logging.exception(msg)
    raise Exception(msg)


def get_machine_name():
    """Returns a name that can be used to identify this machine in a network
    A fully qualified domain name is returned if available. Otherwise returns
    the string `localhost`
    """
    hostname = socket.gethostname()
    try:
        # We need the ip-address to perform a reverse lookup to deal with
        # differences in how the clusters are getting their fqdn's
        ip_addr = socket.gethostbyname(hostname)
        rev_name = reversename.from_address(ip_addr)
        resolved_host = str(resolver.resolve(rev_name, "PTR")[0])
        if resolved_host[-1] == ".":
            resolved_host = resolved_host[:-1]
        return resolved_host
    except (resolver.NXDOMAIN, exception.Timeout):
        # If local address and reverse lookup not working - fallback
        # to socket fqdn which are using /etc/hosts to retrieve this name
        return socket.getfqdn()
    except socket.gaierror:
        return "localhost"


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
    cert_name = get_machine_name()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NO"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bergen"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sandsli"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Equinor"),
            x509.NameAttribute(NameOID.COMMON_NAME, "{}".format(cert_name)),
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
                    x509.DNSName("{}".format(cert_name)),
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
        self, port: int = None, use_token: bool = True, generate_cert: bool = True
    ) -> None:
        self.host = _get_ip_address()
        self.port = find_open_port() if port is None else port
        self.protocol = "wss" if generate_cert else "ws"
        self.url = f"{self.protocol}://{self.host}:{self.port}"
        self.client_uri = f"{self.url}/client"
        self.dispatch_uri = f"{self.url}/dispatch"

        self._socket = bind_socket(self.host, self.port)

        if generate_cert:
            cert, key, pw = _generate_certificate(ip_address=self.host)
        else:
            cert, key, pw = None, None, None  # type: ignore
        self.cert = cert
        self._key = key
        self._key_pw = pw

        self.token = _generate_authentication() if use_token else None

    def get_socket(self):
        # NOTE: socket objects do not seem to provide a reliable method to check
        # if they are not bound. There is a ._closed attribute, but that is
        # private. Here we check if the underlying file-descriptor is valid,
        # which should work on both Unix and Windows.
        if self._socket.fileno() < 0:
            self._socket = bind_socket(self.host, self.port)
        return self._socket

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
                with open(cert_path, "w") as f1:
                    f1.write(self.cert)

                key_path = tmp_path / "ee.key"
                with open(key_path, "wb") as f2:
                    f2.write(self._key)
                context = ssl.SSLContext(protocol=protocol)
                context.load_cert_chain(cert_path, key_path, self._key_pw)
                return context
        finally:
            tempfile.tempdir = backup_default_tmp
