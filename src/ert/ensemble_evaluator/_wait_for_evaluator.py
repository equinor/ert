import logging
import ssl

logger = logging.getLogger(__name__)

WAIT_FOR_EVALUATOR_TIMEOUT = 60


def get_ssl_context(cert: str | bytes | None) -> ssl.SSLContext | bool:
    if cert is None:
        return False
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.load_verify_locations(cadata=cert)
    return ssl_context
