import socket

from ert.services._storage_main import _get_host_list
from ert.shared import net_utils


def test_get_host_list():
    hosts = _get_host_list()
    assert len(hosts) > 0
    print(hosts)


def test_socket_gethostname():
    print(socket.gethostname())


def test_socket_getfqdn():
    print(socket.getfqdn())


def test_get_machine_name():
    print(net_utils.get_machine_name())