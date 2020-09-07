import os
import sys

from ert_shared.storage.autoclient import AutoClient

from unittest.mock import Mock, patch


@patch.dict(os.environ, {})
def test_bind_parsing():
    with patch("ert_shared.storage.autoclient.socket") as mock_socket, patch(
        "ert_shared.storage.autoclient.subprocess"
    ):
        mock_sock = Mock()
        mock_socket.socket.return_value = mock_sock
        mock_sock.getsockname.return_value = ("", 0)

        AutoClient("0.0.0.0:0")

        mock_sock.bind.assert_called_with(("0.0.0.0", 0))
