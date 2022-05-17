from typing import Awaitable, Dict

import ert.data

TransmitterCoroutine = Awaitable[Dict[int, Dict[str, ert.data.RecordTransmitter]]]
