from typing import Awaitable, Dict

import ert

TransmitterCoroutine = Awaitable[Dict[int, Dict[str, ert.data.RecordTransmitter]]]
