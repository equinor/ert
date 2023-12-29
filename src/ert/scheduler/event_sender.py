from __future__ import annotations

import asyncio
import ssl
from typing import TYPE_CHECKING, Any, Mapping, Optional

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from websockets import Headers, connect

if TYPE_CHECKING:
    from ert.ensemble_evaluator.identifiers import EvGroupRealizationType


class EventSender:
    def __init__(
        self,
        ens_id: Optional[str],
        ee_uri: Optional[str],
        ee_cert: Optional[str],
        ee_token: Optional[str],
    ) -> None:
        self.ens_id = ens_id
        self.ee_uri = ee_uri
        self.ee_cert = ee_cert
        self.ee_token = ee_token
        self.events: asyncio.Queue[CloudEvent] = asyncio.Queue()

    async def send(
        self,
        type: EvGroupRealizationType,
        source: str,
        attributes: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = CloudEvent(
            {
                "type": type,
                "source": f"/ert/ensemble/{self.ens_id}/{source}",
                **(attributes or {}),
            },
            data,
        )
        await self.events.put(event)

    async def publisher(self) -> None:
        if not self.ee_uri:
            return
        tls: Optional[ssl.SSLContext] = None
        if self.ee_cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self.ee_cert)
        headers = Headers()
        if self.ee_token:
            headers["token"] = self.ee_token

        async for conn in connect(
            self.ee_uri,
            ssl=tls,
            extra_headers=headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            while True:
                event = await self.events.get()
                await conn.send(to_json(event))
