import io
import argparse
from ._base import BaseStorage
import numpy as np
import numpy.typing as npt
from numpy.lib.format import write_array, read_array
from ert_shared.services import Storage


_CREATE_ENSEMBLE = """
mutation($size: Int!, $params: [String]) {
  createExperiment(name: "benchmark") {
    createEnsemble(size: $size, parameterNames: $params) {
      id
    }
  }
}
"""


class ErtStorage(BaseStorage[npt.NDArray[np.float64]]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._context = Storage.start_server()
        self._storage = self._context.__enter__()
        self._storage.wait_until_ready()

        with self._storage.session() as session:
            ensemble = session.post(
                "/gql",
                json={
                    "query": _CREATE_ENSEMBLE,
                    "variables": {
                        "size": args.ensemble_size,
                        "params": [],
                    },
                },
            )
        self.ensemble_id = ensemble.json()["data"]["createExperiment"]["createEnsemble"]["id"]

    def __del__(self) -> None:
        self._context.__exit__(None, None, None)

    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        with self._storage.session() as session:
            stream = io.BytesIO()
            write_array(stream, array)
            session.post(
                f"/ensembles/{self.ensemble_id}/records/{name}/matrix",
                content=stream.getvalue(),
                headers={"content-type": "application/x-numpy"},
            )

    async def save_parameter_async(self, name: str, array: npt.NDArray[np.float64]) -> None:
        async with await self._storage.async_session() as session:
            stream = io.BytesIO()
            write_array(stream, array)
            await session.post(
                f"/ensembles/{self.ensemble_id}/records/{name}/matrix",
                content=stream.getvalue(),
                headers={"content-type": "application/x-numpy"},
            )

    def save_response(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        with self._storage.session() as session:
            stream = io.BytesIO()
            write_array(stream, array)
            session.post(
                f"/ensembles/{self.ensemble_id}/records/{name}/matrix",
                content=stream.getvalue(),
                params={"realization_index": iens},
                headers={"content-type": "application/x-numpy"},
            )

    async def save_response_async(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        async with await self._storage.async_session() as session:
            stream = io.BytesIO()
            write_array(stream, array)
            await session.post(
                f"/ensembles/{self.ensemble_id}/records/{name}/matrix",
                content=stream.getvalue(),
                params={"realization_index": iens},
                headers={"content-type": "application/x-numpy"},
            )

    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array
