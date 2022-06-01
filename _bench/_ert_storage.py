import io
from ._base import BaseStorage, Namespace
import numpy as np
import numpy.typing as npt
from typing import Optional, List
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
    def __init__(self, args: Namespace, keep: bool) -> None:
        super().__init__(args, keep)
        self._context = Storage.start_server()
        self._storage = self._context.__enter__()
        self._storage.wait_until_ready()

        self.ensemble_id = (
            self._load_ensemble() if keep else self._create_ensemble(args)
        )
        self.async_session = self._storage.async_session()

    def _create_ensemble(self, args: Namespace) -> str:
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
        ensemble_id = ensemble.json()["data"]["createExperiment"]["createEnsemble"][
            "id"
        ]
        (self.path / "ensemble_id").write_text(ensemble_id)
        return ensemble_id

    def _load_ensemble(self) -> str:
        return (self.path / "ensemble_id").read_text()

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

    async def save_parameter_async(
        self, name: str, array: npt.NDArray[np.float64]
    ) -> None:
        stream = io.BytesIO()
        write_array(stream, array)
        await self.async_session.post(
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
        stream = io.BytesIO()
        write_array(stream, array)
        await self.async_session.post(
            f"/ensembles/{self.ensemble_id}/records/{name}/matrix",
            content=stream.getvalue(),
            params={"realization_index": iens},
            headers={"content-type": "application/x-numpy"},
        )

    def load_response(
        self, name: str, iens: Optional[List[int]]
    ) -> npt.NDArray[np.float64]:
        if iens is None:
            with self._storage.session() as session:
                content = session.get(
                    f"/ensembles/{self.ensemble_id}/records/{name}",
                    headers={"content-type": "application/x-numpy"},
                ).content
                print(content)
                stream = io.BytesIO(content)
                return read_array(stream)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array
