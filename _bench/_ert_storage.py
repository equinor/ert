import io
from ._base import NumpyBaseStorage, Namespace
import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence
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


class ErtStorage(NumpyBaseStorage):
    def __init__(self, args: Namespace, keep: bool) -> None:
        super().__init__(args, keep)
        self._context = Storage.start_server()
        self._storage = self._context.__enter__()
        self._storage.wait_until_ready()

        self.ensemble_id = (
            self._load_ensemble() if keep else self._create_ensemble(args)
        )
        self.session = self._storage.session()
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
        stream = io.BytesIO()
        write_array(stream, array)
        self.session.post(
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

    def load_parameter(self, name: str) -> npt.NDArray[np.float64]:
        return self._load(name, None)

    def load_response(
        self, name: str, iens: Optional[Sequence[int]]
    ) -> npt.NDArray[np.float64]:
        return self._load(name, iens)

    async def load_response_async(self, name: str, iens: Optional[Sequence[int]]) -> npt.NDArray[np.float64]:
        return await self._load_async(name, iens)

    def _load(self, name: str, iens: Optional[Sequence[int]]) -> npt.NDArray[np.float64]:
        if iens is None:
            with self._storage.session() as session:
                content = session.get(
                    f"/ensembles/{self.ensemble_id}/records/{name}",
                    headers={"accept": "application/x-numpy"},
                ).content
                stream = io.BytesIO(content)
                return read_array(stream)
        else:
            with self._storage.session() as session:
                data = []
                for i in iens:
                    content = session.get(
                        f"/ensembles/{self.ensemble_id}/records/{name}",
                        params={"realization_index": i},
                        headers={"accept": "application/x-numpy"},
                    ).content
                    stream = io.BytesIO(content)
                    data.append(read_array(stream))
                return np.array(data)

    async def _load_async(self, name: str, iens: Optional[Sequence[int]]) -> npt.NDArray[np.float64]:
        if iens is None:
            content = await self.async_session.get(
                f"/ensembles/{self.ensemble_id}/records/{name}",
                headers={"accept": "application/x-numpy"},
            )
            stream = io.BytesIO(content.content)
            return read_array(stream)
        else:
            data = []
            for i in iens:
                content = await self.async_session.get(
                    f"/ensembles/{self.ensemble_id}/records/{name}",
                    params={"realization_index": i},
                    headers={"accept": "application/x-numpy"},
                )
                stream = io.BytesIO(content.content)
                data.append(read_array(stream))
            return np.array(data)
