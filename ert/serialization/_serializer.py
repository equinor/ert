import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import aiofiles
import yaml
from ecl.summary import EclSum


class Serializer(ABC):
    @abstractmethod
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("not implemented")

    @abstractmethod
    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def encode_to_path(
        self, obj: Any, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def decode_from_path(
        self, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError("not implemented")


class _json_serializer(Serializer):
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        return json.dumps(obj, *args, **kwargs)

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        return json.loads(series, *args, **kwargs)

    async def encode_to_path(
        self, obj: Any, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> None:
        async with aiofiles.open(path, mode="wt", encoding="utf-8") as filehandle:
            await filehandle.write(json.dumps(obj))

    async def decode_from_path(
        self, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> Any:
        async with aiofiles.open(path, mode="rt", encoding="utf-8") as filehandle:
            contents = await filehandle.read()
        return self.decode(contents, *args, **kwargs)


class _yaml_serializer(Serializer):
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        res: str = yaml.dump(obj, *args, **kwargs)
        return res

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        return yaml.safe_load(series)

    async def encode_to_path(
        self, obj: Any, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> None:
        async with aiofiles.open(path, mode="wt", encoding="utf-8") as filehandle:
            await filehandle.write(yaml.dump(obj))

    async def decode_from_path(
        self, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> Any:
        async with aiofiles.open(path, mode="rt", encoding="utf-8") as filehandle:
            contents = await filehandle.read()
            return self.decode(contents, *args, **kwargs)


class _ecl_sum_serializer(Serializer):
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("not implemented")

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")

    async def encode_to_path(
        self, obj: Any, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError("not implemented")

    async def decode_from_path(
        self, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> Any:
        key = kwargs.get("key", None)
        if key is None:
            raise ValueError("key must be provided as a keyword argument")

        eclsum = EclSum(str(path))
        return dict(zip(map(str, eclsum.dates), eclsum.numpy_vector(key)))
