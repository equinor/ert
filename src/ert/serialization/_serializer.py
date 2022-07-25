import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import aiofiles
import yaml


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
    def encode(
        self, obj: Any, *args: Any, indent: Optional[int] = 4, **kwargs: Any
    ) -> str:
        # pylint: disable=arguments-differ
        return json.dumps(obj, *args, indent=indent, **kwargs)

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        return json.loads(series, *args, **kwargs)

    async def encode_to_path(
        self, obj: Any, path: Union[str, Path], *args: Any, **kwargs: Any
    ) -> None:
        async with aiofiles.open(path, mode="wt", encoding="utf-8") as filehandle:
            await filehandle.write(json.dumps(obj, indent=4))

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
