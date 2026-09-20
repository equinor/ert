import pytest

from ert.storage.mode import BaseMode, Mode, ModeError, require_write


class SomeClass(BaseMode):
    def write_or_default(self) -> str:
        return "good" if self.can_write else "fail"

    @require_write
    def raises_unless_write(self) -> bool:
        return True

    @require_write
    def concat_if_write(self, a: str, b: str) -> str:
        return a + b


def test_read_mode():
    obj = SomeClass(Mode.READ)
    assert obj.write_or_default() == "fail"
    with pytest.raises(
        ModeError,
        match="This operation requires write access, but we only have read access",
    ):
        obj.raises_unless_write()
    with pytest.raises(
        ModeError,
        match="This operation requires write access, but we only have read access",
    ):
        obj.concat_if_write("foo", "bar")


def test_write_mode():
    obj = SomeClass(Mode.WRITE)
    assert obj.write_or_default() == "good"
    assert obj.raises_unless_write()
    assert obj.concat_if_write("foo", "bar") == "foobar"
