from res._lib.active_list import ActiveList, ActiveMode


def __repr__(self):
    return f"ActiveList(mode = {self.getMode()}, active_size = {self.getActiveSize(0)})"

ActiveList.__repr__ = __repr__
del __repr__

__all__ = ["ActiveList", "ActiveMode"]
