from __future__ import annotations

from typing import Optional

from resdata.rd_util import get_num_cpu


def get_num_cpu_from_data_file(data_file: str) -> Optional[int]:
    return get_num_cpu(data_file)
