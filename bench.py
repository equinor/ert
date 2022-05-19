#!/usr/bin/env python3
from res.enkf.enkf_fs import EnkfFs
from res._lib.enkf_fs import write_param_vector_raw, read_param_vector_raw
import numpy as np


def bench_enkffs():
    fs = EnkfFs.createFileSystem("_tmp_enkf_fs", mount=True)
    assert fs is not None
    expect_mat = np.random.rand(5, 7)
    write_param_vector_raw(fs, expect_mat, "FOPR", 0)
    print(f"{expect_mat=}")

    actual_mat = read_param_vector_raw(fs, "FOPR", 0)
    print(f"{actual_mat=}")

    assert (expect_mat == actual_mat).all()


if __name__ == "__main__":
    bench_enkffs()
