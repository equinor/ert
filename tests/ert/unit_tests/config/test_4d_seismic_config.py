from pathlib import Path

import numpy as np
import pytest
from surfio import IrapHeader, IrapSurface

from ert.config import ErtConfig
from ert.config.parsing import ConfigDict


def test_that_4d_seismic_attributes_can_be_parsed(use_tmpdir):
    ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file
        """,
    )


@pytest.fixture
def surface():
    return IrapSurface(
        IrapHeader(
            ncol=3,
            nrow=2,
            xori=0.0,
            yori=0.0,
            xinc=2.0,
            yinc=2.0,
            xmax=2.0,
            ymax=2.0,
            rot=0.0,
            xrot=0.0,
            yrot=0.0,
        ),
        values=np.zeros((3, 2)),
    )


def test_that_4d_seismic_attribute_is_added_to_ert_config_dict(
    use_tmpdir, surface, monkeypatch
):
    def from_dict_assertion_mock(config_dict: ConfigDict):
        assert "SEISMIC_4D_ATTRIBUTE" in config_dict

    monkeypatch.setattr(ErtConfig, "from_dict", from_dict_assertion_mock)

    surface.to_binary_file(Path("file"))
    ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file
        """,
    )
