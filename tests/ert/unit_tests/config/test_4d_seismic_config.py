from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from surfio import IrapHeader, IrapSurface

from ert.config import ErtConfig
from ert.config.parsing import ConfigDict
from ert.config.seismic_attribute_config import SeismicAttributeConfig


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


def test_that_4d_seismic_attribute_response_is_added_to_ensemble_response_config(
    use_tmpdir, surface
):
    surface.to_binary_file(Path("file"))
    config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file
        """,
    )
    assert "seismic_attribute" in config.ensemble_config.response_configs


def test_that_multiple_seismic_attribute_response_is_added_to_ensemble_response_config(
    use_tmpdir, surface
):
    surface.to_binary_file(Path("file1"))
    surface.to_binary_file(Path("file2"))
    config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file1
        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file2
        """,
    )
    seismic_attribute_config: SeismicAttributeConfig = (
        config.ensemble_config.response_configs.get("seismic_attribute")
    )
    assert seismic_attribute_config is not None
    assert len(seismic_attribute_config.seismic_attributes) == 2


def test_that_keyword_arguments_are_casted_without_prefixes_to_seismic_attributes(
    use_tmpdir, surface
):
    """The parameters START: END: and FILE: are included in the string values of the
    keyword arguments provided to the constructor of SeismicAttributeConfig. We want to
    ensure these are removed.

    START: and END: are prefixes
    FILE: appears in from of the filename, but after the rest of the absolute path."""
    file = "file"
    start = "2025-01-01"
    end = "2025-12-31"
    surface.to_binary_file(Path("file"))
    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        f"SEISMIC_4D_ATTRIBUTE START:{start} END:{end} FILE:{file}\n"
    )
    seismic_attribute_config: SeismicAttributeConfig = (
        config.ensemble_config.response_configs.get("seismic_attribute")
    )
    seismic_attribute = seismic_attribute_config.seismic_attributes.pop()
    assert seismic_attribute.start == datetime.fromisoformat(start)
    assert seismic_attribute.end == datetime.fromisoformat(end)
    assert seismic_attribute.file == Path(file).resolve()
