from ert.config import ErtConfig


def test_that_4d_seismic_attributes_can_be_parsed(use_tmpdir):
    ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC_4D_ATTRIBUTE START:2020-01-01 END:2020-02-01 FILE:file
        """,
    )
