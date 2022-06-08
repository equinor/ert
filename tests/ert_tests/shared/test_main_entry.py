import os
import sys
import logging
from unittest.mock import MagicMock, patch, mock_open

import pytest

from ert_shared import main
from ert_shared.main import log_config

from utils import SOURCE_DIR


def test_main_logging(monkeypatch, caplog):
    parser_mock = MagicMock()
    parser_mock.func.side_effect = ValueError("This is a test")
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=parser_mock))
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with pytest.raises(
        SystemExit, match='ERT crashed unexpectedly with "This is a test"'
    ):
        main.main()
    assert 'ERT crashed unexpectedly with "This is a test"' in caplog.text
    assert "Traceback" in caplog.text


def test_main_logging_argparse(monkeypatch, caplog):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "valid_file", MagicMock(return_value=True))
    monkeypatch.setattr(main, "run_cli", MagicMock())
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with caplog.at_level(logging.INFO):
        main.main()
    assert "mode='test_run'" in caplog.text


def test_main_logging_config(monkeypatch, caplog, tmp_path):
    content = "Content of config.ert\nMore content."
    ert_config = tmp_path / "config.ert"
    with open(ert_config, "w", encoding="utf-8") as file_obj:
        file_obj.write(content)
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "run_cli", MagicMock())
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", str(ert_config)])
    with caplog.at_level(logging.INFO):
        main.main()
    assert f"Content of the configuration file ({str(ert_config)}):" in caplog.text
    assert content in caplog.text


def test_api_database_default(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())

    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    mocked_start_server = MagicMock()
    monkeypatch.setattr(
        "ert_shared.services.storage_service.BaseService.start_server",
        mocked_start_server,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert", "api"],
    )

    main.main()
    # We expect default value from Storage class, validate that no explicit
    # value is given for database_url
    mocked_start_server.assert_called_once_with(res_config=None, verbose=True)


def test_api_database_url_forwarded(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())

    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    mocked_start_server = MagicMock()
    monkeypatch.setattr(
        "ert_shared.services.storage_service.BaseService.start_server",
        mocked_start_server,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert", "api", "--database-url", "TEST_DATABASE_URL"],
    )

    main.main()
    mocked_start_server.assert_called_once_with(
        res_config=None, database_url="TEST_DATABASE_URL", verbose=True
    )


def test_vis_database_url_forwarded(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())

    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    mocked_connect_or_start_server = MagicMock()
    monkeypatch.setattr(
        "ert_shared.services.storage_service.BaseService.connect_or_start_server",
        mocked_connect_or_start_server,
    )
    monkeypatch.setattr(
        "ert_shared.services.storage_service.BaseService.start_server",
        MagicMock,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert", "vis", "--database-url", "TEST_DATABASE_URL"],
    )

    main.main()
    mocked_connect_or_start_server.assert_called_once_with(
        res_config=None, database_url="TEST_DATABASE_URL", verbose=False
    )


@pytest.mark.parametrize(
    "config_content, expected",
    [
        pytest.param("--Comment", "", id="Line comment"),
        pytest.param(" --Comment", "", id="Line comment with whitespace"),
        pytest.param("\t--Comment", "", id="Line comment with whitespace"),
        pytest.param("KEY VALUE", "KEY VALUE\n", id="Config line"),
        pytest.param("KEY VALUE --Comment", "KEY VALUE\n", id="Inline comment"),
        pytest.param(
            'FORWARD_MODEL("--argument_or_comment")',
            'FORWARD_MODEL("--argument_or_comment")\n',
            id="Not able to determine inline comment or part of config",
        ),
    ],
)
def test_logging_config(caplog, config_content, expected):
    base_content = "Content of the configuration file (file_name):\n{}"
    logger = logging.getLogger(__name__)
    config_path = "file_name"
    with patch("builtins.open", mock_open(read_data=config_content)), patch(
        "os.path.isfile", MagicMock(return_value=True)
    ):
        with caplog.at_level(logging.INFO):
            log_config(config_path, logger)
    expected = base_content.format(expected)
    assert expected in caplog.messages


def test_logging_snake_oil_config(caplog):
    """
    Run logging on an actual config file with line comments
    and inline comments to check the result
    """
    logger = logging.getLogger(__name__)
    config_path = os.path.join(
        SOURCE_DIR,
        "test-data",
        "local",
        "snake_oil_structure",
        "ert",
        "model",
        "user_config.ert",
    )
    with caplog.at_level(logging.INFO):
        log_config(config_path, logger)
    assert (
        """
JOBNAME SNAKE_OIL_STRUCTURE_%d
DEFINE  <USER>          TEST_USER
DEFINE  <SCRATCH>       scratch/ert
DEFINE  <CASE_DIR>      the_extensive_case
DEFINE  <ECLIPSE_NAME>  XYZ
DATA_FILE           ../../eclipse/model/SNAKE_OIL.DATA
GRID                ../../eclipse/include/grid/CASE.EGRID
RUNPATH             <SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d
ECLBASE             eclipse/model/<ECLIPSE_NAME>-%d
ENSPATH             ../output/storage/<CASE_DIR>
RUNPATH_FILE        ../output/run_path_file/.ert-runpath-list_<CASE_DIR>
REFCASE             ../input/refcase/SNAKE_OIL_FIELD
UPDATE_LOG_PATH     ../output/update_log/<CASE_DIR>
RANDOM_SEED 3593114179000630026631423308983283277868
NUM_REALIZATIONS              10
MAX_RUNTIME                   23400
MIN_REALIZATIONS              50%
QUEUE_SYSTEM                  LSF
QUEUE_OPTION LSF MAX_RUNNING  100
QUEUE_OPTION LSF LSF_RESOURCE select[x86_64Linux] same[type:model]
QUEUE_OPTION LSF LSF_SERVER   simulacrum
QUEUE_OPTION LSF LSF_QUEUE    mr
MAX_SUBMIT                    13
UMASK                         007
GEN_DATA super_data INPUT_FORMAT:ASCII RESULT_FILE:super_data_%d  REPORT_STEPS:1
GEN_KW SIGMA          ../input/templates/sigma.tmpl          coarse.sigma              ../input/distributions/sigma.dist
RUN_TEMPLATE             ../input/templates/seed_template.txt     seed.txt
INSTALL_JOB SNAKE_OIL_SIMULATOR ../../snake_oil/jobs/SNAKE_OIL_SIMULATOR
INSTALL_JOB SNAKE_OIL_NPV ../../snake_oil/jobs/SNAKE_OIL_NPV
INSTALL_JOB SNAKE_OIL_DIFF ../../snake_oil/jobs/SNAKE_OIL_DIFF
FORWARD_MODEL SNAKE_OIL_SIMULATOR
FORWARD_MODEL SNAKE_OIL_NPV
FORWARD_MODEL SNAKE_OIL_DIFF
HISTORY_SOURCE REFCASE_HISTORY
OBS_CONFIG ../input/observations/obsfiles/observations.txt
TIME_MAP   ../input/refcase/time_map.txt
SUMMARY WOPR:PROD
SUMMARY WOPT:PROD
SUMMARY WWPR:PROD
SUMMARY WWCT:PROD
SUMMARY WWPT:PROD
SUMMARY WBHP:PROD
SUMMARY WWIR:INJ
SUMMARY WWIT:INJ
SUMMARY WBHP:INJ
SUMMARY ROE:1
LOAD_WORKFLOW_JOB ../bin/workflows/workflowjobs/UBER_PRINT
LOAD_WORKFLOW     ../bin/workflows/MAGIC_PRINT"""  # noqa: E501
        in caplog.text
    )
