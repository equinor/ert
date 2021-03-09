from datetime import datetime
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_gui.model.job_list import JobListProxyModel
from ert_gui.model.node import NodeType
from ert_gui.model.snapshot import COLUMNS, SnapshotModel
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Job
from ert_shared.status.entity.state import JOB_STATE_FAILURE, JOB_STATE_START
from PyQt5.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api
from tests.gui.conftest import partial_snapshot


def _id_to_col(identifier):
    for col, fields in enumerate(COLUMNS[NodeType.STEP]):
        if fields[1] == identifier:
            return col
    raise ValueError(f"{identifier} not a column in {COLUMNS}")


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    partial = partial_snapshot(full_snapshot)
    source_model = SnapshotModel()

    model = JobListProxyModel(None, 0, 0, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)
    source_model._add_snapshot(full_snapshot, 1)

    source_model._add_partial_snapshot(partial, 0)
    source_model._add_partial_snapshot(partial, 1)

    qtmodeltester.check(model, force_py=True)


def test_changes(full_snapshot):
    source_model = SnapshotModel()

    model = JobListProxyModel(None, 0, 0, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data() == JOB_STATE_START
    )

    partial = PartialSnapshot(full_snapshot)
    start_time = datetime(year=2020, month=10, day=27)
    end_time = datetime(year=2020, month=10, day=28)
    partial.update_job(
        "0",
        "0",
        "0",
        "0",
        job=Job(
            status=JOB_STATE_FAILURE,
            start_time=start_time,
            end_time=end_time,
        ),
    )
    source_model._add_partial_snapshot(partial, 0)
    assert model.index(0, _id_to_col(ids.START_TIME), QModelIndex()).data() == str(
        start_time
    )
    assert model.index(0, _id_to_col(ids.END_TIME), QModelIndex()).data() == str(
        end_time
    )
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == JOB_STATE_FAILURE
    )


def test_no_cross_talk(full_snapshot):
    source_model = SnapshotModel()

    model = JobListProxyModel(None, 0, 0, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)
    source_model._add_snapshot(full_snapshot, 1)

    # Test that changes to iter=1 does not bleed into iter=0
    partial = PartialSnapshot(full_snapshot)
    partial.update_job("0", "0", "0", "0", job=Job(status=JOB_STATE_FAILURE))
    source_model._add_partial_snapshot(partial, 1)
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data() == JOB_STATE_START
    )

    model.set_step(1, 0, 0, 0)
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == JOB_STATE_FAILURE
    )
