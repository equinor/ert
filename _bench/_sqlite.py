import os
import argparse
import numpy as np
import numpy.typing as npt
from ._base import BaseStorage
from sqlalchemy.exc import IntegrityError
import sqlalchemy.orm
from contextlib import contextmanager
from typing import Generator, Optional

os.environ["ERT_STORAGE_DATABASE_URL"] = "sqlite:///_tmp_Sqlite.db"

import ert_storage.database_schema as ds
import ert_storage.database


class Sqlite(BaseStorage[npt.NDArray[np.float64]]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        ert_storage.database.Base.metadata.create_all(bind=ert_storage.database.engine)

        with self._session() as db:
            ensemble = ds.Ensemble(
                active_realizations=range(args.ensemble_size),
                parameter_names=[],
                response_names=[],
                experiment=ds.Experiment(name="benchmark"),
                size=args.ensemble_size,
            )

            db.add(ensemble)
            db.commit()
            db.refresh(ensemble)
            self._ensemble_id = ensemble.id

    def save_parameter(self, name: str, content: npt.NDArray[np.float64]) -> None:
        with self._session() as db:
            record = self._new_record_matrix(db, name, None, None)
            if (
                record.realization_index is None
                and record.record_class is ds.RecordClass.parameter
            ):
                if content.ndim <= 1:
                    raise RuntimeError(
                        f"Ensemble-wide parameter record '{record.name}' for ensemble '{record.record_info.ensemble.id}'"
                        "must have dimensionality of at least 2"
                    )

            matrix_obj = ds.F64Matrix(content=content.tolist())
            record.f64_matrix = matrix_obj
            self._create_record(db, record)

    def save_response(
        self, name: str, content: npt.NDArray[np.float64], iens: int
    ) -> None:
        with self._session() as db:
            record = self._new_record_matrix(db, name, iens, None)
            if (
                record.realization_index is None
                and record.record_class is ds.RecordClass.parameter
            ):
                if content.ndim <= 1:
                    raise RuntimeError(
                        f"Ensemble-wide parameter record '{record.name}' for ensemble '{record.record_info.ensemble.id}'"
                        "must have dimensionality of at least 2"
                    )

            matrix_obj = ds.F64Matrix(content=content.tolist())
            record.f64_matrix = matrix_obj
            self._create_record(db, record)


    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array

    def _new_record(
        self, db: sqlalchemy.orm.Session, name: str, realization_index: Optional[int]
    ) -> ds.Record:
        ensemble = db.query(ds.Ensemble).filter_by(id=self._ensemble_id).one()

        q = (
            db.query(ds.Record)
            .join(ds.RecordInfo)
            .filter_by(ensemble_pk=ensemble.pk, name=name)
        )
        if (
            ensemble.size != -1
            and realization_index is not None
            and realization_index not in ensemble.active_realizations
        ):
            raise RuntimeError(
                f"Realization index {realization_index} outside "
                f"of allowed realization indices {ensemble.active_realizations}"
            )
        q = q.filter(
            (ds.Record.realization_index == None)
            | (ds.Record.realization_index == realization_index)
        )

        if q.count() > 0:
            raise RuntimeError(
                f"Ensemble-wide record '{name}' for ensemble '{self._ensemble_id}' already exists"
            )

        return ds.Record(
            record_info=ds.RecordInfo(ensemble=ensemble, name=name),
            realization_index=realization_index,
        )

    def _new_record_matrix(
        self,
        db: sqlalchemy.orm.Session,
        name: str,
        realization_index: Optional[int],
        prior: Optional[str],
    ) -> ds.Record:
        record = self._new_record(db, name, realization_index)
        ensemble = record.record_info.ensemble
        if record.name in ensemble.parameter_names:
            record_class = ds.RecordClass.parameter
        elif record.name in ensemble.response_names:
            record_class = ds.RecordClass.response
        else:
            record_class = ds.RecordClass.other

        if prior is not None:
            if record_class is not ds.RecordClass.parameter:
                raise RuntimeError("Priors can only be specified for parameter records")
            record.record_info.prior = db.query(ds.Prior).filter_by(name=prior).one()

        record.record_info.record_class = record_class
        record.record_info.record_type = ds.RecordType.f64_matrix
        return record

    def _create_record(self, db: sqlalchemy.orm.Session, record: ds.Record) -> None:
        nested = db.begin_nested()
        try:
            db.add(record)
            db.commit()
        except IntegrityError:
            # Assuming this is a UNIQUE constraint failure due to an existing
            # record_info with the same name and ensemble. Try to fetch the
            # record_info
            nested.rollback()
            record_info = record.record_info
            old_record_info = (
                db.query(ds.RecordInfo)
                .filter_by(ensemble=record_info.ensemble, name=record_info.name)
                .one()
            )

            # Check that the parameters match
            if record_info.record_class != old_record_info.record_class:
                raise RuntimeError(
                    "Record class of new record does not match previous record class"
                )
            if record_info.record_type != old_record_info.record_type:
                raise RuntimeError(
                    "Record type of new record does not match previous record type"
                )

            record = ds.Record(
                record_info=old_record_info,
                f64_matrix=record.f64_matrix,
                file=record.file,
                realization_index=record.realization_index,
            )
            db.add(record)
            db.commit()

    @contextmanager
    def _session(self) -> Generator[sqlalchemy.orm.Session, None, None]:
        with ert_storage.database.Session() as db:
            try:
                yield db
                db.commit()
                db.close()
            except:
                db.rollback()
                db.close()
                raise
