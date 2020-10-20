from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.tool import recursive_update
from ert_shared.ensemble_evaluator.entity.forward_model_job import _ForwardModelJob


class _EnsembleResponseEvent:
    def __init__(
        self, event_index, status=None, forward_models=None, realizations=None
    ):
        self._event_index = event_index
        self._status = status
        self._forward_models = forward_models
        self._realizations = realizations

    def __repr__(self):
        return str(self.to_dict())

    def is_terminated(self):
        return self._status == ids.TERMINATED

    def is_done(self):
        return self._status == ids.DONE

    def is_running(self):
        return not (self.is_terminated() or self.is_done())

    def is_partial(self):
        """The forward_models referred to here are the models as they are
        intended to run, not how they ran or are currently running. If this
        field is missing, it can be safe to assume it is an event describing
        how the models ran or are running."""
        return self._forward_models is None

    def merge_event(self, other):
        if self.is_partial() or not other.is_partial():
            raise ValueError("Can only merge partial event into non-partial event")

        if self._event_index >= other._event_index:
            return

        self._status = self._status if other._status is None else other._status
        self._event_index = other._event_index

        recursive_update(self._realizations, other._realizations)

    def to_dict(self):
        return {
            ids.STATUS: self._status,
            ids.FORWARD_MODELS: self._forward_models.to_dict()
            if self._forward_models is not None
            else None,
            ids.REALIZATIONS: self._realizations,
            ids.EVENT_INDEX: self._event_index,
        }

    @classmethod
    def from_dict(cls, data):
        event_index = data[ids.EVENT_INDEX]
        forward_models = data.get(ids.FORWARD_MODELS)
        if forward_models is not None:
            forward_models = [
                _ForwardModelJob.from_dict(fmj_dict) for fmj_dict in forward_models
            ]
        realizations = data.get(ids.REALIZATIONS)
        status = data.get(ids.STATUS)
        return cls(
            event_index,
            forward_models=forward_models,
            realizations=realizations,
            status=status,
        )


def create_evaluator_event(event_index, realizations, status):
    return _EnsembleResponseEvent(
        event_index,
        forward_models=None,
        realizations=realizations,
        status=status,
    )


def create_unindexed_evaluator_event(realizations, status):
    return create_evaluator_event(
        event_index=-1, realizations=realizations, status=status
    )


def create_evaluator_event_from_dict(data):
    return _EnsembleResponseEvent.from_dict(data)


def create_evaluator_snapshot_from_ensemble(ensemble):
    eev = _EnsembleResponseEvent(0, )


def create_evaluator_snapshot(forward_models, realizations, status=ids.CREATED):
    # realizations = {}
    # for forward_model in forward_models:
    #     realization_fmjs[forward_model.id] = {"status": "unknown", "data": None}
    # realizations[realization_index] = {
    #     "status": "unknown",
    #     "forward_models": realization_fmjs,
    # }
    # for realization_index in realization_indexes:
    #     realization_fmjs = {}
    return _EnsembleResponseEvent(
        event_index=0,
        status=status,
        forward_models=forward_models,
        realizations=realizations,
    )
