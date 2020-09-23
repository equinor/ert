class GeneralEvent(object):
    def __init__(
        self,
        phase_name,
        current_phase,
        total_phases,
        progress,
        indeterminate,
        sim_states,
        runtime,
    ):
        self.phase_name = phase_name
        self.current_phase = current_phase
        self.total_phases = total_phases
        self.progress = progress
        self.indeterminate = indeterminate
        self.sim_states = sim_states
        self.runtime = runtime


class DetailedEvent(object):
    def __init__(self, details, iteration):
        self.details = details
        self.iteration = iteration


class EndEvent(object):
    def __init__(self, failed, failed_msg=None):
        self.failed = failed
        self.failed_msg = failed_msg
