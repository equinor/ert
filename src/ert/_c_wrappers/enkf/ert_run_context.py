from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List

from ert._c_wrappers.enkf.run_arg import RunArg
from ert.shared.runpaths import Runpaths

if TYPE_CHECKING:
    from ert.storage import EnsembleAccessor


@dataclass
class RunContext:
    sim_fs: EnsembleAccessor
    runpaths: Runpaths
    initial_mask: List[bool] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self):
        self.run_id = uuid.uuid4()
        self.run_args = []
        paths = self.runpaths.get_paths(
            list(range(len(self.initial_mask))), self.iteration
        )
        job_names = self.runpaths.get_jobnames(
            list(range(len(self.initial_mask))), self.iteration
        )

        for iens, (run_path, job_name, active) in enumerate(
            zip(paths, job_names, self.initial_mask)
        ):
            self.run_args.append(
                RunArg(
                    str(self.run_id),
                    self.sim_fs,
                    iens,
                    self.iteration,
                    run_path,
                    job_name,
                    active,
                )
            )

    @property
    def mask(self):
        return [real.active for real in self]

    def is_active(self, index: int) -> bool:
        return self[index].active

    @property
    def active_realizations(self):
        return [i for i, real in enumerate(self) if real.active]

    def __len__(self):
        return len(self.initial_mask)

    def __getitem__(self, item) -> "RunArg":
        return self.run_args[item]

    def __iter__(self) -> Iterator["RunArg"]:
        yield from self.run_args

    def deactivate_realization(self, realization_nr: int) -> str:
        self[realization_nr].active = False
