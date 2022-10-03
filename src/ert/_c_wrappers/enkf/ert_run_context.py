import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List

from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.run_arg import RunArg


@dataclass
class RunContext:
    sim_fs: EnkfFs
    target_fs: EnkfFs = None
    mask: List[bool] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    jobnames: List[str] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self):
        self.run_id = f"{uuid.uuid4()}:{datetime.now().strftime('%Y-%m-%dT%H%M')}"
        self.run_args = []
        if self.jobnames and self.paths:
            for iens, (job_name, path) in enumerate(zip(self.jobnames, self.paths)):
                self.run_args.append(
                    RunArg(
                        str(self.run_id),
                        self.sim_fs,
                        iens,
                        self.iteration,
                        path,
                        job_name,
                    )
                )
        if not self.target_fs:
            self.target_fs = self.sim_fs

    def is_active(self, index: int) -> bool:
        try:
            return self.mask[index]
        except IndexError:
            return False

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, item) -> RunArg:
        return self.run_args[item]

    def __iter__(self) -> Iterator[RunArg]:
        yield from self.run_args

    def deactivate_realization(self, realization_nr):
        self.mask[realization_nr] = False
