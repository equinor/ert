import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import EnkfInitModeEnum
from res.enkf.run_arg import RunArg


@dataclass
class ErtRunContext:
    sim_fs: Optional[EnkfFs]
    target_fs: Optional[EnkfFs]
    mask: List[bool]
    paths: List[str]
    jobnames: Optional[List[str]]
    itr: int = 0
    init_mode: EnkfInitModeEnum.INIT_CONDITIONAL = EnkfInitModeEnum.INIT_CONDITIONAL

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
                        self.itr,
                        path,
                        job_name,
                    )
                )

    @classmethod
    def ensemble_experiment(
        cls, sim_fs, mask: List[bool], paths, jobnames, itr
    ) -> "ErtRunContext":
        return cls(
            sim_fs=sim_fs,
            target_fs=None,
            mask=mask,
            paths=paths,
            jobnames=jobnames,
            itr=itr,
        )

    @classmethod
    def ensemble_smoother(
        cls, sim_fs, target_fs, mask: List[bool], paths, jobnames, itr
    ) -> "ErtRunContext":
        return cls(
            sim_fs,
            target_fs,
            mask,
            paths,
            jobnames,
            itr,
        )

    @classmethod
    def ensemble_smoother_update(
        cls,
        sim_fs,
        target_fs,
    ) -> "ErtRunContext":
        return cls(
            mask=[],
            sim_fs=sim_fs,
            target_fs=target_fs,
            paths=[],
            jobnames=[],
        )

    @classmethod
    def case_init(cls, sim_fs, mask=None) -> "ErtRunContext":
        if mask == None:
            mask = []
        return cls(
            init_mode=EnkfInitModeEnum.INIT_FORCE,
            mask=mask,
            sim_fs=sim_fs,
            target_fs=None,
            paths=[],
            jobnames=[],
        )

    def is_active(self, index: int) -> bool:
        try:
            return self.mask[index]
        except IndexError:
            return False

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, item) -> RunArg:
        return self.run_args[item]

    def __iter__(self) -> RunArg:
        yield from self.run_args

    def __repr__(self):
        return f"ErtRunContext(size = {len(self)})"

    def get_id(self) -> str:
        return self.run_id

    def get_mask(self) -> List[bool]:
        return self.mask

    def get_iter(self) -> int:
        return self.itr

    def get_target_fs(self) -> EnkfFs:
        return self.target_fs

    def get_sim_fs(self) -> EnkfFs:
        return self.sim_fs

    def get_init_mode(self):
        return self.init_mode

    def deactivate_realization(self, realization_nr):
        self.mask[realization_nr] = False
