import uuid
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List

from ert._c_wrappers.enkf.run_arg import RunArg
from ert._c_wrappers.enkf.runpaths import Runpaths
from ert._c_wrappers.util import SubstitutionList

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.enkf_fs import EnkfFs


@dataclass
class RunContext:
    sim_fs: "EnkfFs"
    path_format: str
    format_string: str
    runpath_file: Path
    initial_mask: List[bool] = field(default_factory=list)
    iteration: int = 0
    global_substitutions: InitVar[Dict[str, str]] = None

    def __post_init__(self, global_substitutions):
        subst_list = SubstitutionList()
        if global_substitutions:
            for k, v in global_substitutions.items():
                subst_list.addItem(k, v)
        self.substituter = subst_list
        self.runpaths = Runpaths(
            self.path_format,
            self.format_string,
            self.runpath_file,
            self.substituter.substitute_real_iter,
        )
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

    def __getitem__(self, item) -> RunArg:
        return self.run_args[item]

    def __iter__(self) -> Iterator[RunArg]:
        yield from self.run_args

    def deactivate_realization(self, realization_nr):
        self[realization_nr].active = False
