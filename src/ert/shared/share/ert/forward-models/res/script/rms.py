#!/usr/bin/env python
import argparse
import json
import os
import os.path
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import yaml


class RMSConfig:
    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "rms_config.yml")

    def __init__(self):
        config_file = os.getenv("RMS_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        with open(config_file, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError:
                raise ValueError(f"Failed to parse: {config_file} as yaml")

        self._config = config

    @property
    def executable(self):
        exe = self._config["executable"]
        if not os.access(exe, os.X_OK):
            raise OSError(f"The executable: {exe} can not run")

        return exe

    @property
    def wrapper(self):
        exe = self._config.get("wrapper", None)
        if exe is not None and shutil.which(exe) is None:
            raise OSError(f"The executable: {exe} is not found")
        return exe

    @property
    def threads(self):
        return self._config.get("threads")

    def env(self, version):
        env_versions = self._config.get("env", {})
        return env_versions.get(version, {})


@contextmanager
def pushd(path):
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)


class RMSRunException(Exception):
    pass


class RMSRun:
    _single_seed_file = "RMS_SEED"
    _multi_seed_file = "random.seeds"
    _max_seed = 2146483648
    _seed_factor = 7907

    def __init__(
        self,
        iens,
        project,
        workflow,
        run_path="rms",
        target_file=None,
        export_path="rmsEXPORT",
        import_path="rmsIMPORT",
        version=None,
        readonly=True,
        allow_no_env=False,
    ):
        if not os.path.isdir(project):
            raise OSError(f"The project:{project} does not exist as a directory.")

        self.config = RMSConfig()
        self.project = os.path.abspath(project)
        self.workflow = workflow
        self.run_path = run_path
        self.version = version
        self.readonly = readonly
        self.import_path = import_path
        self.export_path = export_path
        self.allow_no_env = allow_no_env
        if target_file is None:
            self.target_file = None
        else:
            if os.path.isabs(target_file):
                self.target_file = target_file
            else:
                self.target_file = os.path.join(os.getcwd(), target_file)

            if os.path.isfile(self.target_file):
                self.target_file_mtime = os.path.getmtime(self.target_file)
            else:
                self.target_file_mtime = None

        self.init_seed(iens)

    def init_seed(self, iens):
        if "RMS_SEED" in os.environ:
            seed = int(os.getenv("RMS_SEED"))
            for x in range(iens):
                seed *= RMSRun._seed_factor
        else:
            single_seed_file = os.path.join(self.run_path, RMSRun._single_seed_file)
            multi_seed_file = os.path.join(self.run_path, RMSRun._multi_seed_file)

            if os.path.exists(single_seed_file):
                # Using existing single seed file
                with open(single_seed_file, encoding="utf-8") as fileH:
                    seed = int(float(fileH.readline()))
            elif os.path.exists(multi_seed_file):
                with open(multi_seed_file, encoding="utf-8") as fileH:
                    seed_list = [int(x) for x in fileH.readlines()]
                seed = seed_list[iens + 1]
            else:
                random.seed()
                seed = random.randint(0, RMSRun._max_seed)

        self.seed = seed % RMSRun._max_seed

    def run(self):
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        self_exe, _ = os.path.splitext(os.path.basename(sys.argv[0]))
        exec_env = {}

        config_env = self.config.env(self.version)
        if not config_env and not self.allow_no_env:
            raise RMSRunException(
                f"RMS environment not specified for version: {self.version}"
            )
        exec_env_file = f"{self_exe}_exec_env.json"
        user_env = {}
        if os.path.isfile(exec_env_file):
            with open(exec_env_file, encoding="utf-8") as f:
                user_env = json.load(f)

        for var in set(config_env.keys()) | set(user_env.keys()):
            exec_env[var] = ":".join(
                filter(None, [user_env.get(var), config_env.get(var)])
            )
            if not exec_env[var].strip():
                exec_env.pop(var)

        with pushd(self.run_path):
            now = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time.time()))
            with open("RMS_SEED_USED", "a+", encoding="utf-8") as filehandle:
                filehandle.write(f"{now} ... {self.seed}\n")

            if not os.path.exists(self.export_path):
                os.makedirs(self.export_path)

            if not os.path.exists(self.import_path):
                os.makedirs(self.import_path)

            exit_status = self.exec_rms(exec_env)

        if exit_status != 0:
            raise RMSRunException(f"The RMS run failed with exit status: {exit_status}")

        if self.target_file is None:
            return

        if not os.path.isfile(self.target_file):
            raise RMSRunException(
                f"The RMS run did not produce the expected file: {self.target_file}"
            )

        if self.target_file_mtime is None:
            return

        if os.path.getmtime(self.target_file) == self.target_file_mtime:
            raise RMSRunException(
                f"The target file:{self.target_file} is unmodified - "
                "interpreted as failure"
            )

    def exec_rms(self, exec_env):
        # The rms exec environement needs to be injected between executing the
        # wrapper and launching rms. PATH_PREFIX must be set in advance.
        prefix_path = exec_env.pop("PATH_PREFIX", "")
        env_args = ["env", *(f"{key}={value}" for key, value in exec_env.items())]
        args = (
            ["env", f"PATH_PREFIX={prefix_path}", self.config.wrapper] + env_args
            if self.config.wrapper is not None
            else env_args
        )

        args += [
            self.config.executable,
            "-project",
            self.project,
            "-seed",
            str(self.seed),
            "-nomesa",
            "-export_path",
            self.export_path,
            "-import_path",
            self.import_path,
            "-batch",
            self.workflow,
        ]

        if self.version:
            args += ["-v", self.version]

        if self.readonly:
            args += ["-readonly"]

        if self.config.threads:
            args += ["-threads", str(self.config.threads)]
        comp_process = subprocess.run(args=args, check=False)
        return comp_process.returncode


def _build_argument_parser():
    description = "Wrapper script to run rms."
    usage = (
        "The script must be invoked with minimum three positional arguments:\n\n"
        "   rms  iens  project  workflow \n\n"
        "Optional arguments supported: \n"
        "  target file [-t][--target-file]\n"
        "  run path [-r][--run-path] default=rms/model\n"
        "  import path [-i][--import-path] default=./ \n"
        "  export path [-e][--export-path] default=./ \n"
        "  version [-v][--version]\n"
    )
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument(
        "iens",
        type=int,
        help="Realization number",
    )
    parser.add_argument(
        "project",
        help="The RMS project we are running",
    )
    parser.add_argument(
        "workflow",
        help="The rms workflow we intend to run",
    )
    parser.add_argument(
        "-r",
        "--run-path",
        default="rms/model",
        help="The directory which will be used as cwd when running rms",
    )
    parser.add_argument(
        "-t",
        "--target-file",
        default=None,
        help="name of file which should be created/updated by rms",
    )
    parser.add_argument(
        "-i",
        "--import-path",
        default="./",
        help="the prefix of all relative paths when rms is importing",
    )
    parser.add_argument(
        "-e",
        "--export-path",
        default="./",
        help="the prefix of all relative paths when rms is exporting",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        help="The version of rms to use",
    )
    parser.add_argument(
        "-a",
        "--allow-no-env",
        action="store_true",
        help="Allow RMS to run without a site configured environment",
    )
    return parser


def run(
    iens,
    project,
    workflow,
    run_path="rms",
    target_file=None,
    export_path="rmsEXPORT",
    import_path="rmsIMPORT",
    version=None,
    readonly=True,
    allow_no_env=False,
):
    run_object = RMSRun(
        iens,
        project,
        workflow,
        run_path=run_path,
        target_file=target_file,
        export_path=export_path,
        import_path=import_path,
        version=version,
        readonly=readonly,
        allow_no_env=allow_no_env,
    )
    run_object.run()


# The first three arguments; iens, project and workflow are positional
# and *must* be supplied. The run_path and target_file arguments are optional.

if __name__ == "__main__":
    # old style jobs pass inn empty arguments as "" and causes argparse to fail
    sys.argv = [arg for arg in sys.argv if arg != ""]
    arg_parser = _build_argument_parser()
    args = arg_parser.parse_args()

    run(
        args.iens,
        args.project,
        args.workflow,
        run_path=args.run_path,
        target_file=args.target_file,
        import_path=args.import_path,
        export_path=args.export_path,
        version=args.version,
        allow_no_env=args.allow_no_env,
    )
