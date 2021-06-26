import os
import sys
from subprocess import check_output

from pathlib import Path
from setuptools import find_packages, setup, Extension
from setuptools_scm import get_version


CXX = os.getenv("CXX", "c++")  # C++ compiler binary


def get_ecl_include():
    from ecl import get_include

    return get_include()


def get_data_files():
    data_files = []
    for root, _, files in os.walk("share/ert"):
        data_files.append((root, [os.path.join(root, name) for name in files]))
    return data_files


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


def get_libres_extension():
    cxx_std = "-std=c++17"

    srcdir = Path(__file__).parent / "libres" / "lib"

    if sys.platform == "darwin":
        os.environ["LDFLAGS"] = "-framework Accelerate"
    elif sys.platform == "linux":
        pass
    else:
        sys.exit("Unsupported operating system. ERT supports only Linux and macOS")

    return Extension(
        "res._lib",
        [str(srcdir / src) for src in LIBRES_SOURCES],
        language="c++",
        extra_compile_args=[cxx_std],
        include_dirs=[
            get_ecl_include(),
            str(srcdir / "private-include/ext/json"),
            str(srcdir / "include"),
        ],
        define_macros=[
            ("RES_VERSION_MAJOR", "1"),
            ("RES_VERSION_MINOR", "1"),
            ("INTERNAL_LINK", "1"),
        ],
        libraries=["lapack"],
    )


extra_files = package_files("ert_gui/resources/")
logging_configuration = package_files("ert_logging/")
ert3_example_files = package_files("ert3_examples/")


with open("README.md") as f:
    long_description = f.read()

packages = find_packages(
    exclude=["*.tests", "*.tests.*", "tests.*", "tests", "tests*", "libres"],
)

# Given this unusual layout where we cannot fall back on a "root package",
# package_dir is built manually from libres_packages.
res_files = get_data_files()


LIBRES_SOURCES = [
    "res_util/res_log.cpp",
    "res_util/log.cpp",
    "res_util/es_testdata.cpp",
    "res_util/arg_pack.cpp",
    "res_util/ui_return.cpp",
    "res_util/subst_list.cpp",
    "res_util/subst_func.cpp",
    "res_util/matrix_stat.cpp",
    "res_util/matrix_blas.cpp",
    "res_util/matrix_lapack.cpp",
    "res_util/matrix.cpp",
    "res_util/template.cpp",
    "res_util/path_fmt.cpp",
    "res_util/res_env.cpp",
    "res_util/res_portability.cpp",
    "res_util/util_printf.cpp",
    "res_util/block_fs.cpp",
    "res_util/res_version.cpp",
    "res_util/regression.cpp",
    "res_util/thread_pool.cpp",
    "res_util/template_loop.cpp",
    "config/conf_util.cpp",
    "config/conf.cpp",
    "config/conf_data.cpp",
    "config/config_parser.cpp",
    "config/config_content.cpp",
    "config/config_path_stack.cpp",
    "config/config_content_item.cpp",
    "config/config_content_node.cpp",
    "config/config_error.cpp",
    "config/config_path_elm.cpp",
    "config/config_root_path.cpp",
    "config/config_schema_item.cpp",
    "config/config_settings.cpp",
    "rms/rms_file.cpp",
    "rms/rms_tag.cpp",
    "rms/rms_tagkey.cpp",
    "rms/rms_type.cpp",
    "rms/rms_util.cpp",
    "sched/history.cpp",
    "analysis/analysis_module.cpp",
    "analysis/bootstrap_enkf.cpp",
    "analysis/cv_enkf.cpp",
    "analysis/enkf_linalg.cpp",
    "analysis/fwd_step_enkf.cpp",
    "analysis/fwd_step_log.cpp",
    "analysis/module_data_block.cpp",
    "analysis/module_data_block_vector.cpp",
    "analysis/module_info.cpp",
    "analysis/module_obs_block.cpp",
    "analysis/module_obs_block_vector.cpp",
    "analysis/null_enkf.cpp",
    "analysis/sqrt_enkf.cpp",
    "analysis/std_enkf.cpp",
    "analysis/stepwise.cpp",
    "job_queue/ext_job.cpp",
    "job_queue/ext_joblist.cpp",
    "job_queue/forward_model.cpp",
    "job_queue/job_status.cpp",
    "job_queue/job_list.cpp",
    "job_queue/job_node.cpp",
    "job_queue/job_queue.cpp",
    "job_queue/job_queue_status.cpp",
    "job_queue/local_driver.cpp",
    "job_queue/lsf_driver.cpp",
    "job_queue/queue_driver.cpp",
    "job_queue/rsh_driver.cpp",
    "job_queue/slurm_driver.cpp",
    "job_queue/torque_driver.cpp",
    "job_queue/workflow.cpp",
    "job_queue/workflow_job.cpp",
    "job_queue/workflow_joblist.cpp",
    "job_queue/environment_varlist.cpp",
    "job_queue/job_kw_definitions.cpp",
    "enkf/active_list.cpp",
    "enkf/time_map.cpp",
    "enkf/analysis_config.cpp",
    "enkf/analysis_iter_config.cpp",
    "enkf/block_fs_driver.cpp",
    "enkf/block_obs.cpp",
    "enkf/callback_arg.cpp",
    "enkf/cases_config.cpp",
    "enkf/container.cpp",
    "enkf/container_config.cpp",
    "enkf/data_ranking.cpp",
    "enkf/ecl_config.cpp",
    "enkf/ecl_refcase_list.cpp",
    "enkf/enkf_analysis.cpp",
    "enkf/enkf_config_node.cpp",
    "enkf/enkf_defaults.cpp",
    "enkf/enkf_fs.cpp",
    "enkf/enkf_main.cpp",
    "enkf/enkf_main_jobs.cpp",
    "enkf/enkf_node.cpp",
    "enkf/enkf_obs.cpp",
    "enkf/enkf_plot_data.cpp",
    "enkf/enkf_plot_gendata.cpp",
    "enkf/enkf_plot_gen_kw.cpp",
    "enkf/enkf_plot_gen_kw_vector.cpp",
    "enkf/enkf_plot_genvector.cpp",
    "enkf/enkf_plot_tvector.cpp",
    "enkf/enkf_serialize.cpp",
    "enkf/enkf_state.cpp",
    "enkf/enkf_types.cpp",
    "enkf/enkf_util.cpp",
    "enkf/ensemble_config.cpp",
    "enkf/ert_run_context.cpp",
    "enkf/ert_template.cpp",
    "enkf/ert_test_context.cpp",
    "enkf/ert_workflow_list.cpp",
    "enkf/ext_param.cpp",
    "enkf/ext_param_config.cpp",
    "enkf/field.cpp",
    "enkf/field_config.cpp",
    "enkf/field_trans.cpp",
    "enkf/forward_load_context.cpp",
    "enkf/fs_driver.cpp",
    "enkf/fs_types.cpp",
    "enkf/gen_common.cpp",
    "enkf/gen_data.cpp",
    "enkf/gen_data_config.cpp",
    "enkf/gen_kw.cpp",
    "enkf/gen_kw_config.cpp",
    "enkf/value_export.cpp",
    "enkf/gen_obs.cpp",
    "enkf/hook_manager.cpp",
    "enkf/hook_workflow.cpp",
    "enkf/local_config.cpp",
    "enkf/local_dataset.cpp",
    "enkf/local_ministep.cpp",
    "enkf/local_obsdata.cpp",
    "enkf/local_obsdata_node.cpp",
    "enkf/local_updatestep.cpp",
    "enkf/meas_data.cpp",
    "enkf/misfit_ensemble.cpp",
    "enkf/misfit_member.cpp",
    "enkf/misfit_ranking.cpp",
    "enkf/misfit_ts.cpp",
    "enkf/model_config.cpp",
    "enkf/obs_data.cpp",
    "enkf/obs_vector.cpp",
    "enkf/queue_config.cpp",
    "enkf/ranking_table.cpp",
    "enkf/rng_config.cpp",
    "enkf/run_arg.cpp",
    "enkf/runpath_list.cpp",
    "enkf/site_config.cpp",
    "enkf/rng_manager.cpp",
    "enkf/res_config.cpp",
    "enkf/row_scaling.cpp",
    "enkf/state_map.cpp",
    "enkf/summary.cpp",
    "enkf/summary_config.cpp",
    "enkf/summary_key_matcher.cpp",
    "enkf/summary_key_set.cpp",
    "enkf/summary_obs.cpp",
    "enkf/surface.cpp",
    "enkf/surface_config.cpp",
    "enkf/trans_func.cpp",
    "enkf/subst_config.cpp",
    "enkf/log_config.cpp",
    "enkf/config_keys.cpp",
    "external/JSON/cJSON.cpp",
    "pyext.cpp",
    # rml_enkf
    "analysis/modules/rml_enkf_config.cpp",
    "analysis/modules/rml_enkf.cpp",
    "analysis/modules/rml_enkf_common.cpp",
    "analysis/modules/rml_enkf_log.cpp",
    # ies
    "analysis/modules/ies_enkf.cpp",
    "analysis/modules/ies_enkf_config.cpp",
    "analysis/modules/ies_enkf_data.cpp",
    # std_enkf_debug
    "analysis/modules/std_enkf_debug.cpp",
]

setup(
    name="ert",
    author="Equinor ASA",
    author_email="fg_sib-scout@equinor.com",
    description="Ensemble based Reservoir Tool (ERT)",
    use_scm_version={"root": ".", "write_to": "ert_shared/version.py"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/ert",
    packages=packages,
    package_data={
        "ert_gui": extra_files,
        "ert_logging": logging_configuration,
        "ert3_examples": ert3_example_files,
        "res": [
            "fm/rms/rms_config.yml",
            "fm/ecl/ecl300_config.yml",
            "fm/ecl/ecl100_config.yml",
        ],
    },
    include_package_data=True,
    data_files=res_files,
    license="GPL-3.0",
    platforms="any",
    install_requires=[
        "aiofiles",
        "aiohttp",
        "alembic",
        "ansicolors==1.1.8",
        "async-exit-stack; python_version < '3.7'",
        "async-generator",
        "cloudevents",
        "cloudpickle",
        "console-progressbar==1.1.2",
        "cryptography",
        "cwrap",
        "dask_jobqueue",
        "decorator",
        "deprecation",
        "dnspython >= 2",
        "ecl",
        "ert-storage",
        "fastapi",
        "graphlib_backport; python_version < '3.9'",
        "jinja2",
        "matplotlib",
        "numpy",
        "pandas",
        "pluggy",
        "prefect",
        "psutil",
        "pydantic >= 1.8.1",
        "PyQt5",
        "pyrsistent",
        "python-dateutil",
        "pyyaml",
        "qtpy",
        "requests",
        "scipy",
        "semeio>=1.1.3rc0",
        "sqlalchemy",
        "typing-extensions; python_version < '3.8'",
        "uvicorn",
        "websockets >= 9.0.1",
    ],
    setup_requires=["pytest-runner", "setuptools_scm"],
    entry_points={
        "console_scripts": [
            "ert3=ert3.console:main",
            "ert=ert_shared.main:main",
            "job_dispatch.py = job_runner.job_dispatch:main",
        ]
    },
    ext_modules=[get_libres_extension()],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    test_suite="tests",
)
