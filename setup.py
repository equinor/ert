import os
from setuptools import find_packages
from skbuild import setup
from setuptools_scm import get_version


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
        "async-generator; python_version < '3.7'",
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
    extras_require={
        "storage": [
            "ert-storage==0.2.0",
        ],
    },
    setup_requires=["pytest-runner", "setuptools_scm"],
    entry_points={
        "console_scripts": [
            "ert3=ert3.console:main",
            "ert=ert_shared.main:main",
            "job_dispatch.py = job_runner.job_dispatch:main",
        ]
    },
    cmake_args=[
        "-DRES_VERSION=" + get_version(),
        "-DECL_INCLUDE_DIRS=" + get_ecl_include(),
        # we can safely pass OSX_DEPLOYMENT_TARGET as it's ignored on
        # everything not OS X. We depend on C++11, which makes our minimum
        # supported OS X release 10.9
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9",
        f"-DCMAKE_INSTALL_LIBDIR=res/.libs",
    ],
    cmake_source_dir="libres/",
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
