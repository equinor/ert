import os
import sys

from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup

# Corporate networks tend to be behind a proxy server with their own non-public
# SSL certificates. Conan keeps its own certificates, whose path we can override
if "CONAN_CACERT_PATH" not in os.environ:
    # Look for a RHEL-compatible system-wide file
    for file_ in ("/etc/pki/tls/cert.pem",):
        if not os.path.isfile(file_):
            continue
        os.environ["CONAN_CACERT_PATH"] = file_
        break


def get_ecl_include():
    from ecl import get_include

    return get_include()


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


with open("README.md") as f:
    long_description = f.read()

packages = find_packages(
    exclude=["*.tests", "*.tests.*", "tests.*", "tests", "tests*", "libres"],
)

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
        "ert_shared": package_files("ert_shared/share/"),
        "ert_gui": package_files("ert_gui/resources/"),
        "ert_logging": ["logger.conf"],
        "ert3_examples": package_files("ert3_examples/"),
        "res": [
            "fm/rms/rms_config.yml",
            "fm/ecl/ecl300_config.yml",
            "fm/ecl/ecl100_config.yml",
        ],
    },
    include_package_data=True,
    license="GPL-3.0",
    platforms="any",
    install_requires=[
        "aiofiles",
        "aiohttp",
        "alembic",
        "ansicolors==1.1.8",
        "async-exit-stack; python_version < '3.7'",
        "async-generator",
        "beartype >= 0.9.1",
        "cloudevents",
        "cloudpickle",
        "tqdm>=4.62.0",
        "cryptography",
        "cwrap",
        "dask_jobqueue",
        "decorator",
        "deprecation",
        "dnspython >= 2",
        "ecl >= 2.12.0",
        "ert-storage >= 0.3.4, < 0.3.7",
        "fastapi==0.70.1",
        "graphene",
        "graphlib_backport; python_version < '3.9'",
        "jinja2",
        "matplotlib",
        "numpy",
        "packaging",
        "pandas <= 1.3.5",
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
        "SALib",
        "scipy",
        "semeio",
        "sqlalchemy",
        "typing-extensions; python_version < '3.8'",
        "uvicorn < 0.17.0; python_version <= '3.6'",
        "uvicorn >= 0.17.0; python_version > '3.6'",
        "websockets >= 9.0.1",
        "httpx",
    ],
    setup_requires=["pytest-runner", "setuptools_scm"],
    entry_points={
        "console_scripts": [
            "ert3=ert3.console:main",
            "ert=ert_shared.main:main",
            "job_dispatch.py = job_runner.job_dispatch:main",
        ]
    },
    cmake_args=[
        "-DECL_INCLUDE_DIRS=" + get_ecl_include(),
        # we can safely pass OSX_DEPLOYMENT_TARGET as it's ignored on
        # everything not OS X. We depend on C++17, which makes our minimum
        # supported OS X release 10.15
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
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
