import os
import setuptools
import skbuild
from setuptools_scm import get_version


def get_ecl_include():
    from ecl import get_include

    return get_include()


def get_data_files():
    data_files = []
    for root, _, files in os.walk("share/ert"):
        data_files.append((root, [os.path.join(root, name) for name in files]))
    return data_files


version = get_version(
    relative_to=__file__,
    write_to="python/res/_version.py",
    write_to_template='# config: utf-8\n#\nversion = "{version}"',  # black-compatible version string
)

with open("README.md") as f:
    long_description = f.read()


skbuild.setup(
    name="equinor-libres",
    author="Equinor ASA",
    author_email="fg_sib-scout@equinor.com",
    description="Part of the Ensemble based Reservoir Tool (ERT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/libres",
    packages=setuptools.find_packages(
        where="python",
        exclude=["doc", "*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_dir={"": "python"},
    package_data={
        "res": [
            "fm/rms/rms_config.yml",
            "fm/ecl/ecl300_config.yml",
            "fm/ecl/ecl100_config.yml",
        ]
    },
    data_files=get_data_files(),
    license="GPL-3.0",
    platforms="any",
    install_requires=[
        "cloudevents",
        "cwrap",
        "ecl",
        "jinja2",
        "numpy",
        "pandas",
        "psutil",
        "pyyaml",
        "requests",
        "websockets >= 9.0.1",
    ],
    entry_points={
        "console_scripts": ["job_dispatch.py = job_runner.job_dispatch:main"]
    },
    cmake_args=[
        "-DRES_VERSION=" + version,
        "-DECL_INCLUDE_DIRS=" + get_ecl_include(),
        # we can safely pass OSX_DEPLOYMENT_TARGET as it's ignored on
        # everything not OS X. We depend on C++11, which makes our minimum
        # supported OS X release 10.9
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9",
        "-DCMAKE_INSTALL_LIBDIR=python/res/.libs",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    version=version,
    test_suite="tests",
)
