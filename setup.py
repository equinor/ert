from setuptools import setup, find_packages


import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("ert_gui/resources/")
logging_configuration = package_files("ert_logging/")

setup(
    name="ert",
    author="Equinor ASA",
    author_email="fg_sib-scout@equinor.com",
    use_scm_version={"root": ".", "write_to": "ert_shared/version.py"},
    entry_points={
        "console_scripts": [
            "ert3=ert3.console:main",
            "ert=ert_shared.main:main",
        ]
    },
    packages=find_packages(exclude=["tests*"]),
    package_data={"ert_gui": extra_files, "ert_logging": logging_configuration},
    include_package_data=True,
    license="Open Source",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
        "dask_jobqueue",
        "decorator",
        "deprecation",
        "dnspython >= 2",
        "equinor-libres == 11.0.0b0",
        "fastapi",
        "graphlib_backport; python_version < '3.9'",
        "jinja2",
        "matplotlib",
        "numpy",
        "pandas",
        "pluggy",
        "prefect",
        "pydantic >= 1.8.1",
        "PyQt5",
        "pyrsistent",
        "python-dateutil",
        "pyyaml",
        "qtpy",
        "scipy",
        "semeio",
        "sqlalchemy",
        "typing-extensions; python_version < '3.8'",
        "uvicorn",
        "websockets >= 9.0.1",
    ],
    extras_require={
        "storage": [
            "ert-storage==0.1.7",
        ],
    },
    zip_safe=False,
    tests_require=["pytest", "mock"],
    test_suite="tests",
    setup_requires=["pytest-runner", "setuptools_scm"],
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
)
