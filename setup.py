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
    name="Ensemble Reservoir Tool",
    use_scm_version={"root": ".", "write_to": "ert_shared/version.py"},
    scripts=["ert_shared/bin/ert"],
    packages=find_packages(exclude=["tests*"]),
    package_data={"ert_gui": extra_files, "ert_logging": logging_configuration},
    include_package_data=True,
    license="Open Source",
    long_description=open("README.md").read(),
    install_requires=[
        "ansicolors==1.1.8",
        "console-progressbar==1.1.2",
        "decorator",
        "flask",
        "jinja2",
        "matplotlib < 3.2",
        "numpy",
        "pandas",
        "pluggy",
        "PyQt5",
        "pyyaml",
        "qtpy",
        "scipy",
        "sqlalchemy",
        "decorator",
    ],
    zip_safe=False,
    tests_require=["pytest", "mock"],
    test_suite="tests",
    setup_requires=["pytest-runner", "setuptools_scm"],
)
