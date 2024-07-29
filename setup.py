import os

from setuptools import find_packages, setup


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", "..", path, filename))
    return paths


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "ert": package_files("src/ert/gui/resources/")
        + package_files("src/ert/resources/")
        + ["logging/logger.conf", "logging/storage_log.conf"]
    },
)
