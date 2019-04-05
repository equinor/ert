from setuptools import setup, find_packages


import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('ert_gui/resources/')


setup(
    name='Ensemble Reservoir Tool',
    use_scm_version={'root':'.' , 'write_to': 'ert_gui/version.py'},
    scripts=['ert_gui/bin/ert', 'ert_gui/bin/ert_cli'],
    packages=find_packages(where=".") ,
    packages_dir='.',
    package_data={'ert_gui' : extra_files},
    include_package_data=True,
    license='Open Source',
    long_description=open('README.md').read(),
    install_requires=[
                    'sphinx',
                    'cwrap',
                    'numpy',
                    'pandas',
                    'matplotlib<3',
                    'scipy',
                    'pytest'
                    ],
    zip_safe=False,
    tests_require=['pytest'],
    setup_requires=["pytest-runner", 'setuptools_scm'],
)