from setuptools import setup, find_packages





setup(
    name='Ensemble Reservoir Tool',
    use_scm_version={'root':'.' , 'write_to': 'python/python/ert_gui/version.py'},
    scripts=['bin/ert.in', 'python/python/bin/ert_cli'],
    packages=find_packages(where="python/python") ,
    package_dir={'':'python/python'},
    license='Open Source',
    long_description=open('README.md').read(),
    install_requires=[
                    'sphinx',
                    'cwrap',
                    'numpy',
                    'pandas',
                    'matplotlib<3',
                    'scipy'
                    ],
    zip_safe=False,
    tests_require=['pytest'],
    setup_requires=["pytest-runner", 'setuptools_scm'],
)