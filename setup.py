from setuptools import setup, find_packages





setup(
    name='Ensemble Reservoir Tool',
    use_scm_version=True,
    scripts=['bin/ert.in', 'python/python/bin/ert_cli'],
    packages=find_packages(where="python/python") ,
    package_dir={'':'python/python'},
    package_data={
        '' : 'share/*'
    },
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
    setup_requires=["pytest-runner"],
)