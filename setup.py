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
    scripts=['ert_shared/bin/ert', 'ert_shared/bin/ert3'],
    packages=[
            'ert_shared',
            'ert_shared.models',
            'ert_gui',
            'ert_gui.ertwidgets',
            'ert_gui.ide',
            'ert_gui.plottery',
            'ert_gui.simulation',
            'ert_gui.tools',
            'ert_gui.ertwidgets.models',
            'ert_gui.ide.completers',
            'ert_gui.ide.keywords',
            'ert_gui.ide.wizards',
            'ert_gui.ide.keywords.data',
            'ert_gui.ide.keywords.definitions',
            'ert_gui.plottery.plots',            
            'ert_gui.tools.export',
            'ert_gui.tools.help',
            'ert_gui.tools.ide',
            'ert_gui.tools.load_results',
            'ert_gui.tools.manage_cases',
            'ert_gui.tools.plot',
            'ert_gui.tools.plugins',
            'ert_gui.tools.run_analysis',
            'ert_gui.tools.workflows',
            'ert_gui.tools.plot.customize',
            'ert_gui.tools.plot.widgets',
            'ErtQt',
            'ErtQt.Qt',
            ],
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
                    'pytest',
                    'decorator'
                    ],
    zip_safe=False,
    tests_require=['pytest', 'mock'],
    tests_suite=['tests'],
    setup_requires=["pytest-runner", 'setuptools_scm'],
)
