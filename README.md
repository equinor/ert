<h1 align="center">
<img src="https://raw.githubusercontent.com/equinor/ert/main/src/ert/gui/resources/gui/img/ert_icon.svg" width="200">
</h1>

[![Build Status](https://github.com/equinor/ert/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/build_and_test.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ert)](https://img.shields.io/pypi/pyversions/ert)
[![Code Style](https://github.com/equinor/ert/actions/workflows/style.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/style.yml)
[![Type checking](https://github.com/equinor/ert/actions/workflows/typing.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/typing.yml)
[![codecov](https://codecov.io/gh/equinor/ert/graph/badge.svg?token=keVAcWavZ1)](https://codecov.io/gh/equinor/ert)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ert - Ensemble based Reservoir Tool - is designed for running
ensembles of dynamical models such as reservoir models,
in order to do sensitivity analysis and data assimilation.
ert supports data assimilation using the Ensemble Smoother (ES) and
Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

## Installation

```sh
pip install ert
ert --help
```

or, for the latest development version:

```sh
pip install git+https://github.com/equinor/ert.git@main
ert --help
```

For examples and help with configuration, see the [ert Documentation](https://ert.readthedocs.io/en/latest/getting_started/configuration/poly_new/guide.html#configuration-guide).

# EVEREST™

<h1 align="center">
<img src="https://raw.githubusercontent.com/equinor/ert/main/src/everest/assets/everest_logo.svg" width="300">
</h1>

The primary goal of the EVEREST tool is to find *optimal* well
planning and production strategies by utilizing an ensemble of
reservoir models (e.g., an ensemble of geologically-consistent models).
This will enable robust decisions about drilling schedule and well
placement, in order to achieve results of significant practical value.

```sh
pip install ert[everest]
```

## Developing

We use uv to have one synchronized development environment for all packages.
See [installing uv](https://docs.astral.sh/uv/getting-started/installation/). We
recommend either installing uv using your systems package manager, or creating
a small virtual environment you intall base packages into (such as `uv` and `pre-commit`).

Once uv is installed, you can get a development environment by running:

```sh
git clone https://github.com/equinor/ert
cd ert
uv sync --all-extras
```

### Test setup

The tests can be ran with pytest directly, but this is very slow:

```sh
uv run pytest tests/
```

There are many kinds of tests in the `tests` directory, while iterating on your
code you can run a fast subset of the tests with by using the rapid checks from the
justfile:

```sh
uv run just rapid-tests
```

You can also run all of the checks in parallel with

```sh
uv run just check-all
```

[Git LFS](https://git-lfs.com/) must be installed to get all the files. This is
packaged as `git-lfs` on Ubuntu, Fedora or macOS Homebrew.  For Equinor TGX
users, it is preinstalled.

If you have not used git-lfs before, you might have to make changes to your global Git config for git-lfs to work properly.
```sh
git lfs install
```

test-data/ert/block_storage is a submodule and must be checked out.
```sh
git submodule update --init --recursive
```

If you checked out submodules without having git lfs installed, you can force git lfs to run in all submodules with:
```sh
git submodule foreach "git lfs pull"
```

### Build documentation

You can build the documentation after installation by running
```sh
uv run just build-docs
```
and then open the generated `./ert_docs/index.html` or
`./everest_docs/index.html` in a browser.

To automatically reload on changes you may use

```sh
uv run sphinx-autobuild docs docs/_build/html
```

### Style requirements

There are a set of style requirements, which are gathered in the `pre-commit`
configuration, to have it automatically run on each commit do:

```sh
pip install pre-commit
pre-commit install
```

There is also a pre-push hook configured in `pre-commit` to run a collection of
relatively fast tests, to install this hook:

```sh
pre-commit install --hook-type pre-push
```


### Trouble with setup

As a simple test of your `ert` installation, you may try to run one of the
examples, for instance:


```sh
uv run just poly
```
This opens up the ert graphical user interface with a simple example using
polynomials (see `./test-data/ert/poly_example`).

Finally, test ert by starting and successfully running the experiment.

### Notes

The default maximum number of open files is normally relatively low on MacOS
and some Linux distributions. This is likely to make tests crash with mysterious
error-messages. You can inspect the current limits in your shell by issuing the
command `ulimit -a`. In order to increase maximum number of open files, run
`ulimit -n 16384` (or some other large number) and put the command in your
`.profile` to make it persist.

### ert with a reservoir simulator
To actually get ert to work at your site you need to configure details about
your system; at the very least this means you must configure where your
reservoir simulator is installed. In addition you might want to configure e.g.
queue system in the `site-config` file, but that is not strictly necessary for
a basic test.
