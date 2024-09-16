"""_ert.forward_model_runner is called by ert to run jobs in the runpath.

It is split into its own package for performance reasons,
simply importing ert can take several seconds, which is not ideal when
_ert.forward_model_runner is initialized potentially 100s of times.
"""
