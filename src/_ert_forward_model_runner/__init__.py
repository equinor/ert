"""_ert_forward_model_runner is called by ert to run jobs in the runpath.

Its is split into its own toplevel package for performance reasons,
simply importing ert can take several seconds, which is not ideal when
_ert_forward_model_runner is initialized potentially 100s of times.
"""
