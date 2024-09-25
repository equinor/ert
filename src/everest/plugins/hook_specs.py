from typing import Sequence, Type, TypeVar

from everest.plugins import hookspec

T = TypeVar("T")


@hookspec(firstresult=True)
def visualize_data(api):
    """
    :param :EverestAPI instance
    """


@hookspec(firstresult=True)
def default_site_config_lines():
    """
    :return: List default site config of lines to
    :rtype: List of strings
    """


@hookspec(firstresult=True)
def install_job_directories():
    """
    :return: List default site config of lines to
    :rtype: List of strings
    """


@hookspec()
def site_config_lines():
    """
    :return: List of lines to append to site config file
    :rtype: PluginResponse with data as list[str]
    """


@hookspec(firstresult=True)
def ecl100_config_path():
    """
    :return: Path to ecl100 config file
    :rtype: PluginResponse with data as str
    """


@hookspec(firstresult=True)
def ecl300_config_path():
    """
    :return: Path to ecl300 config file
    :rtype: PluginResponse with data as str
    """


@hookspec(firstresult=True)
def flow_config_path():
    """
    :return: Path to flow config file
    :rtype: PluginResponse with data as str
    """


@hookspec
def get_forward_models():
    """
    Return a list of dicts detailing the names and paths to forward models.

    Example [{"name": "job1", "path":"path1"}, {"name": "job2", "path":"path2"}]
    """


@hookspec(firstresult=True)
def lint_forward_model(job: str, args: Sequence[str]):
    """
    Return a error string, if forward model job failed to lint.
    """


@hookspec
def get_forward_models_schemas():
    """
    Return a dictionary of forward model names and its associated: schemas.
    Example {"add_template": {"-c/--config": WellModelConfig}, ...}
    """


@hookspec
def parse_forward_model_schema(path: str, schema: Type[T]):
    """
    Given a path and schema type, this hook will parse the file.
    """


@hookspec
def installable_workflow_jobs():
    """
    :return: dict with workflow job names as keys and path to config as value
    :rtype: PluginResponse with data as dict[str,str]
    """


@hookspec
def add_log_handle_to_root():
    """
    Create a log handle which will be added to the root logger
    in the main entry point.
    :return: A log handle that will be added to the root logger
    :rtype: logging.Handler
    """
