from everest.plugins import hookimpl


@hookimpl
def visualize_data(api):
    print("No visualization plugin installed!")


@hookimpl
def default_site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "",
    ]


@hookimpl
def site_config_lines():
    return None


@hookimpl
def ecl100_config_path():
    return None


@hookimpl
def ecl300_config_path():
    return None


@hookimpl
def flow_config_path():
    return None


@hookimpl
def get_forward_models():
    return None


@hookimpl
def lint_forward_model():
    return None


@hookimpl
def parse_forward_model_schema(path, schema):
    return None


@hookimpl
def get_forward_models_schemas():
    return None


@hookimpl
def installable_workflow_jobs():
    return None
