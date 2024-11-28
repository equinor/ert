from everest.plugins import hookimpl


@hookimpl
def visualize_data(api):
    print("No visualization plugin installed!")


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


@hookimpl
def get_forward_model_documentations():
    return None
