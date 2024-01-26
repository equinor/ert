import importlib.util
import sys


def import_from_location(name, location):
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None:
        raise ImportError(f"Could not find {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is None:
        raise ImportError(f"No loader for {name}")
    spec.loader.exec_module(module)
    return module
