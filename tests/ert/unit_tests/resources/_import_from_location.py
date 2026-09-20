import importlib.util
import sys


def import_from_location(name, location):
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None:
        raise ImportError(f"Could not find {name}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(name)
    had_previous_module = name in sys.modules
    sys.modules[name] = module
    try:
        if spec.loader is None:
            raise ImportError(f"No loader for {name}")
        spec.loader.exec_module(module)
    finally:
        if had_previous_module:
            sys.modules[name] = previous_module
        else:
            sys.modules.pop(name, None)
    return module
