import json
import os


RESOURCE_STORAGE_FMT = "{}_resources"


def initialize(storage):
    if os.path.exists(storage):
        raise ValueError("Storage: {} already exists")

    with open(storage, "w") as f:
        json.dump({}, f)

    os.mkdir(RESOURCE_STORAGE_FMT.format(storage))


def _add_to_manifest(name, storage, location):
    with open(storage) as f:
        manifest = json.load(f)

    if name in manifest:
        raise KeyError(
            "Resource {} is already in manifest".format(name)
        )

    manifest[name] = location
    with open(storage, "w") as f:
        json.dump(manifest, f)


def _dump_resource(resource, resource_location):
    with open(resource_location, "w") as f:
        f.write(resource)


def _is_valid_name(name):
    return name.isalnum()


def add_resource(resource_name, resource, storage):
    if not _is_valid_name(resource_name):
        err_msg = "{} is an invalid resource name"
        raise ValueError(err_msg.format(resource_name))

    resource_location = os.path.join(
        RESOURCE_STORAGE_FMT.format(storage),
        "{}".format(resource_name),
    )
    _add_to_manifest(resource_name, storage, resource_location)
    _dump_resource(resource, resource_location)


def add_json_resource(resource_name, resource, storage):
    add_resource(resource_name, json.dumps(resource), storage)


def load_resource_names(storage):
    with open(storage) as f:
        manifest = json.load(f)
    return tuple(manifest.keys())


def load_resource(name, storage):
    with open(storage) as f:
        manifest = json.load(f)

    with open(manifest[name]) as f:
        return f.read()


def load_json_resource(name, storage):
    return json.loads(load_resource(name, storage))
