import json
import os


def _register_in_storage(name, resource_storage, observation_location):
    with open(resource_storage) as f:
        storage = json.load(f)

    if name in storage:
        raise KeyError(
            "Observation {} is already a resource".format(name)
        )

    storage[name] = observation_location
    with open(resource_storage, "w") as f:
        json.dump(storage, f)


def _register_in_location(observation, observation_location):
    with open(observation_location, "w") as f:
        json.dump({
            "name": observation.name,
            "type": observation.type,
            "observations": [
                {"value": obs.value, "std": obs.std}
                for obs in observation.observations
            ],
        }, f)


def register_as_resource(observation, resource_storage, resource_location):
    observation_location = os.path.join(
        resource_location,
        "{}.json".format(observation.name),
    )

    _register_in_storage(observation.name, resource_storage, observation_location)
    _register_in_location(observation, observation_location)
