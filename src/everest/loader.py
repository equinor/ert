from everest.config import EverestConfig


# Can be removed when refs from Everest-models are updated to
# EverestConfig.load_yaml_file
def load(config_path: str) -> EverestConfig:
    """
    Static method that receives the path to an ASCII file,
    parses it with respect to YAML format and returns a dictionary.

    The method will check if the path is absolute or relative,
    and in case it is relative it will consider to be relative to the working folder.

    @:param config_path: path to the ascii file
    @:type config_path: str
    @:rtype dict
    @:raises yaml.YAMLError
    """

    return EverestConfig.load_file(config_path)
