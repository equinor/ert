from hypothesis import given

from ert._c_wrappers.enkf import AnalysisConfig, ResConfig

from .config_dict_generator import config_generators


@given(config_generators())
def test_analysis_config_config_same_as_from_file(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        analysis_config_from_file = ResConfig(filename).analysis_config
        analysis_config_from_dict = AnalysisConfig.from_dict(config_dict=config_dict)
        assert analysis_config_from_file == analysis_config_from_dict
