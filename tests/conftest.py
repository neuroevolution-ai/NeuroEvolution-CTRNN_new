import pytest
import os
from tools.helper import config_from_file
from tools.configurations import ExperimentCfg, ContinuousTimeRNNCfg


@pytest.fixture
def config() -> ExperimentCfg:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config_location = os.path.join(current_directory, "basic_test_config.json")
    global_config = config_from_file(config_location)
    return global_config


@pytest.fixture
def brain_config(config: ExperimentCfg) -> ContinuousTimeRNNCfg:
    return config.brain
