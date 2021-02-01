from tools.config_reader import ConfigReader
from tools.experiment import Experiment
import os


class TestConfigReader:

    def test_basic_init(self, tmpdir):
        config_location = os.path.join(os.getcwd(), "tests/basic_test_config.json")
        config = ConfigReader.config_from_file(config_location)

        assert config.brain.number_neurons == 2
        Experiment(configuration=config,
                   result_path=tmpdir,
                   from_checkpoint=None,
                   processing_framework="dask")
