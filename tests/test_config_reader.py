from tools.config_reader import ConfigReader
from tools.experiment import Experiment
import os


class TestConfigReader:

    def test_basic_init(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        config_location = os.path.join(current_directory, "basic_test_config.json")
        config = ConfigReader.config_from_file(config_location)
        assert config.brain.number_neurons == 2

    def test_cnn_init_exp(self, tmpdir):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        config_location = os.path.join(current_directory, "../configurations/cnn_ctrnn.json")

        config = ConfigReader.config_from_file(config_location)
        Experiment(configuration=config,
                   result_path=tmpdir,
                   from_checkpoint=None,
                   processing_framework="dask")

