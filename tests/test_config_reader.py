from tools.config_reader import ConfigReader
import os


class TestConfigReader:

    def test_basic_init(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        config_location = os.path.join(current_directory, "basic_test_config.json")
        config = ConfigReader.config_from_file(config_location)
