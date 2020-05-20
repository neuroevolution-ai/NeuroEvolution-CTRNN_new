from tools.experiment import Experiment
from tools.helper import config_from_file


class TestExperiment:

    def test_run(self, tmpdir):
        config_location = "tests/basic_test_config.json"
        experiment = Experiment(configuration=config_from_file(config_location),
                                result_path=tmpdir,
                                from_checkpoint=None)
        experiment.run()

        # update the expected results when it changed intentionally
        # when you do, don't forget to repeat an experiment you know will yield good results to make
        # sure nothing broke when you changed the underlying algorithm
        assert experiment.result_handler.result_log[-1]["max"] == -99.11361202453168
