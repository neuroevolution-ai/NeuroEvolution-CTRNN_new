from tools.experiment import Experiment
from tools.helper import config_from_file
import os


def mock_eval(a1, a2):
    return [1.0]


class TestExperiment:

    def test_run(self, tmpdir, mocker):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        config_location = os.path.join(current_directory, "basic_test_config.json")
        experiment = Experiment(configuration=config_from_file(config_location),
                                result_path=tmpdir,
                                from_checkpoint=None)
        experiment.run()

        # update the expected results when it changed intentionally
        # when you do, don't forget to repeat an experiment you know will yield good results to make
        # sure nothing broke when you changed the underlying algorithm

        # note; this value depends on the machine
        assert experiment.result_handler.result_log[-1]["max"] == -99.11361202453168

    def test_run_atari(self, tmpdir, mocker):
        mocker.patch('tools.episode_runner.EpisodeRunner.eval_fitness', side_effect=mock_eval)
        current_directory = os.path.dirname(os.path.realpath(__file__))
        config_location = os.path.join(current_directory, "atari_test_config.json")
        experiment = Experiment(configuration=config_from_file(config_location),
                                result_path=tmpdir,
                                from_checkpoint=None)
        experiment.run()
