from tools.experiment import Experiment


def mock_eval(a1, a2):
    return [1.0]


class TestExperiment:

    def test_run(self, tmpdir, config):
        experiment = Experiment(configuration=config,
                                result_path=tmpdir,
                                from_checkpoint=None)
        experiment.run()

        # update the expected results when it changed intentionally
        # when you do, don't forget to repeat an experiment you know will yield good results to make
        # sure nothing broke when you changed the underlying algorithm

        # note; this value depends on the machine
        accepted_results = [-99.11361202453168,  # result on bjoern's notebook
                            -98.95448135483025,  # result on bjoern's desktop
                            ]
        assert experiment.result_handler.result_log[-1]["max"] in accepted_results

    def test_run_atari(self, tmpdir, mocker, config):
        mocker.patch('tools.episode_runner.EpisodeRunner.eval_fitness', side_effect=mock_eval)
        config.environment = 'Qbert-ram-v0'
        config.brain.normalize_input = True
        experiment = Experiment(configuration=config,
                                result_path=tmpdir,
                                from_checkpoint=None)
        experiment.run()
