from tools.experiment import Experiment
from attr import evolve
import os
from tools.helper import config_from_file, sample_from_design_space, config_from_dict
import json


def mock_eval(*nargs, **kwargs):
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
                            -92.24354731262838  # result on Patrick's notebook
                            ]
        assert experiment.result_handler.result_log.chapters["fitness"][-1]["max"] in accepted_results

    def test_run_atari_setup(self, tmpdir, mocker, config):
        config = evolve(config, environment='Qbert-ram-v0')
        Experiment(configuration=config,
                   result_path=tmpdir,
                   from_checkpoint=None)

    def test_init_from_example_configs(self, tmpdir):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        example_conf_path = os.path.join(current_directory, "..", "configurations")
        for conf_name in os.listdir(example_conf_path):
            print("next conf: " + conf_name)
            path = os.path.join(example_conf_path, conf_name)
            if os.path.isdir(path):
                print("skipping, because directory")
                continue
            if conf_name == 'temp.json':
                print("skipping, because temp file")
                continue

            if "design" in conf_name:
                with open(path, "r") as read_file:
                    design_space = json.load(read_file)
                c = config_from_dict(sample_from_design_space(design_space))
            else:
                c = config_from_file(path)

            if c.environment in ['ReacherMemory-v0']:
                print("skipping, because Mujoco")
                continue


            Experiment(configuration=c,
                       result_path=tmpdir,
                       from_checkpoint=None)
