from tools.experiment import Experiment
from attr import evolve
import os
from tools.helper import config_from_file, sample_from_design_space, config_from_dict
import json


def mock_eval(*nargs, **kwargs):
    return [1.0]


class TestExperiment:

    def test_run(self, tmpdir, config):
        experiment_dask = Experiment(configuration=config,
                                     result_path=tmpdir.mkdir("dask"),
                                     from_checkpoint=None,
                                     processing_framework="dask")
        experiment_dask.run()

        # update the expected results when it changed intentionally
        # when you do, don't forget to repeat an experiment you know will yield good results to make
        # sure nothing broke when you changed the underlying algorithm

        # note; this value depends on the NumPy version used, i.e. with or without Intel MKL
        accepted_results = [
            -103.4065390603272,  # standard NumPy
            -102.16727461334207  # NumPy + MKL
        ]
        assert experiment_dask.result_handler.result_log.chapters["fitness"][-1]["max"] in accepted_results

        experiment_mp = Experiment(configuration=config,
                                   result_path=tmpdir.mkdir("mp"),
                                   from_checkpoint=None,
                                   processing_framework="mp")
        experiment_mp.run()

        experiment_sequential = Experiment(configuration=config,
                                           result_path=tmpdir.mkdir("sequential"),
                                           from_checkpoint=None,
                                           processing_framework="sequential")
        experiment_sequential.run()

        assert (experiment_dask.result_handler.result_log.chapters["fitness"][-1]["max"] ==
                experiment_mp.result_handler.result_log.chapters["fitness"][-1]["max"])

        assert (experiment_dask.result_handler.result_log.chapters["fitness"][-1]["max"] ==
                experiment_sequential.result_handler.result_log.chapters["fitness"][-1]["max"])

    def test_run_atari_setup(self, tmpdir, mocker, config):
        config = evolve(config, environment='Qbert-ram-v0')
        Experiment(configuration=config,
                   result_path=tmpdir,
                   from_checkpoint=None,
                   processing_framework="dask")

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
                       from_checkpoint=None,
                       processing_framework="dask")
