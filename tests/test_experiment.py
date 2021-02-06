from tools.experiment import Experiment
from attr import evolve
import os
from tools.helper import sample_from_design_space
from tools.config_reader import ConfigReader
from tools.configurations import ProcGenEnvAttributesCfg
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

        # Note that these versions depend on whether one uses a virtualenv created with the standard Python virtualenv
        # tool package or with Anaconda. Also this value changes when standard NumPy or NumPy + MKL is used.
        # I don't think this list is exhausted, expand or change accordingly if necessary.
        # Also the GitHub Action Runner has a different result than the standard Python virtualenv for some reason
        accepted_results = [
            -103.4065390603272,  # Python 3.8 + conda 4.9.2 + standard NumPy
            -102.16727461334207,  # Python 3.8 + conda 4.9.2 + NumPy + MKL
            -98.80537742389652,  # Python 3.8 + Python virtualenv + standard NumPy
            -98.9895922078135  # Python 3.8 in GitHub Action Runner
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

    def test_run_atari_setup(self, tmpdir, config):
        config = evolve(config, environment='Qbert-ram-v0')
        Experiment(configuration=config,
                   result_path=tmpdir,
                   from_checkpoint=None,
                   processing_framework="mp")

    def test_run_procgen(self, tmpdir, config, mu_lambda_es_config):
        brainCfg = evolve(config.brain,
                          v_mask='logarithmic', v_mask_param=128,
                          w_mask='logarithmic', w_mask_param=128,
                          t_mask='logarithmic', t_mask_param=128)

        # need custom eprunner-config, because procgen experiments need explicit config
        episode_runner_cfg = evolve(config.episode_runner,
                                    environment_attributes=ProcGenEnvAttributesCfg(type="ProcGenAttr"))

        # need to use mu_lambda_es_config, because genome is too large for CMA_ES
        config = evolve(config, environment='procgen:procgen-heist-v0', brain=brainCfg, optimizer=mu_lambda_es_config,
                        episode_runner=episode_runner_cfg)

        exp = Experiment(configuration=config,
                         result_path=tmpdir,
                         from_checkpoint=None,
                         processing_framework="mp")
        exp.run()

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
                c = ConfigReader.config_from_dict(sample_from_design_space(design_space))
            else:
                c = ConfigReader.config_from_file(path)

            if c.environment in ['ReacherMemory-v0']:
                print("skipping, because Mujoco")
                continue

            Experiment(configuration=c,
                       result_path=tmpdir,
                       from_checkpoint=None,
                       processing_framework="mp")
