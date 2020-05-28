import random
import numpy as np
import json
import copy
import pickle
import os
import logging
from tools.configurations import ExperimentCfg, TrainerCmaEsCfg, EpisodeRunnerCfg, ContinuousTimeRNNCfg


def walk_dict(node, callback_node, depth=0):
    for key, item in node.items():
        if isinstance(item, dict):
            callback_node(key, item, depth, False)
            walk_dict(item, callback_node, depth + 1)
        else:
            callback_node(key, item, depth, True)


def sample_from_design_space(node):
    result = {}
    for key in node:
        val = node[key]
        if isinstance(val, list):
            if val:
                val = random.sample(val, 1)[0]
            else:
                # empty lists become None
                val = None

        if isinstance(val, dict):
            result[key] = sample_from_design_space(val)
        else:
            result[key] = val
    return result


def config_from_file(json_path):
    # Load configuration file
    with open(json_path, "r") as read_file:
        config_dict = json.load(read_file)

    # store the serializable version of the config so it can be later be serialized again
    config_dict["raw_dict"] = copy.deepcopy(config_dict)

    if config_dict["neural_network_type"] == 'CTRNN':
        brain_cfg_class = ContinuousTimeRNNCfg
    else:
        raise RuntimeError("unknown neural_network_type: " + str(config_dict["neural_network_type"]))

    if config_dict["trainer_type"] == 'CMA_ES':
        trainer_cfg_class = TrainerCmaEsCfg
    else:
        raise RuntimeError("unknown trainer_type: " + str(config_dict["neural_network_type"]))

    if not config_dict["random_seed"]:
        config_dict["random_seed"] = random.getstate()
        print("setting random seed to: " + str(config_dict["random_seed"]))

    # turn json into nested class so python's type-hinting can do its magic
    config_dict["episode_runner"] = EpisodeRunnerCfg(**(config_dict["episode_runner"]))
    config_dict["trainer"] = trainer_cfg_class(**(config_dict["trainer"]))
    config_dict["brain"] = brain_cfg_class(**(config_dict["brain"]))
    return ExperimentCfg(**config_dict)


def write_checkpoint(base_path, frequency, data):
    if not frequency:
        return
    if data["generation"] % frequency != 0:
        return

    filename = os.path.join(base_path, "checkpoint_" + str(data["generation"]) + ".pkl")
    logging.info("writing checkpoint " + filename)
    with open(filename, "wb") as cp_file:
        pickle.dump(data, cp_file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)


def get_checkpoint(checkpoint):
    with open(checkpoint, "rb") as cp_file:
        cp = pickle.load(cp_file, fix_imports=False)
    return cp


def set_random_seeds(seed, env):
    if type(seed) != int:
        # env.seed only accepts native integer and not np.int32/64
        # so we need to extract the int before passing it to env.seed()
        seed = seed.item()

    random.seed(seed)
    np.random.seed(seed)
    if env:
        env.seed(seed)
        env.action_space.seed(seed)
