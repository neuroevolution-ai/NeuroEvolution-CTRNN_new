import json
import copy
from typing import Type
import logging
import random

from tools.configurations import ExperimentCfg, registered_types, NoveltyCfg, registered_keys


class ConfigReader:
    """reads a configuration from a json string and returns a valid config object"""

    def __init__(self):
        pass

    @classmethod
    def _replace_dicts_with_types(cls, node, depth=0):
        for key, item in node.items():
            if isinstance(item, dict):
                # tree traversal needs to be depth first to avoid TypeError in parents nodes
                cls._replace_dicts_with_types(item, depth + 1)
                if 'type' in item:
                    found_type: Type = registered_types[item["type"]]
                elif key in registered_keys:
                    found_type: Type = registered_keys[key]
                else:
                    # this is a not a dict that needs to be transformed to an object
                    continue
                try:
                    node[key] = found_type(**item)
                except TypeError:
                    logging.error('Couldn\'t turn dictionary into type. '
                                  'See tools.configurations for a list of optional and required attributes for '
                                  'each type.')
                    raise

    @classmethod
    def config_from_file(cls, file_path: str):
        with open(file_path, "r") as read_file:
            config_dict = json.load(read_file)
        return cls.config_from_dict(config_dict)

    @classmethod
    def config_from_dict(cls, config_dict: dict):

        # store the serializable version of the config so it can be later be serialized again during result handling
        raw_dict = copy.deepcopy(config_dict)

        if "novelty" in config_dict:
            # special handling for novelty, because it's values are needed in two different classes
            config_dict["optimizer"]["novelty"] = config_dict["novelty"]
            config_dict["episode_runner"]["novelty"] = config_dict["novelty"]
            del config_dict["novelty"]

        if config_dict['random_seed'] < 0:
            seed = random.randint(1, 10000)
            logging.info("setting random seed to " + str(seed))
            logging.info(
                "if you want to ignore random states, set random_seed to 0. If you want to use a specific seed, "
                "set random_seed to a positive integer.")
            config_dict['random_seed'] = seed
        cls._replace_dicts_with_types(config_dict)

        config_dict["raw_dict"] = raw_dict
        return ExperimentCfg(**config_dict)
