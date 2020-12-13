import json
import copy
from typing import Type
import logging

from tools.configurations import ExperimentCfg, registered_types
from tools.helper import walk_dict_dept_first


def _replace_with_type(key, value, depth, is_leaf):
    if isinstance(value, dict):
        if 'type' in value:
            try:
                found_type: Type = registered_types[value["type"]]
                value = found_type(**(value))
            except KeyError:
                logging.error('key "' + value["type"] + '" not found in tools.configurations.registered_types.')
                raise


class ConfigReader:
    """reads a configuration from a json string and returns a valid config object"""

    def __init__(self):
        pass

    @classmethod
    def config_from_file(cls, file_path: str):
        with open(file_path, "r") as read_file:
            config_dict = json.load(read_file)
        return cls.config_from_dict(config_dict)

    @classmethod
    def config_from_dict(cls, config_dict: dict):
        # store the serializable version of the config so it can be later be serialized again
        walk_dict_dept_first(config_dict, _replace_with_type)
        config_dict["raw_dict"] = copy.deepcopy(config_dict)
        return ExperimentCfg(**config_dict)
