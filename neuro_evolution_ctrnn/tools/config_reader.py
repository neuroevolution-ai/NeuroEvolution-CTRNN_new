import json
import copy
from typing import Type
import logging

from tools.configurations import ExperimentCfg, registered_types


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
                    try:
                        found_type: Type = registered_types[item["type"]]
                        node[key] = found_type(**item)
                    except KeyError:
                        logging.error('key "' + item["type"] + '" not found in tools.configurations.registered_types.')
                        raise
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
        cls._replace_dicts_with_types(config_dict)
        # store the serializable version of the config so it can be later be serialized again during result handling
        config_dict["raw_dict"] = copy.deepcopy(config_dict)
        return ExperimentCfg(**config_dict)
