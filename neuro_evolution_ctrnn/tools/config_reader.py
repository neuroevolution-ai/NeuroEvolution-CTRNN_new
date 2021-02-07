import json
import copy
from typing import Type, Dict
import logging
import random

registered_types: Dict = {}
registered_keys: Dict = {}
registered_brains: Dict = {}


def register(name: str, as_key: bool = False, brain_class: Type = None):
    """
    This decorator registers a class, so that it can later be found by the config_reader with a specific string.
    To map between the class and the string, it uses the content of the type-value of the json node.
    If no type-attribute exists for that node, it uses the key of the node instead.

    We explicitly separate between a mapping by key and by type-string, and thus we manage two separate lists for it.
    Technically this separate isn't needed and it adds clutter, but I think explicit behavior here will help in
    the long run.

    :param name: the name and which the class shall be registered
    :param as_key: if true, the node's key will be used for mapping. If false that type-value will be used
    :param brain_class: this class is the config file for a Brain, the linked brain-class must be passed as a paramter
    :return: a wrapper of the class
    """

    # registering magic
    # See https://realpython.com/primer-on-python-decorators/#decorators-with-arguments
    def _register(type_class: type):
        if brain_class:
            assert name not in registered_brains, 'brain "' + str(name) + '" was already registered'
            registered_brains[name] = brain_class
        if as_key:
            assert name not in registered_keys, 'key "' + str(name) + '" was already registered'
            registered_keys[name] = type_class
        else:
            assert name not in registered_types, 'type "' + str(name) + '" was already registered'
            registered_types[name] = type_class
        return type_class

    return _register


class ConfigReader:
    """Reads a configuration from a JSON string and returns a valid config object."""

    @classmethod
    def _replace_dicts_with_types(cls, node, depth=0):
        for key, item in node.items():
            if isinstance(item, dict):
                # tree traversal needs to be depth first to avoid TypeError in parents nodes
                cls._replace_dicts_with_types(item, depth + 1)
                if "type" in item:
                    found_type: Type = registered_types[item["type"]]
                elif key in registered_keys:
                    found_type: Type = registered_keys[key]
                else:
                    # this is a not a dict that needs to be transformed to an object
                    continue
                try:
                    node[key] = found_type(**item)
                except TypeError:
                    logging.error("Couldn't turn dictionary into type. "
                                  "See tools.configurations for a list of optional and required attributes for "
                                  "each type.")
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

        if config_dict["random_seed"] < 0:
            seed = random.randint(1, 10000)
            logging.info("setting random seed to " + str(seed))
            logging.info(
                "if you want to ignore random states, set random_seed to 0. If you want to use a specific seed, "
                "set random_seed to a positive integer.")
            config_dict["random_seed"] = seed
        cls._replace_dicts_with_types(config_dict)

        config_dict["raw_dict"] = raw_dict
        return config_dict
