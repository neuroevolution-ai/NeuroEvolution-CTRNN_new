import random
import numpy as np
import pickle
import os
import logging
import torch
from typing import get_args
from bz2 import compress
import gym
from tools.configurations import registered_types
from brains import IBrain


def brain_class_from_config_type(config_type_string: str):
    brain_class = None
    target_brain_cfg = registered_types[config_type_string]
    all_brain_classes = []

    def walk_subclass(brain_class_parent):
        for brain_class_child in brain_class_parent.__subclasses__():
            all_brain_classes.append(brain_class_child)
            walk_subclass(brain_class_child)

    walk_subclass(IBrain)
    for brain_class_candidate in all_brain_classes:
        for bases in brain_class_candidate.__orig_bases__:  # iterate over all base-classes of this candidate
            for type_args in get_args(bases):  # iterate over all generic types used with this base-class
                if target_brain_cfg == type_args:
                    if not brain_class:
                        brain_class = brain_class_candidate
                    elif brain_class != brain_class_candidate:
                        raise RuntimeError("Multiple brain classes are registered for this configuration-class.")
    if not brain_class:
        # this is either a typo in the json, or the target brain is not a proper subclass of IBrain, or the
        # config-class is not linked to the brain class as a generic of IBrain.
        raise RuntimeError("no brain_class found for type: " + str(config_type_string))
    return brain_class


def output_to_action(output, action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return np.argmax(output)
    elif isinstance(action_space, gym.spaces.tuple.Tuple):
        index = 0
        action_list = []
        for space in action_space:
            sub_output = output[index:index + space.n]
            action_list.append(output_to_action(sub_output, space))
            index += space.n
        return action_list
    else:
        # for output type box, the data is already in the right format
        return output


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


def write_checkpoint(base_path, frequency, data):
    if not frequency:
        return
    if data["generation"] % frequency != 0:
        return

    filename = os.path.join(base_path, "checkpoint_" + str(data["generation"]) + ".pkl")
    logging.info("Writing checkpoint " + filename)
    with open(filename, "wb") as cp_file:
        pickle.dump(data, cp_file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)


def get_checkpoint(checkpoint):
    with open(checkpoint, "rb") as cp_file:
        cp = pickle.load(cp_file, fix_imports=False)
    return cp


def set_random_seeds(seed, env):
    if not seed:
        return

    if type(seed) != int:
        # env.seed only accepts native integer and not np.int32/64
        # so we need to extract the int before passing it to env.seed()
        seed = seed.item()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env:
        env.seed(seed)
        env.action_space.seed(seed)


def normalized_compression_distance(a, b, a_len=None, b_len=None):
    if not a_len:
        a_len = len(compress(bytearray(a), 2))
    if not b_len:
        b_len = len(compress(bytearray(b), 2))
    ab_len = len(compress(bytearray(np.concatenate((a, b))), 2))
    return (ab_len - min(a_len, b_len)) / max(a_len, b_len)


def equal_elements_distance(a, b, a_len=None, b_len=None):
    count = 0
    for x, y in zip(a, b):
        if x == y:
            count += 1
    # if the new individual is longer, count every new element as novelty
    return len(a) - count


def euklidian_distance(a, b, a_len=None, b_len=None):
    b = np.array(b).flatten()
    a = np.array(a).flatten()
    x = min(len(a), len(b))
    b = b[:x]
    a = a[:x]
    return np.linalg.norm(a - b)
