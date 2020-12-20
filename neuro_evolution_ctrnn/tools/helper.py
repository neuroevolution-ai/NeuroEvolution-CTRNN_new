import random
import numpy as np
import json
import copy
import pickle
import os
import logging
import torch
from typing import Type
from bz2 import compress
import gym

from tools.configurations import (ExperimentCfg, IOptimizerCfg, OptimizerCmaEsCfg, OptimizerMuLambdaCfg,
                                  EpisodeRunnerCfg, ContinuousTimeRNNCfg, FeedForwardCfg,
                                  LSTMCfg, IBrainCfg, NoveltyCfg, ReacherMemoryEnvAttributesCfg,
                                  ConcatenatedBrainLSTMCfg, CnnCtrnnCfg, ConvolutionalNNCfg, AtariEnvAttributesCfg)


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
    logging.info("writing checkpoint " + filename)
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
