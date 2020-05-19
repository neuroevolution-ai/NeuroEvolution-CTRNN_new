
import random
import numpy as np


def walk_dict(node, callback_node, depth=0):
    for key, item in node.items():
        if isinstance(item,dict):
            callback_node(key, item, depth, False)
            walk_dict(item, callback_node, depth + 1)
        else:
            callback_node(key, item, depth, True)


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


