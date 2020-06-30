import gym
import pybullet_envs
from gym import wrappers
from brains.lstm import LSTMNumPy
from brains.i_brain import IBrain

from deap import base
from deap import creator
import pickle
import os
import numpy as np
import json
import argparse

from tools.helper import config_from_dict, config_from_file

def eval_fitness(env_id, config, brain_conf, individual, seed, dir):

    env = gym.make(env_id)
    env = wrappers.Monitor(env, os.path.join(dir, "video"), force=True)
    env.seed(seed)
    fitness_current = 0
    observation_mask = config.observation_mask

    input_space = env.observation_space
    output_space = env.action_space

    # TODO is this still necessary (increasing the seed for each worker?
    # if configuration_data["random_seed_for_environment"] is not -1:
    #     env.seed(configuration_data["random_seed_for_environment"] + i)

    ob = env.reset()
    env.render()
    env._max_episode_steps = config.observation_frames + config.memory_frames + config.action_frames

    # Create brain
    brain = LSTMNumPy(input_space, output_space, individual, brain_conf)
    output_size = IBrain._size_from_space(output_space)
    t = 0
    done = False
    while not done:

        # Perform step of the brain simulation
        action = brain.step(ob)

        if t <= config.observation_frames + config.memory_frames:
            action = np.zeros(output_size)

        # Perform step of the environment simulation
        ob, rew, done, info = env.step(action)

        if t >= config.observation_frames:
            for index in observation_mask:
                ob[index] = 0.0

        if t >= config.observation_frames + config.memory_frames:
            fitness_current += rew

        t += 1

        env.render()

    return fitness_current


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Visualize Memory Experiment")

    parser.add_argument('--dir', metavar='dir', type=str,
                        help='Path to the simulation result',
                        default=None)

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args(args)


args = parse_args()


if args.config:
    config = config_from_file(args.config)
else:
    config = {
      "environment": "AntBulletEnv-v0",
      "random_seed": 0,
      "number_generations": 2500,
      "optimizer": {
        "type": "CMA_ES",
        "population_size": 200,
        "sigma": 1.0,
        "checkpoint_frequency": 10
      },
      "brain": {
        "type": "LSTM_NumPy",
        "normalize_input": False,
        "normalize_input_target": 2,
        "lstm_num_layers": 5,
        "use_biases": True
      },
      "episode_runner": {
        "type": "Memory",
        "number_fitness_runs": 50,
        "reuse_env": True,
        "observation_frames": 20,
        "memory_frames": 20,
        "action_frames": 50,
        "observation_mask": [4, 5, 8, 9, 10]
      }
    }

    config = config_from_dict(config)


if args.checkpoint:
    with open(os.path.join(args.checkpoint), "rb") as read_file_checkpoint:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        checkpoint = pickle.load(read_file_checkpoint)["halloffame"][0]

        eval_fitness(config.environment, config.episode_runner, config.brain, checkpoint, 0, os.path.dirname(args.checkpoint))


elif args.dir:
    with open(os.path.join(args.dir, 'Log.json'), 'r') as read_file_log:
        log = json.load(read_file_log)

    with open(os.path.join(args.dir, 'Configuration.json'), "r") as read_file:
        conf = json.load(read_file)




