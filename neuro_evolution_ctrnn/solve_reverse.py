"""In this experiment instead of training a CTRNN we construct it directly.
The goal is to demonstrate, that CTRNN can solve algorithmical problems very efficiently"""

from tools.experiment import Experiment
from brain_visualizer.brain_visualizer import BrainVisualizerHandler

from tools.configurations import ExperimentCfg, ContinuousTimeRNNCfg, StandardEpisodeRunnerCfg
from tools.helper import config_from_file
import os
import threading
from brains.continuous_time_rnn import ContinuousTimeRNN
from attr import s
import numpy as np
from tools.helper import output_to_action

cfg_path = os.path.join('configurations', 'reverse_fixed.json')
cfg_exp = config_from_file(cfg_path)

experiment = Experiment(configuration=cfg_exp,
                        result_path="",
                        from_checkpoint=None)


@s(auto_attribs=True, frozen=True, slots=True)
class BrainParam:
    V: np.ndarray
    W: np.ndarray
    T: np.ndarray
    y0: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray


def param_to_genom(param):
    return np.concatenate(
        [param.V.flatten(),
         param.W.flatten(),
         param.T.flatten(),
         param.y0.flatten(),
         param.clip_min.flatten(),
         param.clip_max.flatten()])


param = BrainParam(
    V=np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]),
    W=np.array([[-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, -1]]),
    T=np.array([[0, 0, 1, 0],
                [0, 0, -1, 1],
                [0, 0, -1, 1],
                [2, 2, 2, -3],
                [1, 0, 0, 0],
                [0, 1, 0, 0]]).flatten(order='F'),
    y0=np.array([]), clip_min=np.array([]), clip_max=np.array([]))

ind = param_to_genom(param)

env = experiment.env_template
for i in range(1):
    brain = ContinuousTimeRNN(input_space=experiment.input_space, output_space=experiment.output_space, individual=ind,
                              config=cfg_exp.brain)
    ob = env.reset()
    env.unwrapped.input_data = [0, 1, 0, 1, 1]
    env.unwrapped.target = env.unwrapped.target_from_input_data(env.unwrapped.input_data)

    env.render()
    done = False
    fitness_current = 0
    while not done:
        brain_output = brain.step(ob)
        action = output_to_action(brain_output, experiment.output_space)
        print("act: " + str(action))
        ob, rew, done, info = env.step(action)
        fitness_current += rew
        if rew < 0:
            print("error")
            env.render()

        env.render()
    print('score: ' + str(fitness_current))

# todo: use brain_vis to visualize this
t = threading.Thread(target=experiment.visualize,
                     args=[[ind], BrainVisualizerHandler(), 2, False,
                           False])
# t.start()
