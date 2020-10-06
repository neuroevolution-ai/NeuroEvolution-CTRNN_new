from gym import wrappers
import time
import logging
from bz2 import BZ2Compressor
import numpy as np

from tools.helper import set_random_seeds, output_to_action
from tools.configurations import EpisodeRunnerCfg, IBrainCfg
from tools.dask_handler import get_current_worker
from tools.env_handler import EnvHandler
from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.CNN_CTRNN import CnnCtrnn


class EpisodeRunner:
    def __init__(self, config: EpisodeRunnerCfg, brain_config: IBrainCfg, brain_class, input_space, output_space,
                 env_template):
        self.config = config
        self.brain_config = brain_config
        self.brain_class = brain_class
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id
        self.env_handler = EnvHandler(self.config)

    def _get_env(self, record=False, record_force=False):
        if self.config.reuse_env:
            try:
                env = get_current_worker().env
            except:
                if hasattr(self, "env"):
                    env = self.env
                else:
                    self.env = env = self.env_handler.make_env(self.env_id)
        else:
            env = self.env_handler.make_env(self.env_id)

        if record:
            env = wrappers.Monitor(env, record, force=record_force)

        return env

    def eval_fitness(self, individual, seed, render=False, record=None, record_force=False, brain_vis_handler=None,
                     neuron_vis=False, slow_down=0, rounds=None):
        env = self._get_env(record, record_force)
        set_random_seeds(seed, env)
        fitness_total = 0
        steps_total = 0
        number_of_rounds = self.config.number_fitness_runs if rounds is None else rounds
        brain_state_history = []
        for i in range(number_of_rounds):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_config)
            ob = env.reset()
            done = False
            t = 0

            if render:
                env.render()

            if neuron_vis:
                brain_vis = brain_vis_handler.launch_new_visualization(brain=brain, brain_config=self.brain_config,
                                                                       env_id=self.env_id, width=1600 , height=900,
                                                                       color_clipping_range=(255, 2.5, 2.5))
            else:
                brain_vis = None

            while not done:
                brain_output = brain.step(ob)
                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)
                t += 1
                fitness_current += rew

                if brain_vis:
                    brain_vis.process_update(in_values=ob, out_values=brain_output)
                if slow_down:
                    time.sleep(slow_down / 1000.0)
                if render:
                    env.render()
                if self.config.novelty:
                    if self.config.novelty.behavior_source == 'brain':
                        if isinstance(brain, ContinuousTimeRNN):
                            brain_state_history.append(np.tanh(brain.y))
                        elif isinstance(brain, CnnCtrnn):
                            brain_state_history.append(np.tanh(brain.ctrnn.y))
                        else:
                            logging.error('behavior_source == "brain" not yet supported for this kind of brain')

            if render:
                logging.info("steps: " + str(t) + " \tfitness: " + str(fitness_current))

            fitness_total += fitness_current
            steps_total += t

        compressed_behavior = None
        if hasattr(env, 'get_compressed_behavior'):
            # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
            if callable(env.get_compressed_behavior):
                compressed_behavior = env.get_compressed_behavior()

        if self.config.novelty:
            if self.config.novelty.behavior_source == 'brain':
                # todo: remove code duplication. This code is also in BehaviorWrapper
                compressor = BZ2Compressor(2)
                compressed_behavior = b''
                for i in range(self.config.novelty.behavioral_max_length):
                    aggregate = np.zeros(len(brain_state_history[0]), dtype=np.float32)
                    for j in range(self.config.novelty.behavioral_interval):
                        if len(brain_state_history) > j + i * self.config.novelty.behavioral_interval:
                            state = brain_state_history[j + i * self.config.novelty.behavioral_interval]
                            aggregate += state / self.config.novelty.behavioral_interval
                        else:
                            break
                    compressed_behavior += compressor.compress(aggregate.astype(np.float16).tobytes())
                compressed_behavior += compressor.flush()

        return fitness_total / number_of_rounds, compressed_behavior, steps_total
