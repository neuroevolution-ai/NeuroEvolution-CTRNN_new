from gym import wrappers
import time

from tools.helper import set_random_seeds, output_to_action
from tools.configurations import IEpisodeRunnerCfg, IBrainCfg
from tools.dask_handler import get_current_worker
from tools.env_handler import EnvHandler


class IEpisodeRunner:
    def __init__(self, config: IEpisodeRunnerCfg, brain_conf: IBrainCfg, brain_class, input_space, output_space,
                 env_template):
        self.config = config
        self.brain_conf = brain_conf
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

    def eval_fitness(self, individual, seed, *args, **kwargs):
        pass


class TrainEpisodeRunner(IEpisodeRunner):
    def __init__(self, config: IEpisodeRunnerCfg, brain_conf: IBrainCfg, brain_class, input_space, output_space,
                 env_template):
        super().__init__(config, brain_conf, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed, *args, **kwargs):
        env = self._get_env()
        set_random_seeds(seed, env)
        fitness_total = 0

        number_fitness_runs = self.config.number_fitness_runs

        for i in range(number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_conf)

            set_random_seeds(i, env)
            ob = env.reset()

            done = False

            while not done:
                brain_output = brain.step(ob)

                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)

                fitness_current += rew

            fitness_total += fitness_current

        compressed_behavior = None
        if hasattr(env, 'get_compressed_behavior'):
            # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
            if callable(env.get_compressed_behavior):
                compressed_behavior = env.get_compressed_behavior()

        return fitness_total / self.config.number_fitness_runs, compressed_behavior


class VisualizeEpisodeRunner(IEpisodeRunner):
    def __init__(self, config: IEpisodeRunnerCfg, brain_conf: IBrainCfg, brain_class, input_space, output_space,
                 env_template):
        super().__init__(config, brain_conf, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed, render=False, record=None, record_force=False, brain_vis_handler=None,
                     neuron_vis=False, slow_down=0, test=None):
        env = self._get_env(record, record_force)
        set_random_seeds(seed, env)
        fitness_total = 0

        number_fitness_runs = self.config.number_fitness_runs

        for i in range(number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual,
                                     self.brain_conf)

            if render:
                env.render()

            ob = env.reset()

            if neuron_vis:
                brain_vis = brain_vis_handler.launch_new_visualization(brain)
            else:
                brain_vis = None

            done = False

            while not done:
                brain_output = brain.step(ob)

                if brain_vis:
                    brain_vis.process_update(in_values=ob, out_values=action)

                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)

                if slow_down:
                    time.sleep(slow_down / 1000.0)

                fitness_current += rew

                if render:
                    env.render()

            fitness_total += fitness_current

        compressed_behavior = None
        if hasattr(env, 'get_compressed_behavior'):
            # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
            if callable(env.get_compressed_behavior):
                compressed_behavior = env.get_compressed_beIhavior()

        return fitness_total / self.config.number_fitness_runs, compressed_behavior
