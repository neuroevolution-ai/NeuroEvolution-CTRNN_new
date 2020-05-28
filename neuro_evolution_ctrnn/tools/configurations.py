# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import List


class ConfigBase:
    # slots saves a little bit of memory, but more importantly
    # it makes sure no unexpected attributes can appear on the object, which prevents typos among other things
    __slots__: List[str] = []

    def __init__(self, **attr):
        for key in attr:
            setattr(self, key, attr[key])
        for key in dir(self):
            if not hasattr(self, key):
                raise RuntimeError("key missing: " + key)


class EpisodeRunnerCfg(ConfigBase):
    __slots__ = ['number_fitness_runs', 'keep_env_seed_fixed_during_generation']
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool


class ContinuousTimeRNNCfg(ConfigBase):
    __slots__ = ['optimize_y0', 'delta_t', 'optimize_state_boundaries',
                 'set_principle_diagonal_elements_of_W_negative', 'number_neurons',
                 'normalize_input', 'clipping_range_min', 'clipping_range_max', 'normalize_input_target', 'v_mask',
                 'v_mask_param', 'w_mask', 'w_mask_param', 't_mask', 't_mask_param', 'parameter_perturbations']
    optimize_y0: bool
    delta_t: float
    optimize_state_boundaries: str
    set_principle_diagonal_elements_of_W_negative: bool
    number_neurons: int
    normalize_input: bool
    normalize_input_target: float
    clipping_range_min: float
    clipping_range_max: float
    v_mask: str
    v_mask_param: float
    w_mask: str
    w_mask_param: float
    t_mask: str
    t_mask_param: float
    parameter_perturbations: float


class TrainerCmaEsCfg(ConfigBase):
    __slots__ = ['population_size', 'sigma', 'checkpoint_frequency']
    population_size: int
    sigma: float
    checkpoint_frequency: int


class ExperimentCfg(ConfigBase):
    __slots__ = ['neural_network_type', 'environment', 'random_seed',
                 'trainer_type', 'number_generations', 'brain', 'episode_runner', 'trainer', 'raw_dict']

    neural_network_type: str
    environment: str
    random_seed: int
    trainer_type: str
    number_generations: int
    brain: ContinuousTimeRNNCfg
    episode_runner: EpisodeRunnerCfg
    trainer: TrainerCmaEsCfg
    raw_dict: dict
