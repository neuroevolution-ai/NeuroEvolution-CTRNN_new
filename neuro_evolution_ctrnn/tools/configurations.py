# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import Iterable
from collections import namedtuple


class EpisodeRunnerCfg:
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool

    def __init__(self, **attr):
        self.__dict__.update(attr)


class ContinuousTimeRNNCfg:
    __slots__ = ['optimize_y0', 'delta_t', 'optimize_state_boundaries',
                 'set_principle_diagonal_elements_of_W_negative', 'number_neurons',
                 'normalize_input', 'clipping_range_min', 'clipping_range_max', 'normalize_input_target']
    optimize_y0: bool
    delta_t: float
    optimize_state_boundaries: str
    set_principle_diagonal_elements_of_W_negative: bool
    number_neurons: int
    normalize_input: bool
    normalize_input_target: float
    clipping_range_min: Iterable[float]
    clipping_range_max: Iterable[float]

    def __init__(self, **attr):
        for key in attr:
            setattr(self, key, attr[key])
        for key in dir(self):
            if not hasattr(self, key):
                raise RuntimeError("key missing: " + key)

class TrainerCmaEsCfg:
    population_size: int
    sigma: float

    def __init__(self, **attr):
        self.__dict__ = attr


class ExperimentCfg:
    neural_network_type: str
    environment: str
    random_seed: int
    trainer_type: str
    number_generations: int
    brain: ContinuousTimeRNNCfg
    episode_runner: EpisodeRunnerCfg
    trainer: TrainerCmaEsCfg
    _raw_dict: dict

    def __init__(self, **attr):
        self.__dict__.update(attr)
