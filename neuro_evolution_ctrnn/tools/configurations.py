# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import Iterable


class EpisodeRunnerCfg:
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool

    def __init__(self, **attr):
        self.__dict__ = attr


class ContinuousTimeRNNCfg:
    optimize_y0: bool
    delta_t: float
    optimize_state_boundaries: str
    set_principle_diagonal_elements_of_W_negative: bool
    number_neurons: int
    clipping_range_min: Iterable[float]
    clipping_range_max: Iterable[float]

    def __init__(self, **attr):
        self.__dict__ = attr


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
        self.__dict__ = attr
