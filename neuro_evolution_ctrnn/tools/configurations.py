# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import List
import attr


@attr.s(slots=True)
class EpisodeRunnerCfg:
    number_fitness_runs: int = attr.ib()
    keep_env_seed_fixed_during_generation: bool = attr.ib()


@attr.s(slots=True)
class ContinuousTimeRNNCfg:
    type: str = attr.ib()
    optimize_y0: bool = attr.ib()
    delta_t: float = attr.ib()
    optimize_state_boundaries: str = attr.ib()
    set_principle_diagonal_elements_of_W_negative: bool = attr.ib()
    number_neurons: int = attr.ib()
    normalize_input: bool = attr.ib()
    normalize_input_target: float = attr.ib()
    clipping_range_min: float = attr.ib()
    clipping_range_max: float = attr.ib()
    v_mask: str = attr.ib()
    v_mask_param: float = attr.ib()
    w_mask: str = attr.ib()
    w_mask_param: float = attr.ib()
    t_mask: str = attr.ib()
    t_mask_param: float = attr.ib()
    parameter_perturbations: float = attr.ib()


@attr.s(slots=True)
class OptimizerCmaEsCfg:
    type: str = attr.ib()
    population_size: int = attr.ib()
    sigma: float = attr.ib()
    checkpoint_frequency: int = attr.ib()


@attr.s(slots=True)
class ExperimentCfg:
    environment: str = attr.ib()
    random_seed: int = attr.ib()
    number_generations: int = attr.ib()
    brain: ContinuousTimeRNNCfg = attr.ib()
    episode_runner: EpisodeRunnerCfg = attr.ib()
    optimizer: OptimizerCmaEsCfg = attr.ib()
    raw_dict: dict = attr.ib()
