# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import List
import attr


@attr.s(slots=True, auto_attribs=True, frozen=True)
class EpisodeRunnerCfg:
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool
    reuse_env: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ContinuousTimeRNNCfg:
    type: str
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


@attr.s(slots=True, auto_attribs=True, frozen=True)
class OptimizerCmaEsCfg:
    type: str
    population_size: int
    sigma: float
    checkpoint_frequency: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ExperimentCfg:
    environment: str
    random_seed: int
    number_generations: int
    brain: ContinuousTimeRNNCfg
    episode_runner: EpisodeRunnerCfg
    optimizer: OptimizerCmaEsCfg
    raw_dict: dict
