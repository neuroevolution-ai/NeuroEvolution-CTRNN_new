# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import List
import attr
import abc
import typing


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IBrainCfg(abc.ABC):
    type: str
    normalize_input: bool
    normalize_input_target: float
    use_bias: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class NoveltyCfg:
    novelty_weight: float
    distance: str
    novelty_nearest_k: int
    max_recorded_behaviors: int
    recorded_behaviors_per_generation: int
    behavioral_interval: int
    behavioral_max_length: int
    behavior_from_observation: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IEpisodeRunnerCfg(abc.ABC):
    type: str
    number_fitness_runs: int
    reuse_env: bool
    max_steps_per_run: int
    max_steps_penalty: int
    novelty: typing.Optional[NoveltyCfg]


@attr.s(slots=True, auto_attribs=True, frozen=True)
class StandardEpisodeRunnerCfg(IEpisodeRunnerCfg):
    keep_env_seed_fixed_during_generation: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class MemoryExperimentCfg(IEpisodeRunnerCfg):
    observation_frames: int
    memory_frames: int
    action_frames: int
    observation_mask: [int]


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ContinuousTimeRNNCfg(IBrainCfg):
    optimize_y0: bool
    delta_t: float
    optimize_state_boundaries: str
    set_principle_diagonal_elements_of_W_negative: bool
    number_neurons: int
    clipping_range_min: float
    clipping_range_max: float
    v_mask: str
    v_mask_param: float
    w_mask: str
    w_mask_param: float
    t_mask: str
    t_mask_param: float
    parameter_perturbations: float
    neuron_activation: str
    neuron_activation_inplace: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class FeedForwardCfg(IBrainCfg):
    hidden_layers: [int]
    non_linearity: str
    indirect_encoding: bool
    cppn_hidden_layers: [int]


@attr.s(slots=True, auto_attribs=True, frozen=True)
class LSTMCfg(IBrainCfg):
    lstm_num_layers: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IOptimizerCfg(abc.ABC):
    type: str
    checkpoint_frequency: int
    hof_size: int
    novelty: typing.Optional[NoveltyCfg]


@attr.s(slots=True, auto_attribs=True, frozen=True)
class OptimizerMuLambdaCfg(IOptimizerCfg):
    initial_gene_range: int
    tournsize: int
    mu: int
    lambda_: int
    mutpb: float
    extra_from_hof: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class OptimizerCmaEsCfg(IOptimizerCfg):
    population_size: int
    sigma: float


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ExperimentCfg:
    environment: str
    random_seed: int
    number_generations: int
    brain: IBrainCfg
    episode_runner: IEpisodeRunnerCfg
    optimizer: IOptimizerCfg
    raw_dict: dict
    use_worker_processes: bool
