# the configurations need to be in a separate file from the actual objects to avoid circular imports
from typing import List
import attr
import abc


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IBrainCfg(abc.ABC):
    type: str
    normalize_input: bool
    normalize_input_target: float


@attr.s(slots=True, auto_attribs=True, frozen=True)
class EpisodeRunnerCfg:
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool
    reuse_env: bool


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


@attr.s(slots=True, auto_attribs=True, frozen=True)
class LayeredNNCfg(IBrainCfg):
    number_neurons_layer1: int
    number_neurons_layer2: int
    cppn_hidden_size1: int
    cppn_hidden_size2: int
    use_biases: bool
    indirect_encoding: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class LSTMCfg(IBrainCfg):
    lstm_num_layers: int
    use_biases: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IOptimizerCfg(abc.ABC):
    type: str
    checkpoint_frequency: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class OptimizerMuLambdaCfg(IOptimizerCfg):
    population_size: int
    initial_gene_range: int
    mate_indpb_1: float
    mate_indpb_2: float
    mutation_Gaussian_sigma_1: float
    mutation_Gaussian_sigma_2: float
    mutation_Gaussian_sigma_3: float
    mutation_Gaussian_indpb_1: float
    mutation_Gaussian_indpb_2: float
    mutation_Gaussian_indpb_3: float
    elitist_ratio: int
    tournsize: int
    mu: float
    lambda_: float
    mutpb: float
    include_parents_in_next_generation: float


@attr.s(slots=True, auto_attribs=True, frozen=True)
class OptimizerCmaEsCfg(IOptimizerCfg):
    population_size: int
    sigma: float


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ExperimentCfg:
    environment: str
    random_seed: int
    number_generations: int
    brain: ContinuousTimeRNNCfg
    episode_runner: EpisodeRunnerCfg
    optimizer: OptimizerCmaEsCfg
    raw_dict: dict
