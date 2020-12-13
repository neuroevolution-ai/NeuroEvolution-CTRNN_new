# the configurations need to be in a separate file from the actual objects to avoid circular imports
import attr
import abc
from typing import Dict, List, Optional

registered_types: Dict = {}


def register_type(type_id: str, type_class: type):
    registered_types[type_id] = type_class


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
    behavior_source: str


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IEnvAttributesCfg(abc.ABC):
    pass


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ReacherMemoryEnvAttributesCfg(IEnvAttributesCfg):
    observation_frames: int
    memory_frames: int
    action_frames: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class AtariEnvAttributesCfg(IEnvAttributesCfg):
    screen_size: int = 64
    scale_obs: bool = True
    terminal_on_life_loss: bool = True
    grayscale_obs: bool = False


@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class EpisodeRunnerCfg(abc.ABC):
    reuse_env: bool
    keep_env_seed_fixed_during_generation: bool = True
    novelty: Optional[NoveltyCfg] = None
    environment_attributes: Optional[IEnvAttributesCfg] = None
    number_fitness_runs: int = 1
    max_steps_per_run: int = 0
    max_steps_penalty: int = 0
    use_autoencoder: bool = False


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ContinuousTimeRNNCfg(IBrainCfg):
    optimize_y0: bool
    delta_t: float
    optimize_state_boundaries: str
    set_principle_diagonal_elements_of_W_negative: bool
    number_neurons: int
    neuron_activation: str
    neuron_activation_inplace: bool = False
    parameter_perturbations: float = 0.0
    v_mask: str = 'dense'
    v_mask_param: float = 0.0
    w_mask: str = 'dense'
    w_mask_param: float = 0.0
    t_mask: str = 'dense'
    t_mask_param: float = 0.0
    clipping_range_min: float = 0
    clipping_range_max: float = 0


register_type('CTRNN', ContinuousTimeRNNCfg)

@attr.s(slots=True, auto_attribs=True, frozen=True)
class ConvolutionalNNCfg(IBrainCfg):
    conv_size1: int
    conv_feat1: int
    maxp_size1: int
    maxp_stride1: int
    conv_size2: int
    conv_feat2: int
    maxp_size2: int
    maxp_stride2: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class CnnCtrnnCfg(IBrainCfg):
    cnn_conf: ConvolutionalNNCfg
    ctrnn_conf: ContinuousTimeRNNCfg


@attr.s(slots=True, auto_attribs=True, frozen=True)
class FeedForwardCfg(IBrainCfg):
    hidden_layers: List[int]
    non_linearity: str
    indirect_encoding: bool
    cppn_hidden_layers: List[int]


@attr.s(slots=True, auto_attribs=True, frozen=True)
class LSTMCfg(IBrainCfg):
    lstm_num_layers: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ConcatenatedBrainLSTMCfg(IBrainCfg):
    lstm: LSTMCfg
    feed_forward_front: FeedForwardCfg = None
    feed_forward_back: FeedForwardCfg = None


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IOptimizerCfg(abc.ABC):
    type: str
    novelty: Optional[NoveltyCfg] = None
    efficiency_weight: float = 0.0
    fix_seed_for_generation: bool = True
    checkpoint_frequency: int = 0
    hof_size: int = 5


@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class OptimizerMuLambdaCfg(IOptimizerCfg):
    initial_gene_range: int
    mu: int
    lambda_: int
    mutpb: float
    tournsize: int = 0
    extra_from_hof: int = 0
    strategy_parameter_per_gene: bool = False


register_type('MU_ES', OptimizerMuLambdaCfg)


@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class OptimizerCmaEsCfg(IOptimizerCfg):
    population_size: int
    sigma: float
    mu: int


register_type('CMA_ES', OptimizerCmaEsCfg)


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ExperimentCfg:
    environment: str
    number_generations: int
    brain: IBrainCfg
    episode_runner: EpisodeRunnerCfg
    optimizer: IOptimizerCfg
    random_seed: int = -1
    raw_dict: dict = None  # This attribute is for internal use only
