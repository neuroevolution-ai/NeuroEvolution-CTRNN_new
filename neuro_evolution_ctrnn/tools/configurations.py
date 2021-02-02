# the configurations need to be in a separate file from the actual objects to avoid circular imports
import attr
import abc
from typing import Dict, List, Optional

registered_types: Dict = {}
registered_keys: Dict = {}


def register(name: str, as_key: bool = False):
    """
    This decorator registers a class, so that it can later be found by the config_reader with a specific string.
    To map between the class and the string, it uses the content of the type-value of the json node.
    If no type-attribute exists for that node, it uses the key of the node instead.

    We explicitly separate between a mapping by key and by type-string, and thus we manage two separate lists for it.
    Technically this separate isn't needed and it adds clutter, but I think explicit behavior here will help in
    the long run.

    :param name: the name and which the class shall be registered
    :param as_key: if true, the node's key will be used for mapping. If false that type-value will be used
    :return: a wrapper of the class
    """

    # registering magic
    # See https://realpython.com/primer-on-python-decorators/#decorators-with-arguments
    def _register(type_class: type):
        if as_key:
            assert name not in registered_keys, 'key "' + str(name) + '" was already registered'
            registered_keys[name] = type_class
        else:
            assert name not in registered_types, 'type "' + str(name) + '" was already registered'
            registered_types[name] = type_class
        return type_class

    return _register


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IBrainCfg(abc.ABC):
    type: str
    normalize_input: bool
    normalize_input_target: float
    use_bias: bool


@attr.s(slots=True, auto_attribs=True, frozen=True)
class IPytorchBrainCfg(IBrainCfg):
    num_layers: int
    hidden_size: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ILayerBasedBrainCfg(IBrainCfg):
    # The structure of the layers
    # Each list entry translates to the size of one layer
    # The layers are in the given order
    hidden_layer_structure: List[int]
    optimize_initial_neuron_state: bool
    # Whether a neuron can only use its own state form the last timestep
    diagonal_hidden_to_hidden: bool = True


@register('novelty', True)
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
    type: str


@register('ReacherMemoryAttr')
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class ReacherMemoryEnvAttributesCfg(IEnvAttributesCfg):
    observation_frames: int
    memory_frames: int
    action_frames: int


@register('AtariAttr')
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class AtariEnvAttributesCfg(IEnvAttributesCfg):
    screen_size: int = 64
    scale_obs: bool = True
    terminal_on_life_loss: bool = True
    grayscale_obs: bool = False


@register('episode_runner', True)
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


@register('ELMAN')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class ElmanCfg(ILayerBasedBrainCfg):
    pass


@register('LSTMLayered')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class LstmLayeredCfg(ILayerBasedBrainCfg):
    pass


@register('GRU')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class GruCfg(ILayerBasedBrainCfg):
    pass


@register('CTRNN')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class ContinuousTimeRNNCfg(IBrainCfg):
    optimize_initial_neuron_state: bool
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


@register('CNN')
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
    conv_stride1: int = 1
    conv_stride2: int = 1


@register('CNN_CTRNN')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class CnnCtrnnCfg(IBrainCfg):
    cnn_conf: ConvolutionalNNCfg
    ctrnn_conf: ContinuousTimeRNNCfg


@register('FeedForward_PyTorch')
@register('FeedForward_NumPy')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class FeedForwardCfg(IBrainCfg):
    hidden_layers: List[int]
    non_linearity: str
    indirect_encoding: bool
    cppn_hidden_layers: List[int]


@register('LSTM_NumPy')
@register('LSTMNumPy')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class LSTMCfg(IBrainCfg):
    lstm_num_layers: int


@register('ElmanTorch')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class ElmanTorchCfg(IPytorchBrainCfg):
    pass

@register('GruTorch')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class GruTorchCfg(IPytorchBrainCfg):
    pass



@register('LSTM_PyTorch')
@attr.s(slots=True, auto_attribs=True, frozen=True)
class LstmTorchCfg(IPytorchBrainCfg):
    pass


@register('ConcatenatedBrain_LSTM')
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


@register('MU_ES')
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class OptimizerMuLambdaCfg(IOptimizerCfg):
    initial_gene_range: int
    mu: int
    lambda_: int
    mutpb: float
    tournsize: int = 0
    extra_from_hof: int = 0
    strategy_parameter_per_gene: bool = False


@register('CMA_ES')
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class OptimizerCmaEsCfg(IOptimizerCfg):
    population_size: int
    sigma: float
    mu: int


@attr.s(slots=True, auto_attribs=True, frozen=True)
class ExperimentCfg:
    environment: str
    number_generations: int
    brain: IBrainCfg
    episode_runner: EpisodeRunnerCfg
    optimizer: IOptimizerCfg
    random_seed: int = -1
    raw_dict: dict = None  # This attribute is for internal use only
