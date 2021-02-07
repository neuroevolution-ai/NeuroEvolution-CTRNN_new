# the configurations need to be in a separate file from the actual objects to avoid circular imports
import attr
import abc
from typing import Dict, List, Optional, Type
from brains import IBrainCfg
from tools.config_reader import register


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
    behavior_source: str  # Which value should be used for BC? 'action', 'observation' or 'state'


@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
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


@register('ProcGenAttr')
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class ProcGenEnvAttributesCfg(IEnvAttributesCfg):
    screen_size: int = 64
    distribution_mode: str = 'memory'
    use_monochrome_assets: bool = False
    restrict_themes: bool = True
    use_backgrounds: bool = False


@register('episode_runner', True)
@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
class EpisodeRunnerCfg(abc.ABC):
    reuse_env: bool = True
    keep_env_seed_fixed_during_generation: bool = True
    novelty: Optional[NoveltyCfg] = None
    environment_attributes: Optional[IEnvAttributesCfg] = None
    number_fitness_runs: int = 1
    max_steps_per_run: int = 0
    max_steps_penalty: int = 0
    use_autoencoder: bool = False


@attr.s(slots=True, auto_attribs=True, frozen=True, kw_only=True)
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
