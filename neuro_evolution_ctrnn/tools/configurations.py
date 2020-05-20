# the configurations need to be in a separate file from the actual objects to avoid circular imports

class EpisodeRunnerCfg:
    number_fitness_runs: int
    keep_env_seed_fixed_during_generation: bool

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
    brain: any
    episode_runner: EpisodeRunnerCfg
    trainer: TrainerCmaEsCfg
    _raw_dict: dict

    def __init__(self, **attr):
        self.__dict__ = attr
