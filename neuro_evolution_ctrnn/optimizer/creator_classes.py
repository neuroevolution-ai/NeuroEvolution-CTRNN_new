from deap import base


class FitnessMax(base.Fitness):
    weights = (1.0,)


class Individual(list):

    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()
