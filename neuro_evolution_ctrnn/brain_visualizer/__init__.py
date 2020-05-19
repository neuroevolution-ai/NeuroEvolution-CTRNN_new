

class BrainVisualizerHandler(object):
    def __init__(self):
        self.current_visualizer = None

    def launch_new_visualization(self, individual):
        print("launching new visualization", self)
        self.current_visualizer = BrainVisualizer(individual)
        return self.current_visualizer


class BrainVisualizer(object):

    def __init__(self, individual):
        self.individual = individual
        print("initializing new brain visualizer for individual", self.individual)

    def process_update(self, y):
        # print("update called for indivual", self.individual)
        # print("new neuron states received", y)
        pass
