from brain_visualizer.position import Positions
from brain_visualizer.weights import Weights
from brain_visualizer.neurons import Neurons
from brain_visualizer.events import Events
from brain_visualizer.color import Colors
from tools.configurations import IBrainCfg
from brains.i_brain import IBrain

import pygame
import numpy as np
import os
from typing import Tuple


class BrainVisualizerHandler:
    def __init__(self):
        self.current_visualizer = None

    # color_clipping_range for colorClipping Input [0], Graph [1] and Output [2]
    def launch_new_visualization(self,
                                 brain: IBrain,
                                 brain_config: IBrainCfg,
                                 env_id: str,
                                 width: int = 1800,
                                 height: int = 800,
                                 display_color: Tuple[int, int, int] = (0, 0, 0),
                                 neuron_radius: int = 30,
                                 color_clipping_range: Tuple[int, int, int] = (1, 1, 1)):

        self.current_visualizer = BrainVisualizer(brain=brain, brain_config=brain_config, env_id=env_id, width=width,
                                                  height=height, display_color=display_color,
                                                  neuron_radius=neuron_radius,
                                                  color_clipping_range=color_clipping_range)
        return self.current_visualizer


class BrainVisualizer:
    def __init__(self, brain: IBrain, brain_config: IBrainCfg, env_id: str, width: int, height: int,
                 display_color: Tuple[int, int, int], neuron_radius: int, color_clipping_range: Tuple[int, int, int]):
        self.brain = brain
        self.brain_config = brain_config
        self.env_id = env_id

        # Initial pygame module
        successes, failures = pygame.init()
        if failures:
            print("{0} successes and{1} failures".format(successes, failures))

        # Set position of screen (x, y) & create screen (length, width)
        # TODO remove the following when finished debugging
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (2500, 510)  # for a fixed position of the window
        self.screen = pygame.display.set_mode([width, height])
        self.w, self.h = pygame.display.get_surface().get_size()

        # Give it a name
        pygame.display.set_caption('Neurorobotics - Brain Visualizer')

        self.kit_logo = pygame.image.load("resources/kit_grey_50.png")
        self.kit_logo.convert()
        self.kit_rect = self.kit_logo.get_rect()
        self.kit_rect.center = 90, 30

        # Initialize & set font
        pygame.font.init()
        self.my_font = pygame.font.SysFont("Helvetica", 14)

        # Dictionary Graph Neurons
        # Create Graph with Spring layout and get Positions of Neurons back
        self.graph_positions_dict = Positions.get_graph_positions(self)

        # Define all colors
        # Colour.colors(self, display_color)
        self.display_color = display_color

        self.color_clipping_range = color_clipping_range

        # Variables for events
        self.positive_weights = True
        self.negative_weights = True
        self.weights_direction = False
        self.input_weights = True
        self.output_weights = True
        self.weight_val = 0  # Defines how many connections will be drawn, default: every connection
        self.neuron_radius = neuron_radius
        self.neuron_text = True
        self.clicked_neuron = None

        # Color for Numbers
        self.num_color = Colors.bright_grey

        # Define colors used in the program
        # Colors for Weights
        self.color_negative_weight = Colors.custom_red
        self.color_neutral_weight = Colors.dark_grey
        self.color_positive_weight = Colors.light_green

        # Colors Neutral Neurons
        self.color_neutral_neuron = Colors.dark_grey

        # Color Neurons in Graph
        self.color_negative_neuron_graph = Colors.light_blue
        self.color_positive_neuron_graph = Colors.blue

        # Color Input Layer
        self.color_negative_neuron_in = Colors.light_green
        self.color_positive_neuron_in = Colors.green

        # Color in Output Layer
        self.color_negative_neuron_out = Colors.light_orange
        self.color_positive_neuron_out = Colors.orange

    def process_update(self, in_values, out_values):
        # Fill screen with color
        self.screen.fill(self.display_color)

        # Draw Rect for Logo and Blit Logo on the screen
        pygame.draw.rect(self.screen, Colors.dark_grey, (0, 0, self.w, 60))
        self.screen.blit(self.kit_logo, self.kit_rect)

        if self.brain_config.use_bias:
            in_values = np.r_[in_values, [1]]

        ##### Number Neurons
        numberInputNeurons = len(in_values)
        numberNeurons = len(self.brain.W.todense())
        numberOutputNeurons = 1 if isinstance(out_values, np.int64) else len(out_values)

        ########## Draw Legend
        PygameBrainVisualizer.render_InfoText(self, [
            "Input Neurons: " + str(numberInputNeurons),
            "Graph Neurons: " + str(numberNeurons),
            "Output Neurons: " + str(numberOutputNeurons)],
                                              ((self.w / 4) - 80), 5, 18)

        PygameBrainVisualizer.render_InfoText(self, [
            "Positive/Negative Weights [t,w] : " + str(self.positive_weights) + " / " + str(self.negative_weights),
            "Input/Output Weights [q,z] : " + str(self.input_weights) + " / " + str(self.output_weights),
            "Direction [g] : " + str(self.weights_direction)],
                                              ((self.w / 2) - 130), 5, 18)

        if self.weight_val == 0:  # If weight_val is 0 every Connection will be drawn
            text = "all"
        else:
            text = str(self.weight_val)
        PygameBrainVisualizer.render_InfoText(self, [
            "Weights [e,r] : " + text,
            "Values [s] : " + str(self.neuron_text),
            "Simulation : " + str(self.env_id)],
                                              ((3 * self.w / 4) - 80), 5, 18)

        ########## Create Dictionaries with Positions
        ##### Input Dictionary
        inputPositionsDict = Positions.get_input_output_positions(self, numberInputNeurons, True)

        ##### Dictionary Graph Neurons
        # --> self.graph_positions_dict

        ##### Output Dictionary
        outputPositionsDict = Positions.get_input_output_positions(self, numberOutputNeurons, False)

        ########## Draw Weights
        ##### n-1 Linien pro Neuron ; Input zu Neuron
        if self.input_weights:
            Weights.drawWeights(self, inputPositionsDict, self.graph_positions_dict, self.brain.V.todense().T,
                                self.positive_weights, self.negative_weights, self.weights_direction)

        # ##### n-1 Linien pro Neuron ; Neuron zu Neuron
        Weights.drawWeights(self, self.graph_positions_dict, self.graph_positions_dict, self.brain.W.todense(),
                            self.positive_weights, self.negative_weights, self.weights_direction)

        # ##### n-1 Linien pro Neuron ; Neuron zu Output
        if self.output_weights:
            Weights.drawWeights(self, self.graph_positions_dict, outputPositionsDict, self.brain.T.todense(),
                                self.positive_weights, self.negative_weights, self.weights_direction)

        # #### 1 Kreis pro Neuron ; Neuron zu sich selbst ; Radius +5 damit Kreis größer als Neuron ist
        Neurons.drawNeurons(self, self.graph_positions_dict, self.brain.W, 2, self.color_negative_weight,
                            self.color_neutral_weight, self.color_positive_weight, self.neuron_radius + self.weight_val, True,
                            True)

        ########### Draw neurons
        ##### Draw Graph
        Neurons.drawNeurons(self, self.graph_positions_dict, self.brain.y, self.color_clipping_range[1],
                            self.color_negative_neuron_graph, self.color_neutral_neuron, self.color_positive_neuron_graph,
                            self.neuron_radius - 3, False)

        ##### draw ob-values (input)
        # minMax = clipping Range
        Neurons.drawNeurons(self, inputPositionsDict, in_values, self.color_clipping_range[0], self.color_negative_neuron_in,
                            self.color_neutral_neuron, self.color_positive_neuron_in, self.neuron_radius)

        ##### draw action-values (out)
        # minMax = clipping Range
        Neurons.drawNeurons(self, outputPositionsDict, out_values, self.color_clipping_range[2], self.color_negative_neuron_out,
                            self.color_neutral_neuron, self.color_positive_neuron_out, self.neuron_radius)

        ######### Events: Close when x-Button, Show Number of Neuron when click on it
        for event in pygame.event.get():
            Events.handleEvents(self, event, inputPositionsDict, outputPositionsDict)

        # Updates the content of the window
        pygame.display.flip()

    def render_InfoText(self, list_of_strings, x_pos, initial_y_pos, y_step):
        y_pos = initial_y_pos
        for s in list_of_strings:
            textSurface = self.my_font.render(s, False, self.numColor)
            self.screen.blit(textSurface, (x_pos, y_pos))
            y_pos += y_step
