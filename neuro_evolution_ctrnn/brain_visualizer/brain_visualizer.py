# noinspection PyUnresolvedReferences
import os
from typing import Tuple

import pygame
import numpy as np

from brain_visualizer.position import Positions
from brain_visualizer.weights import Weights
from brain_visualizer.neurons import Neurons
from brain_visualizer.events import Events
from brain_visualizer.color import Colors
from tools.configurations import IBrainCfg
from brains.continuous_time_rnn import ContinuousTimeRNN


class BrainVisualizerHandler:
    def __init__(self):
        self.current_visualizer = None

    # color_clipping_range for colorClipping Input [0], Graph [1] and Output [2]
    def launch_new_visualization(self,
                                 brain: ContinuousTimeRNN,
                                 brain_config: IBrainCfg,
                                 env_id: str,
                                 initial_observation: np.ndarray,
                                 width: int = 1800,
                                 height: int = 800,
                                 display_color: Tuple[int, int, int] = (0, 0, 0),
                                 neuron_radius: int = 30,
                                 color_clipping_range: Tuple[int, int, int] = (1, 1, 1)):
        self.current_visualizer = BrainVisualizer(brain=brain, brain_config=brain_config, env_id=env_id,
                                                  initial_observation=initial_observation, width=width, height=height,
                                                  display_color=display_color,
                                                  neuron_radius=neuron_radius,
                                                  color_clipping_range=color_clipping_range)
        return self.current_visualizer


class BrainVisualizer:

    def __init__(self,
                 brain: ContinuousTimeRNN,
                 brain_config: IBrainCfg,
                 env_id: str,
                 initial_observation: np.ndarray,
                 width: int,
                 height: int,
                 display_color: Tuple[int, int, int],
                 neuron_radius: int,
                 color_clipping_range: Tuple[int, int, int]):
        self.brain = brain
        self.brain_config = brain_config
        self.env_id = env_id

        # Initial pygame module
        successes, failures = pygame.init()
        if failures:
            print("{0} successes and {1} failures".format(successes, failures))

        # Set position of screen (x, y) & create screen (length, width)
        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (3839, 2159)  # for a fixed position of the window
        self.screen = pygame.display.set_mode([width, height])
        self.w, self.h = pygame.display.get_surface().get_size()
        self.info_box_size = 60
        self.input_box_width = int(self.w * 0.2)

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

        # Variables for events
        self.positive_weights = True  # Draws positive weights
        self.negative_weights = True  # Draws negative weights
        self.weights_direction = True  # Activates arrows for the edges
        self.input_weights = True
        self.output_weights = True
        self.weight_val = 0  # Defines how many connections will be drawn, default: every connection
        self.input_neuron_radius = neuron_radius
        self.output_neuron_radius = neuron_radius
        self.neuron_radius = neuron_radius
        self.neuron_text = True
        self.clicked_neuron = None

        # Settings for drawing the weights
        self.draw_weight_mode = Weights.WEIGHT_MAX
        # Threshold for drawing connections, 0 means it is disabled
        self.draw_threshold = 0.0

        self.rgb_input = False
        self.input_shape = None

        # If the dimension of the input (the observation) is three-dimensional we assume that the environment delivers
        # RGB pixels as inputs of kind [Width, Height, RGB]
        if initial_observation is not None:
            if len(initial_observation.shape) == 1:
                self.rgb_input = False
            elif len(initial_observation.shape) == 3:
                self.rgb_input = True
                self.draw_weight_mode = self.draw_weight_mode | Weights.IGNORE_ZERO_INPUT
            else:
                # Only one dimensional or three dimensional input is allowed
                raise RuntimeError(
                    "Only one-dimensional or three-dimensional input is supported for the BrainVisualizer.")

            self.input_shape = initial_observation.shape

        # Define colors used in the program
        self.display_color = display_color

        self.color_clipping_range = color_clipping_range

        # Color for Numbers
        self.color_numbers = Colors.bright_grey

        # Colors for Weights
        self.color_negative_weight = Colors.custom_red
        self.color_neutral_weight = Colors.dark_grey
        self.color_positive_weight = Colors.light_green
        self.color_input_connections_positive = Colors.less_dark_green
        self.color_input_connections_negative = Colors.less_dark_blue

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

    def render_info_text(self, list_of_strings, x_pos, initial_y_pos, y_step):
        y_pos = initial_y_pos
        for s in list_of_strings:
            text_surface = self.my_font.render(s, True, self.color_numbers)
            self.screen.blit(text_surface, (x_pos, y_pos))
            y_pos += y_step

    def process_update(self, in_values: np.ndarray, out_values: np.ndarray):
        # Fill screen with neutral_color
        self.screen.fill(self.display_color)

        # Draw Rect for Logo and Blit Logo on the screen
        pygame.draw.rect(self.screen, Colors.dark_grey, (0, 0, self.w, self.info_box_size))
        self.screen.blit(self.kit_logo, self.kit_rect)

        if self.rgb_input:
            in_values = np.concatenate(
                (in_values[:, :, 0].flatten(), in_values[:, :, 1].flatten(), in_values[:, :, 2].flatten()))

        if self.brain_config.use_bias:
            # Add 1 to the end of the input values so that the bias can be added. This is needed because the weight
            # matrix of the brain has one additional value (the bias)
            in_values = np.concatenate((in_values, [1]))

        number_input_neurons = in_values.size
        number_neurons = len(self.brain.W.todense())
        number_output_neurons = 1 if isinstance(out_values, np.int64) else len(out_values)

        # Draw Legend
        # TODO refine this text rendering to use some kind of layout, maybe pygame offers something like a GridLayout
        self.render_info_text(
            ["Input Neurons: " + str(number_input_neurons),
             "Graph Neurons: " + str(number_neurons),
             "Output Neurons: " + str(number_output_neurons)], x_pos=((self.w / 4) - 80), initial_y_pos=5, y_step=18)

        self.render_info_text(
            ["Positive/Negative Weights [t,w] : " + str(self.positive_weights) + " / " + str(self.negative_weights),
             "Input/Output Weights [q,z] : " + str(self.input_weights) + " / " + str(self.output_weights),
             "Direction [g] : " + str(self.weights_direction)], x_pos=((self.w / 2) - 130), initial_y_pos=5, y_step=18)

        if self.weight_val == 0:  # If weight_val is 0 every Connection will be drawn
            text = "all"
        else:
            text = str(self.weight_val)
        self.render_info_text(
            ["Weights [e,r] : " + text, "Values [s] : " + str(self.neuron_text), "Simulation : " + str(self.env_id)],
            x_pos=((3 * self.w / 4) - 80), initial_y_pos=5, y_step=18)

        # self.render_info_text(["Threshold: {}".format(self.draw_threshold)], x_pos=(3 * self.w / 4) + 150,
        #                       initial_y_pos=5, y_step=18)

        # Create Dictionaries with Positions
        # Input Dictionary
        input_positions_dict = Positions.calculate_positions(self, in_values, is_input=True)

        # Output Dictionary
        output_positions_dict = Positions.calculate_positions(self, out_values, is_input=False)

        # Draw Weights
        # This will draw the weights (i.e. the connections) between the input and the neurons
        if self.input_weights:
            Weights.draw_weights(self, input_positions_dict, self.graph_positions_dict, self.brain.V.toarray().T,
                                 is_input=True, in_values=in_values)

        # Connections between the Neurons
        Weights.draw_weights(self, self.graph_positions_dict, self.graph_positions_dict, self.brain.W.toarray(),
                             is_input=False, in_values=None)

        # Connections between the Neurons and the Output
        if self.output_weights:
            Weights.draw_weights(self, self.graph_positions_dict, output_positions_dict, self.brain.T.toarray(),
                                 is_input=False, in_values=None)

        # Draw neurons

        # Draws one circle per neuron, and connections from neurons to itself
        # Radius is increased so that the circle is bigger than the neuron itself
        Neurons.draw_neurons(visualizer=self,
                             positions=self.graph_positions_dict,
                             value_dict=self.brain.W,
                             color_clipping_range=2,
                             negative_color=self.color_negative_weight,
                             neutral_color=self.color_neutral_weight,
                             positive_color=self.color_positive_weight,
                             radius=self.neuron_radius + self.weight_val,  # TODO weight_val necessary here?
                             matrix=True,
                             weight_neuron=True)

        # Draw graph
        Neurons.draw_neurons(visualizer=self,
                             positions=self.graph_positions_dict,
                             value_dict=self.brain.y,
                             color_clipping_range=self.color_clipping_range[1],
                             negative_color=self.color_negative_neuron_graph,
                             neutral_color=self.color_neutral_neuron,
                             positive_color=self.color_positive_neuron_graph,
                             radius=self.neuron_radius - 3)

        # Draw the inputs to the brain
        Neurons.draw_neurons(visualizer=self,
                             positions=input_positions_dict,
                             value_dict=in_values,
                             color_clipping_range=self.color_clipping_range[0],
                             negative_color=self.color_negative_neuron_in,
                             neutral_color=self.color_neutral_neuron,
                             positive_color=self.color_positive_neuron_in,
                             radius=self.input_neuron_radius,
                             is_input=True)

        # Draw the output(s) of the brain
        Neurons.draw_neurons(visualizer=self,
                             positions=output_positions_dict,
                             value_dict=out_values,
                             color_clipping_range=self.color_clipping_range[2],
                             negative_color=self.color_negative_neuron_out,
                             neutral_color=self.color_neutral_neuron,
                             positive_color=self.color_positive_neuron_out,
                             radius=self.output_neuron_radius)

        # Handles keyboard and mouse events in the program
        for event in pygame.event.get():
            Events.handle_events(self, event, input_positions_dict, output_positions_dict)

        # Updates the content of the window
        pygame.display.flip()
