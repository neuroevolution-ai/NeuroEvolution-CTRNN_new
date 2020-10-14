import sys
import math
from typing import Tuple

import pygame

from brain_visualizer.color import Colors
from brain_visualizer import brain_visualizer


class Events:

    KEY_QUIT = pygame.K_ESCAPE
    KEY_INCREASE_WEIGHT_VAL = pygame.K_e
    KEY_DECREASE_WEIGHT_VAL = pygame.K_r
    KEY_INCREASE_INPUT_NEURON_RADIUS = pygame.K_d
    KEY_DECREASE_INPUT_NEURON_RADIUS = pygame.K_f
    KEY_DISPLAY_NEURON_TEXT = pygame.K_s
    KEY_TOGGLE_POSITIVE_WEIGHTS = pygame.K_t
    KEY_TOGGLE_NEGATIVE_WEIGHTS = pygame.K_w
    KEY_TOGGLE_WEIGHT_ARROWS = pygame.K_g
    KEY_TOGGLE_INPUTS = pygame.K_q
    KEY_TOGGLE_OUTPUTS = pygame.K_z
    KEY_INCREASE_THRESHOLD = pygame.K_PLUS
    KEY_DECREASE_THRESHOLD = pygame.K_MINUS
    KEY_PAUSE_VISUALIZATION = pygame.K_SPACE
    KEY_CONTINUE_VISUALIZATION = pygame.K_c
    KEY_DISPLAY_KEYMAP = pygame.K_i

    @staticmethod
    def handle_events(visualizer: "brain_visualizer.BrainVisualizer", event: pygame.event.EventType,
                      input_positions_dict: dict, output_positions_dict: dict) -> None:
        try:
            # TODO maybe instead of if-if-... make it to if-elif-elif-... so only one action is processed
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                Events.draw_neuron_number(visualizer, input_positions_dict, visualizer.graph_positions_dict,
                                          output_positions_dict,
                                          pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                visualizer.clicked_neuron = Events.get_neuron_on_click(pygame.mouse.get_pos(),
                                                                       visualizer.graph_positions_dict)
            elif event.type == pygame.MOUSEBUTTONUP and isinstance(visualizer.clicked_neuron, int):
                Events.change_neuron_pos(visualizer.clicked_neuron, pygame.mouse.get_pos(),
                                         visualizer.graph_positions_dict)
            elif event.type == pygame.KEYDOWN:
                if event.key == Events.KEY_QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.key == Events.KEY_DECREASE_WEIGHT_VAL:
                    visualizer.weight_val = visualizer.weight_val - 1
                elif event.key == Events.KEY_INCREASE_WEIGHT_VAL:
                    visualizer.weight_val = visualizer.weight_val + 1
                elif event.key == Events.KEY_DECREASE_INPUT_NEURON_RADIUS:
                    if visualizer.input_neuron_radius > 5:
                        # visualizer.neuron_radius = visualizer.neuron_radius - 5
                        visualizer.input_neuron_radius -= 5
                elif event.key == Events.KEY_INCREASE_INPUT_NEURON_RADIUS:
                    visualizer.input_neuron_radius += 5
                elif event.key == Events.KEY_DISPLAY_NEURON_TEXT:
                    visualizer.neuron_text = not visualizer.neuron_text
                elif event.key == Events.KEY_TOGGLE_POSITIVE_WEIGHTS:
                    visualizer.positive_weights = not visualizer.positive_weights
                elif event.key == Events.KEY_TOGGLE_NEGATIVE_WEIGHTS:
                    visualizer.negative_weights = not visualizer.negative_weights
                elif event.key == Events.KEY_TOGGLE_WEIGHT_ARROWS:
                    visualizer.weights_direction = not visualizer.weights_direction
                elif event.key == Events.KEY_TOGGLE_INPUTS:
                    visualizer.input_weights = not visualizer.input_weights
                elif event.key == Events.KEY_TOGGLE_OUTPUTS:
                    visualizer.output_weights = not visualizer.output_weights
                elif event.key == Events.KEY_INCREASE_THRESHOLD:
                    visualizer.draw_threshold = round(visualizer.draw_threshold + 0.05, 2)
                elif event.key == Events.KEY_DECREASE_THRESHOLD:
                    visualizer.draw_threshold = round(visualizer.draw_threshold - 0.05, 2)
                elif event.key == Events.KEY_PAUSE_VISUALIZATION:
                    # TODO refine this so that this is not spinning forever
                    pause = True
                    pygame.event.clear(pygame.KEYDOWN)
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == Events.KEY_CONTINUE_VISUALIZATION:
                                    pause = False
                elif event.key == Events.KEY_DISPLAY_KEYMAP:
                    pygame.event.clear(pygame.KEYDOWN)
                    pygame.draw.rect(visualizer.screen, Colors.dark_grey, (0, visualizer.info_box_height))
                    # TODO render text here, all keys and their descriptions, then when key is pressed again, remove rect
                    # TODO and continue

        except AttributeError:
            print("Failure on Pygame-Event")

    # Get the number of the neuron over which the user hovers the mouse
    @staticmethod
    def draw_neuron_number(visualizer: "brain_visualizer.BrainVisualizer", input_positions_dict: dict,
                           graph_positions_dict: dict, output_positions_dict: dict, mouse_pos: Tuple[int, int]) -> None:
        mouse_font = pygame.font.SysFont("Helvetica", 32)

        hovered_input_neuron = Events.get_neuron_when_hover(visualizer, input_positions_dict, mouse_pos)
        if hovered_input_neuron is not None:
            mouse_label = mouse_font.render("Input:" + str(hovered_input_neuron), 1, Colors.bright_grey)
            visualizer.screen.blit(mouse_label, (mouse_pos[0] + 20, mouse_pos[1] - 32))

        hovered_graph_neuron = Events.get_neuron_when_hover(visualizer, graph_positions_dict, mouse_pos)
        if hovered_graph_neuron is not None:
            mouse_label = mouse_font.render("Neuron:" + str(hovered_graph_neuron), 1, Colors.bright_grey)
            visualizer.screen.blit(mouse_label, (mouse_pos[0] + 20, mouse_pos[1] - 32))

        hovered_output_neuron = Events.get_neuron_when_hover(visualizer, output_positions_dict, mouse_pos)
        if hovered_output_neuron is not None:
            mouse_label = mouse_font.render("Output:" + str(hovered_output_neuron), 1, Colors.bright_grey)
            visualizer.screen.blit(mouse_label, (mouse_pos[0] - 250, mouse_pos[1] - 32))

    @staticmethod
    def get_neuron_when_hover(visualizer: "brain_visualizer.BrainVisualizer", positions_dict: dict,
                              mouse_pos: Tuple[int, int]) -> int:
        max_distance = visualizer.neuron_radius
        for i in range(len(positions_dict)):
            neuron_pos = positions_dict[i]
            distance = math.sqrt(((mouse_pos[0] - neuron_pos[0]) ** 2) + ((mouse_pos[1] - neuron_pos[1]) ** 2))

            if distance < max_distance:
                return i

    # Get the number of the neuron on which the user clicked with the mouse
    @staticmethod
    def get_neuron_on_click(mouse_pos: Tuple[int, int], graph_positions_dict: dict) -> int:
        for i in range(len(graph_positions_dict)):
            neuron_pos = graph_positions_dict[i]
            distance = math.sqrt(((mouse_pos[0] - neuron_pos[0]) ** 2) + ((mouse_pos[1] - neuron_pos[1]) ** 2))
            if distance < 30:
                return i

    @staticmethod
    def change_neuron_pos(neuron: int, mouse_pos: Tuple[int, int], graph_positions_dict: dict) -> None:
        graph_positions_dict[neuron] = (mouse_pos[0], mouse_pos[1])
