import pygame
import sys
import math
from pygame.locals import QUIT, MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN, KEYDOWN
from typing import Tuple

from brain_visualizer.color import Colors
from brain_visualizer import brain_visualizer


class Events:

    @staticmethod
    def handle_events(visualizer: "brain_visualizer.BrainVisualizer", event: pygame.event.EventType,
                      input_positions_dict: dict, output_positions_dict: dict) -> None:
        try:
            # TODO maybe instead of if-if-... make it to if-elif-elif-... so only one action is processed
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEMOTION:
                Events.draw_neuron_number(visualizer, input_positions_dict, visualizer.graph_positions_dict,
                                          output_positions_dict,
                                          pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONDOWN:
                visualizer.clicked_neuron = Events.get_neuron_on_click(pygame.mouse.get_pos(),
                                                                       visualizer.graph_positions_dict)
            if event.type == MOUSEBUTTONUP and isinstance(visualizer.clicked_neuron, int):
                Events.change_neuron_pos(visualizer.clicked_neuron, pygame.mouse.get_pos(),
                                         visualizer.graph_positions_dict)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_e:
                    visualizer.weight_val = visualizer.weight_val - 1
                if event.key == pygame.K_r:
                    visualizer.weight_val = visualizer.weight_val + 1
                if event.key == pygame.K_d:
                    if visualizer.neuron_radius > 5:
                        visualizer.neuron_radius = visualizer.neuron_radius - 5
                        print(visualizer.neuron_radius)
                if event.key == pygame.K_f:
                    visualizer.neuron_radius = visualizer.neuron_radius + 5
                if event.key == pygame.K_s:
                    if visualizer.neuron_text:
                        visualizer.neuron_text = False
                    else:
                        visualizer.neuron_text = True
                if event.key == pygame.K_t:
                    if visualizer.positive_weights:
                        visualizer.positive_weights = False
                    else:
                        visualizer.positive_weights = True
                if event.key == pygame.K_w:
                    if visualizer.negative_weights:
                        visualizer.negative_weights = False
                    else:
                        visualizer.negative_weights = True
                if event.key == pygame.K_g:
                    if visualizer.weights_direction:
                        visualizer.weights_direction = False
                    else:
                        visualizer.weights_direction = True
                if event.key == pygame.K_q:
                    if visualizer.input_weights:
                        visualizer.input_weights = False
                    else:
                        visualizer.input_weights = True
                if event.key == pygame.K_z:
                    if visualizer.output_weights:
                        visualizer.output_weights = False
                    else:
                        visualizer.output_weights = True
                if event.key == pygame.K_SPACE:
                    pause = True
                    pygame.event.clear(KEYDOWN)
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_c:
                                    pause = False

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
