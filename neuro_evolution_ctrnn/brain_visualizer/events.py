import pygame
import sys
import math
from pygame.locals import QUIT, MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN, KEYDOWN
from typing import Tuple

from brain_visualizer.color import Colors


class Events:

    @staticmethod
    def handle_events(brain_visualizer, event: pygame.event.EventType, input_positions_dict: dict, output_positions_dict: dict):
        try:
            # TODO maybe instead of if-if-... make it to if-elif-elif-... so only one action is processed
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEMOTION:
                Events.draw_neuron_number(brain_visualizer, input_positions_dict, brain_visualizer.graph_positions_dict, output_positions_dict,
                                          pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONDOWN:
                brain_visualizer.clicked_neuron = Events.get_neuron_on_click(pygame.mouse.get_pos(), brain_visualizer.graph_positions_dict)
            if event.type == MOUSEBUTTONUP and isinstance(brain_visualizer.clicked_neuron, int):
                Events.change_neuron_pos(brain_visualizer.clicked_neuron, pygame.mouse.get_pos(), brain_visualizer.graph_positions_dict)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_e:
                    brain_visualizer.weight_val = brain_visualizer.weight_val - 1
                if event.key == pygame.K_r:
                    brain_visualizer.weight_val = brain_visualizer.weight_val + 1
                if event.key == pygame.K_d:
                    if brain_visualizer.neuron_radius > 5:
                        brain_visualizer.neuron_radius = brain_visualizer.neuron_radius - 5
                        print(brain_visualizer.neuron_radius)
                if event.key == pygame.K_f:
                    brain_visualizer.neuron_radius = brain_visualizer.neuron_radius + 5
                if event.key == pygame.K_s:
                    if brain_visualizer.neuron_text:
                        brain_visualizer.neuron_text = False
                    else:
                        brain_visualizer.neuron_text = True
                if event.key == pygame.K_t:
                    if brain_visualizer.positive_weights:
                        brain_visualizer.positive_weights = False
                    else:
                        brain_visualizer.positive_weights = True
                if event.key == pygame.K_w:
                    if brain_visualizer.negative_weights:
                        brain_visualizer.negative_weights = False
                    else:
                        brain_visualizer.negative_weights = True
                if event.key == pygame.K_g:
                    if brain_visualizer.weights_direction:
                        brain_visualizer.weights_direction = False
                    else:
                        brain_visualizer.weights_direction = True
                if event.key == pygame.K_q:
                    if brain_visualizer.input_weights:
                        brain_visualizer.input_weights = False
                    else:
                        brain_visualizer.input_weights = True
                if event.key == pygame.K_z:
                    if brain_visualizer.output_weights:
                        brain_visualizer.output_weights = False
                    else:
                        brain_visualizer.output_weights = True
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
    def draw_neuron_number(brain_visualizer, input_positions_dict: dict, graph_positions_dict: dict, output_positions_dict: dict, mouse_pos: Tuple[int, int]):
        mouse_font = pygame.font.SysFont("Helvetica", 32)

        hovered_input_neuron = Events.get_neuron_when_hover(brain_visualizer, input_positions_dict, mouse_pos)
        if hovered_input_neuron is not None:
            mouse_label = mouse_font.render("Input:" + str(hovered_input_neuron), 1, Colors.bright_grey)
            brain_visualizer.screen.blit(mouse_label, (mouse_pos[0] + 20, mouse_pos[1] - 32))

        hovered_graph_neuron = Events.get_neuron_when_hover(brain_visualizer, graph_positions_dict, mouse_pos)
        if hovered_graph_neuron is not None:
            mouse_label = mouse_font.render("Neuron:" + str(hovered_graph_neuron), 1, Colors.bright_grey)
            brain_visualizer.screen.blit(mouse_label, (mouse_pos[0] + 20, mouse_pos[1] - 32))

        hovered_output_neuron = Events.get_neuron_when_hover(brain_visualizer, output_positions_dict, mouse_pos)
        if hovered_output_neuron is not None:
            mouse_label = mouse_font.render("Output:" + str(hovered_output_neuron), 1, Colors.bright_grey)
            brain_visualizer.screen.blit(mouse_label, (mouse_pos[0] - 250, mouse_pos[1] - 32))

    @staticmethod
    def get_neuron_when_hover(brain_visualizer, positions_dict: dict, mouse_pos: Tuple[int, int]) -> int:
        max_distance = brain_visualizer.neuron_radius
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
    def change_neuron_pos(neuron: int, mouse_pos: Tuple[int, int], graph_positions_dict: dict):
        graph_positions_dict[neuron] = (mouse_pos[0], mouse_pos[1])
