import pygame
import os
import time
import numpy as np


class Pygame():
    def pygame(self):

        # Initial pygame module
        pygame.init()

        # Set position of screen (x, y) & create screen (length, width)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (115, 510)
        screen = pygame.display.set_mode([1000, 500])

        # Set display color
        displayColor = 0, 0, 0
        screen.fill((displayColor))

        # Set Color for Numbers
        numColor = 255, 255, 255

        # initialize & set font
        pygame.font.init()
        myfont = pygame.font.SysFont("Arial", 15)

        # Calling render function creates a surface
        # brainSate = self.brain.y
        # textSurface = myfont.render(str(brainSate), False, numColor)

        # Surface auf den screen bringen
        # screen.blit(textSurface, (50, 50))

        # draw circles
        brainSate = self.brain.y
        numberNeurons = len(brainSate)
        for x in range(numberNeurons):
            val = str(self.brain.y[x])
            color = abs((self.brain.y[x]) * 255) * 10
            textSurface = myfont.render(('%.5s' % val), False, numColor)
            x = x + 1
            if x <= 10:
                pos_y = 50
                pos_x = x * 80
                pygame.draw.circle(screen, (0, 0, color), (pos_x, pos_y), 30)
                screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))
            elif x > 10 and x <= 20:
                x1 = x - 10
                pos_y = 150
                pos_x = x1 * 80
                pygame.draw.circle(screen, (0, 50, color), (pos_x, pos_y), 30)
                screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

            elif x > 20 and x <= 30:
                x2 = x - 20
                pos_y = 250
                pos_x = x2 * 80
                pygame.draw.circle(screen, (50, 50, color), (pos_x, pos_y), 30)
                screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        val = str(0)
        color = (0 * 255)
        textSurface = myfont.render(('%.5s' % val), False, numColor)
        pos_y = 350
        pos_x = 80
        pygame.draw.circle(screen, (0, 50, color), (pos_x, pos_y), 30)
        screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        val = str(0.5)
        color = (0.5 * 255)
        textSurface = myfont.render(('%.5s' % val), False, numColor)
        pos_y = 350
        pos_x = 160
        pygame.draw.circle(screen, (0, 50, color), (pos_x, pos_y), 30)
        screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        val = str(1)
        color = (1 * 255)
        textSurface = myfont.render(('%.5s' % val), False, numColor)
        pos_y = 350
        pos_x = 240
        pygame.draw.circle(screen, (0, 50, color), (pos_x, pos_y), 30)
        screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        val = str(0.8)
        color = (0.8 * 255)
        textSurface = myfont.render(('%.5s' % val), False, numColor)
        pos_y = 350
        pos_x = 320
        pygame.draw.circle(screen, (0, 50, color), (pos_x, pos_y), 30)
        screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        # Update the Screen
        pygame.display.flip()

        # Updates the content of the window
        pygame.display.flip()

        # time.sleep(1)

# def increaseBrainState():
#     global brainSate
#     brainSate += 1
#     return brainSate
#
# class PygameBrainVisualizerHandler(object):
#     def __init__(self):
#         self.current_visualizer = None
#
#     def launch_new_visualization(self, brain):
#         self.current_visualizer = PygameBrainVisualizer(brain)
#         return self.current_visualizer
#
#
# class PygameBrainVisualizer(object):
#
#     def __init__(self, brain):
#         self.brain = brain
#
#         # Initial pygame module
#         pygame.init()
#
#         pygame.font.init()
#
#         # Create screen length, width
#         screen = pygame.display.set_mode([800, 600])
#
#         # Set display color
#         displayColor = 255, 255, 255
#
#         # Set Number Color
#         numColor = 0, 0, 0
#
#         # set font
#         myfont = pygame.font.SysFont("Arial", 25)
#
#         # Status of the Brain
#         brainSate = 0
#
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 # Stop while, when click on the close button of window
#                 if event.type == pygame.QUIT:
#                     running = False
#                 # Stop while, when hit ESC
#                 elif event.type == KEYDOWN:
#                     if event.key == K_ESCAPE:
#                         running = False
#
#             screen.fill((displayColor))
#
#             increaseBrainState()
#
#             # Create surface
#             textSurface = myfont.render(str(brainSate), False, numColor)
#
#             # Surface auf den screen bringen
#             screen.blit(textSurface, (0, 0))
#
#             # Update the Screen
#             pygame.display.flip()
#
#             # Updates the content of the window
#             pygame.display.flip()
#
#             time.sleep(1)
#
#         pygame.quit()
