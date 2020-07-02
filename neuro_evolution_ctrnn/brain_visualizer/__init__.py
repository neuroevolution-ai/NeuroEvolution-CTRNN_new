import tkinter as tk

from neuro_evolution_ctrnn.brain_visualizer.pygame_visualizer import Pygame
#from neuro_evolution_ctrnn.brain_visualizer.pyvis import Pyvis
#from neuro_evolution_ctrnn.brain_visualizer.graphTool import GraphTool
#from neuro_evolution_ctrnn.brain_visualizer.netwulf import Netwulf
from neuro_evolution_ctrnn.brain_visualizer.matplotlib import Matplotlib
#from neuro_evolution_ctrnn.brain_visualizer.graphics import Graphics
#from neuro_evolution_ctrnn.brain_visualizer.plotly import Plotly


class BrainVisualizerHandler(object):
    def __init__(self):
        self.current_visualizer = None

    def launch_new_visualization(self, brain):
        self.current_visualizer = PygameBrainVisualizer(brain)
        return self.current_visualizer


class PygameBrainVisualizer(object):

    def __init__(self, brain):
        self.brain = brain

        # from pyvis.network import Network
        #
        #
        # g = Network()
        # arr = brain.y
        # for i in range(0, 29):
        #     x = arr[i]
        #
        #     g.add_node(i, x)
        #
        # a = g.nodes()
        # g.show("basic.html")
        # print(arr)


    def process_update(self, in_values, out_values):
        Pygame.pygame(self)
        #Pyvis.pyvis(self)       # Browser
        #GraphTool.graphTool(self)      # PDF
        #Netwulf.netwulf(self)   # Browser
        #Matplotlib.matplotlib(self)    # Plot
        #Graphics.graphics(self)   # nix
        #Plotly.plotly(self)    # Browser



        #
        # from pyvis.network import Network
        #
        # global brainSate
        # brainSate = self.brain.y
        #
        # g = Network()
        # arr = brainSate
        # for i in range(0, 29):
        #     x = arr[i]
        #     g.add_node(i, value=x)
        #
        #
        # g.show("basic.html")
        # print(arr)
        #
        # time.sleep(5)



        # # Initial pygame module
        # #pygame.init()
        #
        # pygame.font.init()
        #
        # # Create screen length, width
        # screen = pygame.display.set_mode([1500, 600])
        #
        # # Set display color
        # displayColor = 255, 255, 255
        #
        # # Set Number Color
        # numColor = 0, 0, 0
        #
        # # set font
        # myfont = pygame.font.SysFont("Arial", 15)
        #
        # # value for test
        # global val
        # val = 0
        #
        # def increaseValue(self):
        #     global val
        #     val += 1
        #     return val
        #
        # global brainSate
        # brainSate = 0
        #
        # def increaseBrainSate(self):
        #     global brainSate
        #     brainSate = self.brain.y
        #     return brainSate
        # #
        # # running = True
        # # while running:
        # # for event in pygame.event.get():
        # #     # Stop while, when click on the close button of window
        # #     if event.type == pygame.QUIT:
        # #         running = False
        # #     # Stop while, when hit ESC
        # #     elif event.type == KEYDOWN:
        # #         if event.key == K_ESCAPE:
        # #             running = False
        #
        # screen.fill((displayColor))
        #
        # #val = increaseValue(self)
        #
        # brainSate = increaseBrainSate(self)
        #
        # # Create surface
        # textSurface = myfont.render(str(brainSate), False, numColor)
        #
        # # Surface auf den screen bringen
        # screen.blit(textSurface, (0, 0))
        #
        # # Update the Screen
        # pygame.display.flip()
        #
        # # Updates the content of the window
        # pygame.display.flip()
        #
        # #time.sleep(1)


class BrainVisualizer(object):

    def __init__(self, brain):
        self.brain = brain

        root = tk.Tk()
        self.root = root
        self.app = Window(root)
        root.wm_title("Tkinter window")
        root.geometry("200x200")

    def process_update(self, in_values, out_values):
        self.app.set_text("new states: \n" + str(self.brain.y))
        self.root.update_idletasks()
        self.root.update()


class Window(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=1)
        self.text = tk.StringVar()
        self.set_text("waiting for update...")
        tk.Label(self, textvariable=self.text).pack()

    def set_text(self, text):
        self.text.set(text)

