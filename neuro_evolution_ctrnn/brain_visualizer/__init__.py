import tkinter as tk


class BrainVisualizerHandler(object):
    def __init__(self):
        self.current_visualizer = None

    def launch_new_visualization(self, brain):
        self.current_visualizer = BrainVisualizer(brain)
        return self.current_visualizer


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
