class Colour():

    def colors(self, displayColor):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.grey = (169, 169, 169)
        self.brightGrey = (220, 220, 220)
        self.darkGrey = (79, 79, 79)
        # Color for Display
        self.displayColor = displayColor
        # Color for Numbers
        self.numColor = (220, 220, 220)
        # Colors for Weights
        self.colorNegativeWeight = (185, 19, 44)  # rot
        self.colorNeutralWeight = self.darkGrey
        self.colorPositiveWeight = (188, 255, 169)  # hellgrün
        # Colors Neutral Neurons
        self.colorNeutralNeuron = self.darkGrey
        # Color Neurons in Graph
        self.colorNegNeuronGraph = (187, 209, 251)  # Hellblau
        self.colorPosNeuronGraph = (7, 49, 129)  # Blau
        # Color Input Layer
        self.colorNegNeuronIn = (188, 255, 169)  # hellgrün
        self.colorPosNeuronIn = (49, 173, 14)  # grün
        # Color in Output Layer
        self.colorNegNeuronOut = (255, 181, 118)  # Hell-Orange
        self.colorPosNeuronOut = (255, 142, 46)  # orange

    def interpolateColor(color1, color2, val):
        R = int((color2[0] - color1[0]) * abs(val) + color1[0])
        G = int((color2[1] - color1[1]) * abs(val) + color1[1])
        B = int((color2[2] - color1[2]) * abs(val) + color1[2])
        color = (R, G, B)
        return color
