

class Colour():
    def interpolateColor(color1, color2, val):
        R = int((color2[0] - color1[0]) * abs(val) + color1[0])
        G = int((color2[1] - color1[1]) * abs(val) + color1[1])
        B = int((color2[2] - color1[2]) * abs(val) + color1[2])
        color = (R, G, B)
        return color


