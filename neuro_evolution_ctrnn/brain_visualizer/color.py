class Colors:
    black = (0, 0, 0)
    white = (255, 255, 255)
    grey = (169, 169, 169)
    bright_grey = (220, 220, 220)
    dark_grey = (79, 79, 79)
    custom_red = (185, 19, 44)
    light_green = (188, 255, 169)
    green = (49, 173, 14)
    light_blue = (187, 209, 251)
    blue = (7, 49, 129)
    light_orange = (255, 181, 118)
    orange = (255, 142, 46)

    @staticmethod
    def interpolate_color(color1, color2, val):
        r = int((color2[0] - color1[0]) * abs(val) + color1[0])
        g = int((color2[1] - color1[1]) * abs(val) + color1[1])
        b = int((color2[2] - color1[2]) * abs(val) + color1[2])
        return (r, g, b)
