from typing import Tuple


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
    pink = (255, 192, 203)
    aqua = (0, 255, 255, 10)
    dark_green = (0, 51, 0)
    less_dark_green = (0, 102, 0)
    dark_red = (128, 0, 0)
    less_dark_red = (153, 51, 51)
    dark_blue = (0, 0, 102)
    less_dark_blue = (0, 51, 102)

    @staticmethod
    def interpolate_color(
            color1: Tuple[int, int, int], color2: Tuple[int, int, int], val: float) -> Tuple[int, int, int]:
        r = int((color2[0] - color1[0]) * abs(val) + color1[0])
        g = int((color2[1] - color1[1]) * abs(val) + color1[1])
        b = int((color2[2] - color1[2]) * abs(val) + color1[2])
        return (r, g, b)
