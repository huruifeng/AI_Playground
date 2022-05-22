##########################
## Ruifeng Hu
## 05-11-2022
## Lexington, MA
## hurufieng.cn@hotmail.com
##########################

##--------------------------------------------
# Using the Genetic Algorithm(GA) to draw
#   the FireFox logo
#

##------------------------------------------

import copy
import numpy as np
import skimage
from PIL import Image, ImageDraw


class Triangle:
    shape = [256, 256]
    def __init__(self, x, y, color, alpha):
        self.x = x
        self.y = y
        self.color = color
        self.alpha = alpha
        self.img_t=None

    def draw_t2(self):
        self.img_t = Image.new('RGBA', self.shape)
        draw = ImageDraw.Draw(self.img_t)
        draw.polygon([(self.x[0], self.y[0]),
                      (self.x[1], self.y[1]),
                      (self.x[2], self.y[2])],
                     fill=(self.color[0], self.color[1], self.color[2], int(self.alpha*256)))
        return self.img_t

class Individual:
    shape = [256, 256]

    def __init__(self):
        self.triangles = []
        self.fitness = 0
        self.img = None

    def copy(self):
        return copy.deepcopy(self)

    def draw_im(self):
        self.img = np.ones(self.shape+[3], dtype=np.uint8) * 255
        for t in self.triangles:
            xx, yy = skimage.draw.polygon(t.x, t.y)
            skimage.draw.set_color(self.img, (xx, yy), t.color, t.alpha)
        return self.img

    def draw_im2(self):
        self.img = Image.new('RGB', self.shape)
        draw = ImageDraw.Draw(self.img)
        draw.polygon([(0, 0), (0, 255), (255, 255), (255, 0)], fill=(255, 255, 255, 255))
        for triangle in self.triangles:
            self.img = Image.alpha_composite(self.img, triangle.draw_t2())
        return self.img

    def f(self):
        im = self.draw_im()
        # The bigest diatance (self.target_im.size * ((3 * 255 ** 2) ** 0.5) ** 2) ** 0.5
        # the euclidean distance of the 3-D array.
        d = np.linalg.norm(self.target_im - im)
        self.fitness = (self.target_im.size * 195075) ** 0.5 - d

