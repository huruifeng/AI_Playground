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
    shape = [256, 256,3]
    def __init__(self, x, y, color, alpha):
        self.x = x
        self.y = y
        self.color = color
        self.alpha = alpha
        self.img_t=None

class Individual:
    shape = [256, 256,3]
    target_im = None

    def __init__(self):
        self.triangles = []
        self.fitness = 0
        self.img = None

    def copy(self):
        return copy.deepcopy(self)

    def draw_im(self):
        self.img = np.ones(self.shape, dtype=np.uint8) * 255
        for t in self.triangles:
            xx, yy = skimage.draw.polygon(t.x, t.y)
            skimage.draw.set_color(self.img, (xx, yy), t.color, t.alpha)
        return self.img

    def calc_fitness(self):
        im = self.draw_im()

        # the euclidean distance of the 3-D array.
        d = np.linalg.norm(np.where(self.target_im > im, self.target_im - im, im - self.target_im))

        # The bigest diatance (self.target_im.size * ((3 * 255 ** 2) ** 0.5) ** 2) ** 0.5
        self.fitness = np.sqrt(self.target_im.size * 195075) - d

