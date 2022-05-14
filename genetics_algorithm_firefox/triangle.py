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

from PIL import Image, ImageDraw
import numpy as np
class Color(object):
    '''
    Define color: RGBa
    '''
    def __init__(self):
        self.r = np.random.randint(0, 255)
        self.g = np.random.randint(0, 255)
        self.b = np.random.randint(0, 255)
        self.a = np.random.randint(100,160)

class Triangle(object):
    '''
    Define Triangle
    properties：
            ax,ay,bx,by,cx,cy: The location od the three vertexes
            color 			 : color of the triangle
            img_t			 : image of the triangle
    Method:
            draw_t(self, size=(256, 256))   : draw the triangle
    '''

    def __init__(self, size=(256, 256)):
        self.ax = np.random.randint(0, size[0])
        self.ay = np.random.randint(0, size[1])
        self.bx = np.random.randint(0, size[0])
        self.by = np.random.randint(0, size[1])
        self.cx = np.random.randint(0, size[0])
        self.cy = np.random.randint(0, size[1])
        self.color = Color()
        self.size = size

        self.img_t = None

    def draw_it(self):
        self.img_t = Image.new('RGBA', self.size)
        draw = ImageDraw.Draw(self.img_t)
        draw.polygon([(self.ax, self.ay),
                      (self.bx, self.by),
                      (self.cx, self.cy)],
                     fill=(self.color.r, self.color.g, self.color.b, self.color.a))
        return self.img_t
