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
import os

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
        self.a = np.random.randint(95,115)

def mutate_or_not(rate):
    return True if rate > np.random.rand() else False

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
    max_mutate_rate = 0.08
    mid_mutate_rate = 0.3
    min_mutate_rate = 0.8

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


    def mutate_from(self,parent):
        if mutate_or_not(self.max_mutate_rate):
            self.ax =np.random.randint(0, 256)
            self.ay =np.random.randint(0, 256)
        if mutate_or_not(self.mid_mutate_rate):
            self.ax = min(max(0, parent.ax +np.random.randint(-15, 15)), 256)
            self.ay = min(max(0, parent.ay +np.random.randint(-15, 15)), 256)
        if mutate_or_not(self.min_mutate_rate):
            self.ax = min(max(0, parent.ax +np.random.randint(-3, 3)), 256)
            self.ay = min(max(0, parent.ay +np.random.randint(-3, 3)), 256)

        if mutate_or_not(self.max_mutate_rate):
            self.bx =np.random.randint(0, 256)
            self.by =np.random.randint(0, 256)
        if mutate_or_not(self.mid_mutate_rate):
            self.bx = min(max(0, parent.bx +np.random.randint(-15, 15)), 256)
            self.by = min(max(0, parent.by +np.random.randint(-15, 15)), 256)
        if mutate_or_not(self.min_mutate_rate):
            self.bx = min(max(0, parent.bx +np.random.randint(-3, 3)), 256)
            self.by = min(max(0, parent.by +np.random.randint(-3, 3)), 256)

        if mutate_or_not(self.max_mutate_rate):
            self.cx =np.random.randint(0, 256)
            self.cy =np.random.randint(0, 256)
        if mutate_or_not(self.mid_mutate_rate):
            self.cx = min(max(0, parent.cx +np.random.randint(-15, 15)), 256)
            self.cy = min(max(0, parent.cy +np.random.randint(-15, 15)), 256)
        if mutate_or_not(self.min_mutate_rate):
            self.cx = min(max(0, parent.cx +np.random.randint(-3, 3)), 256)
            self.cy = min(max(0, parent.cy +np.random.randint(-3, 3)), 256)

        # color
        if mutate_or_not(self.max_mutate_rate):
            self.color.r =np.random.randint(0, 255)
        if mutate_or_not(self.mid_mutate_rate):
            self.color.r = min(max(0, parent.color.r +np.random.randint(-30, 30)), 255)
        if mutate_or_not(self.min_mutate_rate):
            self.color.r = min(max(0, parent.color.r +np.random.randint(-10, 10)), 255)

        if mutate_or_not(self.max_mutate_rate):
            self.color.g =np.random.randint(0, 255)
        if mutate_or_not(self.mid_mutate_rate):
            self.color.g = min(max(0, parent.color.g +np.random.randint(-30, 30)), 255)
        if mutate_or_not(self.min_mutate_rate):
            self.color.g = min(max(0, parent.color.g +np.random.randint(-10, 10)), 255)

        if mutate_or_not(self.max_mutate_rate):
            self.color.b =np.random.randint(0, 255)
        if mutate_or_not(self.mid_mutate_rate):
            self.color.b = min(max(0, parent.color.b +np.random.randint(-30, 30)), 255)
        if mutate_or_not(self.min_mutate_rate):
            self.color.b = min(max(0, parent.color.b +np.random.randint(-10, 10)), 255)

        # alpha
        if mutate_or_not(self.mid_mutate_rate):
            self.color.a =np.random.randint(95, 115)

class Canvas(object):
    '''
    define the individual
    properties：
            mutate_rate
            size
            target_pixels
    methods：
            add_triangles(self, num=1)
            mutate_from_parent(self, parent)
            calc_match_rate(self
            draw_it(self, i)
    '''


    mutate_rate = 0.01
    size = (256, 256)
    target_pixels = []


    def __init__(self):
        self.triangles = []
        self.match_rate = 0
        self.img = None


    def add_triangles(self, num=1):
        for i in range(0, num):
            triangle = Triangle()
            self.triangles.append(triangle)

    def mutate_from_parent(self, parent):
        flag = False
        for triangle in parent.triangles:
            t = triangle
            if mutate_or_not(self.mutate_rate):
                flag = True
                a = Triangle()
                a.mutate_from(t)
                self.triangles.append(a)
                continue
            self.triangles.append(t)
        if not flag:
            self.triangles.pop()
            t = parent.triangles[np.random.randint(0, len(parent.triangles) - 1)]
            a = Triangle()
            a.mutate_from(t)
            self.triangles.append(a)

    def calc_match_rate(self):
        if self.match_rate > 0:
            return self.match_rate
        self.match_rate = 0
        self.img = Image.new('RGBA', self.size)
        draw = ImageDraw.Draw(self.img)
        draw.polygon([(0, 0), (0, 255), (255, 255), (255, 0)], fill=(255, 255, 255, 255))
        for triangle in self.triangles:
            self.img = Image.alpha_composite(self.img, triangle.draw_it())
        arrs = [np.array(x) for x in list(self.img.split())]
        for i in range(3):
            self.match_rate += np.sum(np.square(arrs[i]-self.target_pixels[i]))

    def draw_it(self, i,path):
        #self.img.save(os.path.join(PATH, "%s_%d_%d_%d.png" % (PREFIX, len(self.triangles), i, self.match_rate)))
        self.img.save(os.path.join(path, "%d.png" % (i)))



