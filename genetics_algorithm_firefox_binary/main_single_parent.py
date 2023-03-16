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
# Using 100 triangles to draw a FireFox logo,
#   each triangle is a chromosome,
#   and each chromosome has 10 genes:
#       ax,ay, bx,by, cx,cy, r,g,b,a
#
# https://www.pianshen.com/article/7818211114/
##------------------------------------------
import gc
import os

from PIL import Image, ImageDraw
from skimage import io
from matplotlib import pyplot as plt

import numpy as np
import skimage


#######
target_image = "data/firefox_768.png"
target_shape = (128,128,3)

results_folder = "results/firefox_768_single"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)


def bin2dec(x):
    # Converting binary into decimal
    dec_num = 0
    m = 1
    for digit in x:
        digit = int(digit)
        dec_num = dec_num + (digit * m)
        m = m * 2
    return dec_num

class GA:
    def __init__(self):
        self.pop_size = 40
        self.chr_n = 100
        self.generations = 1000000
        self.cross_rate = 0.6
        self.mutate_rate = 0.01

        im = skimage.io.imread(target_image)
        if im.shape[2] == 4:
            im = skimage.color.rgba2rgb(im)
            im = (255 * im).astype(np.uint8)

        self.target_im = skimage.transform.resize(
            im, target_shape, mode='reflect', preserve_range=True).astype(np.uint8)

        skimage.io.imsave(os.path.join(results_folder, f'target.png'), self.target_im)

        self.max_diff = np.sqrt(self.target_im.size * 255.0 * 255.0 * 3)

        ## code: ax,ay,bx,by,cx,cy,r,g,b,a
        self.x_code_length = int(np.ceil(np.log2(target_shape[0])))  # length of encoded x
        self.y_code_length = int(np.ceil(np.log2(target_shape[1])))  # length of encoded y
        chr_l = int(3 * (self.x_code_length + self.y_code_length) + 8 * 4)
        self.code_length = int(self.chr_n * chr_l)

    def draw_ff(self, indiv):
        im = np.ones(self.target_im.shape, dtype=np.uint8) * 255
        p = 0
        for i in range(100):
            ## each triangle
            ax, ay = indiv[p:p + self.x_code_length], indiv[p + self.x_code_length:p + self.x_code_length + self.y_code_length]
            p = p + self.x_code_length + self.y_code_length
            bx, by = indiv[p:p + self.x_code_length], indiv[p + self.x_code_length:p + self.x_code_length + self.y_code_length]
            p = p + self.x_code_length + self.y_code_length
            cx, cy = indiv[p:p + self.x_code_length], indiv[p + self.x_code_length:p + self.x_code_length + self.y_code_length]
            p = p + self.x_code_length + self.y_code_length

            r, g, b, a = indiv[p:p + 8], indiv[p + 8:p + 16], indiv[p + 16:p + 24], indiv[p + 24:p + 32]
            p = p + 32

            ax, ay = bin2dec(ax), bin2dec(ay)  ##Modify: ax, ay = min(255,bin2dec(ax)),min(255, bin2dec(ay))
            bx, by = bin2dec(bx), bin2dec(by)
            cx, cy = bin2dec(cx), bin2dec(cy)
            r, g, b, a = bin2dec(r), bin2dec(g), bin2dec(b), bin2dec(a)/510

            xx, yy = skimage.draw.polygon([ax,bx,cx],[ay,by,cy])
            skimage.draw.set_color(im, (xx, yy), [r,g,b], a)
        return im

    def f(self, indiv):
        im = self.draw_ff(indiv)

        # the euclidean distance of the 3-D array.
        d = np.linalg.norm(np.where(self.target_im > im, self.target_im - im, im - self.target_im))

        # The bigest diatance (self.target_im.size * ((3 * 255 ** 2) ** 0.5) ** 2) ** 0.5
        return np.sqrt(self.target_im.size * 195075) - d

    def cal_fitness(self,population):
        fitness = np.zeros(self.pop_size)
        for i, indiv in enumerate(population):
            fitness[i] = self.f(indiv)
        return fitness

    def mutate(self, indiv):
        indiv = indiv.copy()
        mutation_index = np.random.rand(self.code_length) < self.mutate_rate
        indiv = np.logical_xor(indiv, mutation_index)
        return indiv

    def evolve(self):
        pop = np.random.randint(0, 2, (self.pop_size, self.code_length), dtype=np.int32)
        pop_fit = self.cal_fitness(pop)

        # Select the best one
        parent = pop[np.argmax(pop_fit)]
        parent_fit = np.max(pop_fit)
        del pop
        gc.collect()

        # Start GA
        g = 0
        while g < self.generations:
            childList = []
            # generate individuals from previous generation
            for j in range(self.pop_size):
                indiv_j = self.mutate(parent)
                childList.append(indiv_j)

            child_fit = self.cal_fitness(childList)

            # Select the best one
            child = childList[np.argmax(child_fit)]
            child_best_fit = np.max(child_fit)
            del childList
            gc.collect()

            print('Generation:%8d - Parent: %7d, Best child: %7d, Similarity: %.6f'
                  % (g, parent_fit, child_best_fit, parent_fit/self.max_diff))
            ## replace the parent by child if child is better
            if parent_fit < child_best_fit:
                parent = child
                parent_fit = child_best_fit

            child = None
            if g % 20 == 0:
                skimage.io.imsave(os.path.join(results_folder, f'{g:0>8}.png'), self.draw_ff(parent))

            ## next generation
            g += 1

ga = GA()
ga.evolve()







