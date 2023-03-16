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
from Individual import Individual, Triangle

#######
target_image = "data/firefox_768.png"
target_shape = (128,128,3)

results_folder = "results/firefox_768_single"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)


Individual.shape = target_shape
Triangle.shape = target_shape

class GA:
    def __init__(self):
        self.pop_size = 30
        self.gene_n = 100
        self.generations = 1000000
        self.mutate_rate = 0.01

        im = skimage.io.imread(target_image)
        if im.shape[2] == 4:
            im = skimage.color.rgba2rgb(im)
            im = (255 * im).astype(np.uint8)

        self.target_im = skimage.transform.resize(
            im, target_shape, mode='reflect', preserve_range=True).astype(np.uint8)
        skimage.io.imsave(os.path.join(results_folder, f'target.png'), self.target_im)

        Individual.target_im = self.target_im

        self.max_diff = np.sqrt(self.target_im.size * 255.0 * 255.0 * 3)

    def generate_pop(self):
        pop = []
        for i in range(self.pop_size):
            print('Generating parent:%d' % (i))
            indiv_i = Individual()
            for j in range(self.gene_n):
                x = np.random.randint(0, self.target_im.shape[0], 3, dtype=np.uint8)
                y = np.random.randint(0, self.target_im.shape[1], 3, dtype=np.uint8)
                color = np.random.randint(0, 256, 3)
                alpha = np.random.random() * 0.45
                indiv_i.triangles.append(Triangle(x, y, color, alpha))
                indiv_i.calc_fitness()

            pop.append(indiv_i)
        return pop

    def mutate(self, indiv):
        indiv = indiv.copy()
        for t in indiv.triangles:
            if np.random.random() < self.mutate_rate:
                t.x = np.random.randint(0, self.target_im.shape[0], 3, dtype=np.uint8)
                t.y = np.random.randint(0, self.target_im.shape[1], 3, dtype=np.uint8)
                t.color = np.random.randint(0, 256, 3)
                t.alpha = np.random.random() * 0.45
        return indiv

    def evolve(self):
        pop = self.generate_pop()

        # Select the best one
        parent = sorted(pop, key=lambda x: x.fitness)[-1]
        del pop
        gc.collect()

        # Start GA
        g = 0
        while g < self.generations:
            childList = []
            # generate individuals from previous generation
            for j in range(self.pop_size):
                indiv_j = self.mutate(parent)
                indiv_j.calc_fitness()
                childList.append(indiv_j)

            # Select the best one
            child = sorted(childList, key=lambda x: x.fitness)[-1]
            del childList
            gc.collect()

            print('Generation:%10d - Parent: %10d, Best child: %10d, Similarity: %.6f'
                  % (g, parent.fitness, child.fitness, parent.fitness/self.max_diff))
            ## replace the parent by child if child is better
            parent = parent if parent.fitness > child.fitness else child

            child = None
            if g % 20 == 0:
                parent.draw_im()
                skimage.io.imsave(os.path.join(results_folder, f'{g:0>8}.png'), parent.img)

            ## next generation
            g += 1

ga = GA()
ga.evolve()







