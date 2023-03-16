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
import os

from PIL import Image, ImageDraw
from skimage import io
from matplotlib import pyplot as plt

import numpy as np
import skimage
from Individual import Individual, Triangle

from datetime import datetime

class GA:
    def __init__(self):
        self.pop_size = 80
        self.gene_n = 100
        self.generations = 1000000
        self.cross_rate = 0.6
        self.mutate_rate = 0.008

        im = skimage.io.imread(target_image)
        if im.shape[2] == 4:
            im = skimage.color.rgba2rgb(im)
            im = (255 * im).astype(np.uint8)

        self.target_im = skimage.transform.resize(
            im, target_shape, mode='reflect', preserve_range=True).astype(np.uint8)
        skimage.io.imsave(os.path.join(results_folder, f'target.png'), self.target_im)

        self.max_diff = np.sqrt(self.target_im.size * 255.0 * 255.0 * 3)

    def generate_pop(self):
        pop = np.empty((self.pop_size,),dtype=Individual)
        for i in range(self.pop_size):
            indiv_i = Individual()
            for j in range(self.gene_n):
                x = np.random.randint(0, self.target_im.shape[0], 3, dtype=np.uint8)
                y = np.random.randint(0, self.target_im.shape[1], 3, dtype=np.uint8)
                color = np.random.randint(0, 256, 3)
                alpha = np.random.random() * 0.45
                indiv_i.triangles.append(Triangle(x, y, color, alpha))
            pop[i] = indiv_i
        return pop

    def draw_ff(self, indiv):
        im = np.ones(self.target_im.shape, dtype=np.uint8) * 255
        for t in indiv.triangles:
            xx, yy = skimage.draw.polygon(t.x,t.y)
            skimage.draw.set_color(im, (xx, yy), t.color, t.alpha)
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

    def select(self, pop, fit):
        fit = fit - np.min(fit)
        fit = fit + np.max(fit) / 2 + 0.01
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fit / fit.sum())
        children = []
        for i in idx:
            children.append(pop[i].copy())
        return children

    def crossover(self, pop):
        for i in range(0, self.pop_size, 2):
            if np.random.random() < self.cross_rate:
                A = pop[i]
                B = pop[i + 1]
                p = np.random.randint(1, self.gene_n)
                A.triangles[p:], B.triangles[p:] = B.triangles[p:], A.triangles[p:]
                pop[i] = A
                pop[i + 1] = B
        return pop

    def mutate(self, pop):
        for indiv in pop:
            for t in indiv.triangles:
                if np.random.random() < self.mutate_rate:
                    t.x = np.random.randint(0, self.target_im.shape[0], 3, dtype=np.uint8)
                    t.y = np.random.randint(0, self.target_im.shape[1], 3, dtype=np.uint8)
                    t.color = np.random.randint(0, 256, 3)
                    t.alpha = np.random.random() * 0.45
        return pop

    def evolve(self):
        pop = self.generate_pop()
        pop_fit = self.cal_fitness(pop)
        for g in range(self.generations):
            best_idx = np.argmax(pop_fit)
            best_fit = pop_fit[best_idx]
            best_indiv = pop[best_idx]
            print(f'Generation:{g:0>5}: Best fitness:{best_fit} - Similarity:{best_fit/self.max_diff}')
            if g % 10 == 0:
                skimage.io.imsave(os.path.join(results_folder, f'{g:0>8}.png'), ga.draw_ff(best_indiv))

            chd = self.select(pop, pop_fit)
            chd = self.crossover(chd)
            chd = self.mutate(chd)
            chd_fit = self.cal_fitness(chd)

            # if the best child is NOT better than the best parent,
            # keep the best parent in the children population (replace the worest child)
            best_child = np.max(chd_fit)
            if best_child < best_fit:
                rm_index = np.argmin(chd_fit)
                chd[rm_index] = best_indiv
                chd_fit[rm_index] = best_fit

            pop = chd
            pop_fit = chd_fit

#######
target_image = "data/firefox_768.png"
target_shape = (128,128,3)

results_folder = "results/firefox_768"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

Individual.shape = target_shape
Triangle.shape = target_shape

ga = GA()
ga.evolve()







