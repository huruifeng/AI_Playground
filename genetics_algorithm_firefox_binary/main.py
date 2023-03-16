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
# https://zhuanlan.zhihu.com/p/373939677
# https://www.keyangou.com/topic/1143
##------------------------------------------
import os

from skimage import io
from matplotlib import pyplot as plt

import numpy as np
import skimage

from datetime import datetime

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
        chr_l= int(3 * (self.x_code_length + self.y_code_length) + 8*4)
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

    def select(self, pop, fit):
        fit = fit - np.min(fit)
        fit = fit + np.max(fit) / 2 + 0.01
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fit / fit.sum())
        children = []
        for i in idx:
            children.append(pop[i].copy())
        return np.array(children)

    def crossover(self, pop):
        for i in range(0, self.pop_size, 2):
            if np.random.random() < self.cross_rate:
                ## Uniform crossover
                A = pop[i]
                B = pop[i + 1]
                crossover_index = np.random.rand(1, self.code_length) < 0.5
                crossover_index2 = ~crossover_index
                A_new = np.uint8(np.logical_or(np.logical_and(A, crossover_index), np.logical_and(B, crossover_index2)))
                B_new = np.uint8(np.logical_or(np.logical_and(A, crossover_index2), np.logical_and(B, crossover_index)))
                pop[i] = A_new
                pop[i + 1] = B_new
        return pop

    def mutate(self, pop):
        mutation_index = np.random.rand(self.pop_size, self.code_length) < self.mutate_rate
        pop = np.logical_xor(pop, mutation_index)
        return pop

    def evolve(self):
        pop = np.random.randint(0, 2, (self.pop_size, self.code_length), dtype=np.int32)
        pop_fit = self.cal_fitness(pop)
        for g in range(self.generations):
            best_idx = np.argmax(pop_fit)
            best_fit = pop_fit[best_idx]
            best_indiv = pop[best_idx]
            print(f'Generation:{g:0>5}: Best fitness:{best_fit} - Similarity:{best_fit/self.max_diff}')
            if g % 20 == 0:
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

ga = GA()
ga.evolve()







