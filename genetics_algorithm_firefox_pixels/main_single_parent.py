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
import gc
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage


def cal_fitness(pop):
    ## Calculate the fitness

    pop_size = len(pop)
    fitness = np.zeros((pop_size,))
    for i in range(pop_size):
        fit = np.sum(np.square((pop[i] - target_img)))
        # fit = np.linalg.norm(np.where(pop[i] > target_img,pop[i] - target_img, target_img - pop[i]))
        fitness[i] = fit
    return fitness


def init_indiv(x, y, z):
    """Initialize one individual
    :param x: image height
    :param y: image width
    :param z: alpha channel, 4 - rgba, 3 - rgb
    """
    indiv = np.random.random((x, y, z))
    return indiv

def init_pop(pop_size=20):
    """Initialize the population
    :param target:
    :param p: 群体大小
    :param jobs: 进程数
    """
    x,y,z = target_img.shape
    pop = []
    for i in range(pop_size):
        pop.append(init_indiv(x,y,z))

    return pop

def mutation(parent,mutation_rate):
    width = 5

    x, y, z = target_img.shape
    indv_m =parent.copy()
    m_x = np.random.choice(range(x), int(x * mutation_rate),replace=False)
    m_y = np.random.choice(range(y), int(x * mutation_rate),replace=False)
    for i,j in zip(m_x, m_y):
        channel = np.random.randint(0,3) ## OR: z-1
        center_p = indv_m[i,j][channel]
        sx = list(range(max(0, i-width), min(i+width, x-1)))
        sy = list(range(max(0, j-width), min(j+width, y-1)))
        mtemp = indv_m[sx]
        normal_rgba = np.random.normal(center_p,.01, size=mtemp[:,sy].shape[:2])
        normal_rgba[normal_rgba>1] = 1
        normal_rgba[normal_rgba<0] = 0
        mtemp[:,sy, channel] = normal_rgba
        indv_m[sx] = mtemp

    return indv_m

def evol():
    pop_size = 20
    mutation_rate = 0.01
    generations = 50000

    pop = init_pop(pop_size)
    pop_fit = cal_fitness(pop)

    # Select the best one
    parent = pop[np.argmin(pop_fit)]
    parent_fit = np.min(pop_fit)
    del pop
    gc.collect()

    # Start GA
    g = 0
    while g < generations:
        childList = []
        # generate individuals from previous generation
        for j in range(pop_size):
            indiv_j = mutation(parent,mutation_rate)
            childList.append(indiv_j)

        child_fit = cal_fitness(childList)

        # Select the best one
        child = childList[np.argmin(child_fit)]
        child_best_fit = np.min(child_fit)
        del childList
        gc.collect()

        print('Generation:%8d - Parent: %7d, Best child: %7d, Similarity: %.6f'
              % (g, parent_fit, child_best_fit, 1-parent_fit / max_diff))
        ## replace the parent by child if child is better
        if parent_fit > child_best_fit:
            parent = child
            parent_fit = child_best_fit

        child = None
        if g % 20 == 0:
            skimage.io.imsave(os.path.join(results_folder, f'{g:0>8}.png'), (parent*255).astype(np.uint8))

        ## next generation
        g += 1


target_image = "data/firefox_768.png"
results_folder = "results/firefox_768_single"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

im = skimage.io.imread(target_image)

if im.shape[2] == 4:
    im = skimage.color.rgba2rgb(im)

# im = im/255.0

target_img = skimage.transform.resize(im, (512,512), mode='reflect', preserve_range=True)
skimage.io.imsave(os.path.join(results_folder, f'target.png'), target_img)
max_diff = target_img.size*4

evol()

