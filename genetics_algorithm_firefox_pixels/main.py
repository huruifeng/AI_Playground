##########################
## Ruifeng Hu
## 05-11-2022
## Lexington, MA
## hurufieng.cn@hotmail.com
##########################

##--------------------------------------------
# Using the Genetic Algorithm(GA) to draw
#   the FireFox logo
# https://www.cfanz.cn/resource/detail/ZzmQvALXBMEDN
##------------------------------------------
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage


def cal_fitness(pop):
    ## Calculate the fitness

    fitness = np.zeros((pop.shape[0],))
    for i in range(pop.shape[0]):
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

    return np.array(pop)

def selection(pop,fit):
    pop_size = pop.shape[0]
    fit = fit - np.min(fit)
    fit = fit + np.max(fit) / 2 + 0.01
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=fit / fit.sum())
    children = []
    for i in idx:
        children.append(pop[i].copy())
    return np.array(children)

def corssover(pop,cross_rate):
    new_pop = []
    for i in range(0,pop.shape[0],2):
        p1 = pop[i]
        p2 = pop[i+1]
        x, y, z = target_img.shape

        new_p1 = p1.copy()
        new_p2 = p2.copy()

        if np.random.random() < cross_rate:
            x1_idx = np.random.choice(range(x), int(x / 2),replace=False)
            y1_idx = np.random.choice(range(y), int(y / 2),replace=False)

            x2_idx = list(set(range(x)) - set(x1_idx))
            y2_idx = list(set(range(y)) - set(y1_idx))

            new_p1[x2_idx,y2_idx] = p2[x2_idx,y2_idx]
            new_p2[x1_idx, y1_idx] = p1[x1_idx, y1_idx]

        new_pop.append(new_p1)
        new_pop.append(new_p2)
    return np.array(new_pop)

def mutation(pop,mutation_rate):
    width = 5
    new_pop = []
    x, y, z = target_img.shape
    for i in range(pop.shape[0]):
        indv_m = pop[i].copy()
        m_x = np.random.choice(range(x), int(x * mutation_rate),replace=False)
        m_y = np.random.choice(range(y), int(x * mutation_rate),replace=False)
        for i,j in zip(m_x, m_y):
            channel = np.random.randint(0,z-1)
            center_p = indv_m[i,j][channel]
            sx = list(range(max(0, i-width), min(i+width, x-1)))
            sy = list(range(max(0, j-width), min(j+width, y-1)))
            mtemp = indv_m[sx]
            normal_rgba = np.random.normal(center_p,.01, size=mtemp[:,sy].shape[:2])
            normal_rgba[normal_rgba>1] = 1
            normal_rgba[normal_rgba<0] = 0
            mtemp[:,sy, channel] = normal_rgba
            indv_m[sx] = mtemp
        new_pop.append(indv_m)
    return np.array(new_pop)

def evol():
    pop_size = 20
    cross_rate = 0.99
    mutation_rate = 0.05
    generations = 50000

    pop = init_pop(pop_size)
    pop_fit = cal_fitness(pop)
    for g in range(generations):
        best_idx = np.argmin(pop_fit)
        best_fit = pop_fit[best_idx]
        best_indiv = pop[best_idx]
        print(f'Generation:{g:0>5}: Best fitness:{best_fit} - Similarity:{1 - best_fit/max_diff}')
        if g % 100 == 0:
            skimage.io.imsave(os.path.join(results_folder, f'{g:0>8}.png'),(best_indiv*255).astype(np.uint8))

        chd = selection(pop, pop_fit)
        chd = corssover(chd,cross_rate)
        chd = mutation(chd,mutation_rate)
        chd_fit = cal_fitness(chd)

        # if the best child is NOT better than the best parent,
        # keep the best parent in the children population (replace the worest child)
        best_child = np.min(chd_fit)
        if best_child > best_fit:
            rm_index = np.argmax(chd_fit)
            chd[rm_index] = best_indiv
            chd_fit[rm_index] = best_fit

        pop = chd
        pop_fit = chd_fit


target_image = "data/firefox_768.png"
results_folder = "results/firefox_768"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

im = skimage.io.imread(target_image)
if im.shape[2] == 4:
    im = skimage.color.rgba2rgb(im)
# im = im/255.0

target_img = skimage.transform.resize(im, (256,256), mode='reflect', preserve_range=True)
skimage.io.imsave(os.path.join(results_folder, f'target.png'), target_img)
max_diff = target_img.size*4

evol()

