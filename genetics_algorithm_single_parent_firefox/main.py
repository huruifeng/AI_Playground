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

import matplotlib
import  matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from triangle import Triangle, Canvas

target_image = "data/firefox_768.png"
results_folder = "results/firefox_768"
if not os.path.exists(results_folder):
    os.makedirs(results_folder,exist_ok=True)
img = Image.open(target_image).resize((256, 256)).convert('RGBA')
img_size = img.size


if __name__ == "__main__":
    ## GA parameters
    population_size = 20  # population size
    chromosome_size = 100  # number of gene on chrome.
    generation_size = 5000000   # generation number
    mutate_rate = 0.6    # mutation rate
    keep_rate = 0.5

    Canvas.target_pixels = [np.array(x) for x in list(img.split())]
    # Initialize population. Generate 20 individuals
    parentList = []
    for i in range(population_size):
        print('Generating parent:%d' % (i))
        parentList.append(Canvas())
        parentList[i].add_triangles(chromosome_size)
        parentList[i].calc_match_rate()

    # Select the best one
    parent = sorted(parentList, key=lambda x: x.match_rate)[0]
    del parentList
    gc.collect()

    # Start GA
    i = 0
    while i < generation_size:
        childList = []
        # generate 20 individuals from previous generation
        for j in range(population_size):
            childList.append(Canvas())
            childList[j].mutate_from_parent(parent)
            childList[j].calc_match_rate()

        # Select the best one
        child = sorted(childList, key=lambda x: x.match_rate)[0]
        del childList
        gc.collect()

        print('Generation:%10d - parent rate %10d, best child rate %10d' % (i, parent.match_rate, child.match_rate))
        ## replace the parent by child if child is better
        parent = parent if parent.match_rate < child.match_rate else child

        child = None
        if i % 100 == 0:
            parent.draw_it(i,results_folder)

        ## next generation
        i += 1










