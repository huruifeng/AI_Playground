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

import matplotlib
import  matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from trangle import Triangle

## Traget function
def f(x_true, x):
    err = x_true - x
    return err

def fitness(population,population_size,chromosome_size):
    fitness_value = np.zeros((population_size,))
    individual = np.zeros((population_size,),dtype=np.int)

    for p in range(population_size):
        ## Draw a canva
        new_img = Image.new('RGBA', img_size)
        draw = ImageDraw.Draw(new_img)
        draw.polygon([(0, 0), (0, img_size[1]), img_size, (img_size[0], 0)], fill=(255, 255, 255, 255))
        match_rate = 0
        for c in range(chromosome_size):
            chromosome_t = population[p,c]
            new_img = Image.alpha_composite(new_img, chromosome_t.draw_it())
        # pixels = [new_img.getpixel((x, y)) for x in range(0, new_img.size[0], 2) for y in range(0, new_img.size[1], 2)]
        # for i in range(0, min(len(pixels), len(target_pixels))):
        #     delta_red   = pixels[i][0] - target_pixels[i][0]
        #     delta_green = pixels[i][1] - target_pixels[i][1]
        #     delta_blue  = pixels[i][2] - target_pixels[i][2]
        #     match_rate +=  delta_red * delta_red + delta_green * delta_green + delta_blue  * delta_blue

        ## Same as above code, but high efficient
        arrs = [np.array(x) for x in list(new_img.split())]  # split  intto R,G,B,A channel
        for i in range(3):
            match_rate += np.sum(np.square(arrs[i] - target_pixels[i]))
        fitness_value[p] = match_rate
        individual[p] = p

    return individual,fitness_value

def selection(population_ranked,population_size,keep_rate):
    selected = population_ranked[:int(population_size*keep_rate), :]
    return selected

def crossover_mutation(selected_population,population_size,chromosome_size,mutate_rate):
    keep_size = selected_population.shape[0]
    new_size = population_size - keep_size
    new_population = np.empty((new_size, chromosome_size), dtype=Triangle)
    for i in range(new_size):
        p1 = np.random.randint(keep_size)
        parent_1 = selected_population[p1, :]
        p2 = np.random.randint(keep_size)
        parent_2 = selected_population[p2,:]

        chroms_n = int(chromosome_size/2)
        chroms_index = np.random.randint(0,chromosome_size,chroms_n)
        chroms_p1 = parent_1[chroms_index]
        chroms_index = np.random.randint(0, chromosome_size, chromosome_size-chroms_n)
        chroms_p2 = parent_2[chroms_index]

        # exchange the genes after cross_position
        child = np.concatenate([chroms_p1, chroms_p2])
        child_mutated = mutation(child,chromosome_size,mutate_rate)

        new_population[i, :] = child_mutated

    new_population = np.concatenate([selected_population,new_population])
    return new_population

def mutation(indvi,chromosome_size,mutate_rate):
    mutate_range = 15
    new_indvi = []
    for t in range(chromosome_size):
        new_t = Triangle(size = img_size)
        ## position
        mutate_a =  True if mutate_rate > np.random.rand() else False
        if mutate_a:
            new_t.ax = min(max(0,indvi[t].ax + np.random.randint(-mutate_range, mutate_range)), new_t.size[0])
            new_t.ay = min(max(0,indvi[t].ay + np.random.randint(-mutate_range, mutate_range)), new_t.size[1])

        mutate_b = True if mutate_rate > np.random.rand() else False
        if mutate_b:
            new_t.bx = min(max(0,indvi[t].bx + np.random.randint(-mutate_range, mutate_range)), new_t.size[0])
            new_t.by = min(max(0,indvi[t].by + np.random.randint(-mutate_range, mutate_range)), new_t.size[1])

        mutate_c = True if mutate_rate > np.random.rand() else False
        if mutate_c:
            new_t.cx = min(max(0,indvi[t].cx + np.random.randint(-mutate_range, mutate_range)), new_t.size[0])
            new_t.cy = min(max(0,indvi[t].cy + np.random.randint(-mutate_range, mutate_range)), new_t.size[1])

        ## Color
        mutate_cr = True if mutate_rate > np.random.rand() else False
        if mutate_cr:
            new_t.color.r = min(max(0,indvi[t].color.r + np.random.randint(-mutate_range, mutate_range)), 255)

        mutate_cg = True if mutate_rate > np.random.rand() else False
        if mutate_cg:
            new_t.color.g = min(max(0,indvi[t].color.g + np.random.randint(-mutate_range, mutate_range)), 255)

        mutate_cb = True if mutate_rate > np.random.rand() else False
        if mutate_cb:
            new_t.color.b = min(max(0,indvi[t].color.b + np.random.randint(-mutate_range, mutate_range)), 255)

        # alpha
        mutate_ca = True if mutate_rate > np.random.rand() else False
        if mutate_ca:
            new_t.color.a = min(max(0,indvi[t].color.a + np.random.randint(-mutate_range, mutate_range)), 255)

        new_indvi.append(new_t)

    return np.array(new_indvi)

def draw_best(indvi,generation_n):
    best_img = Image.new('RGBA', img_size)
    draw = ImageDraw.Draw(best_img)
    draw.polygon([(0, 0), (0, img_size[1]), img_size, (img_size[0], 0)], fill=(255, 255, 255, 255))
    for c in range(chromosome_size):
        chromosome_t = indvi[c] ## the best individual
        best_img = Image.alpha_composite(best_img, chromosome_t.draw_it())
        best_img.save(os.path.join(results_folder, str(generation_n)+".png"))

def run_GA(population,population_size,chromosome_size,generation_size,mutate_rate,keep_rate):
    for g in range(generation_size):
        # calculate the fitness of the individuals in current generation
        individual, fitness_value = fitness(population,population_size,chromosome_size)

        ## rank the indiidual based on the fitness_values in a decrease order
        index_ranked = np.argsort(fitness_value)
        individual_ranked = individual[index_ranked]
        population_ranked = population[index_ranked,:]
        print(f'Generation:{g:3d} - '
              f'Top individual: {individual_ranked[0:10]}, '
              f'Best fitness: {fitness_value[index_ranked][0]}')
        if g % 100 ==0:
            draw_best(population_ranked[0],g)

        # population selection
        selected_population = selection(population_ranked,population_size,keep_rate)

        # generate new population
        population = crossover_mutation(selected_population,population_size,chromosome_size,mutate_rate) # chromosome crossover



target_image = "data/firefox_600.png"
results_folder = "results/firefox_600"
if not os.path.exists(results_folder):
    os.makedirs(results_folder,exist_ok=True)
img = Image.open(target_image).resize((256, 256)).convert('RGBA')
img_size = img.size
# target_pixels = [img.getpixel((x, y)) for x in range(0, img_size[0], 2) for y in range(0, img_size[1], 2)]
target_pixels = [np.array(x) for x in list(img.split())]


if __name__ == "__main__":
    ## GA parameters
    population_size = 60  # population size
    chromosome_size = 100  # number of gene on chrome.
    generation_size = 5000000   # generation number
    mutate_rate = 0.6    # mutation rate
    keep_rate = 0.5

    # initialize the population
    population = np.array([[ Triangle(size=img_size) for i in range(chromosome_size)] for j in range(population_size)])

    # run GA
    run_GA(population,population_size,chromosome_size,generation_size,mutate_rate,keep_rate)














