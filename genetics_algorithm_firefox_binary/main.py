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
##------------------------------------------
import gc
import os

import numpy as np
from PIL import Image, ImageDraw

def bin2dec(x):
    # Converting binary into decimal
    dec_num = 0
    m = 1
    for digit in x:
        digit = int(digit)
        dec_num = dec_num + (digit * m)
        m = m * 2
    return dec_num

def DrawImg(individual,img_size,x_length,y_length): # Draw the image
    ind_img = Image.new('RGBA', img_size)
    # draw = ImageDraw.Draw(best_img)
    # draw.polygon([(0, 0), (0, img_size[1]), img_size, (img_size[0], 0)], fill=(255, 255, 255, 255))
    ## decode the chromosome
    p = 0
    for i in range(100):
        ## each triangle
        ax, ay = individual[p:p + x_length], individual[p + x_length:p + x_length + y_length]
        p = p + x_length + y_length
        bx, by = individual[p:p + x_length], individual[p + x_length:p + x_length + y_length]
        p = p + x_length + y_length
        cx, cy = individual[p:p + x_length], individual[p + x_length:p + x_length + y_length]
        p = p + x_length + y_length

        r,g,b,a = individual[p:p + 8], individual[p + 8:p + 16],individual[p+16:p + 24], individual[p + 24:p + 32]
        p=p+32

        ax,ay,bx,by,cx,cy = bin2dec(ax),bin2dec(ay),bin2dec(bx),bin2dec(by),bin2dec(cx),bin2dec(cy)
        r,g,b,a = bin2dec(r), bin2dec(g), bin2dec(b), bin2dec(a)

        img_t = Image.new('RGBA', img_size)
        draw = ImageDraw.Draw(img_t)
        draw.polygon([(ax, ay),(bx, by),(cx, cy)], fill=(r, g, b, a))

        ind_img = Image.alpha_composite(ind_img, img_t)

    return ind_img

def save_img(indvi,img_size,x_length,y_length,generation_n,results_folder):
    best_img = DrawImg(indvi,img_size,x_length,y_length)
    # draw = ImageDraw.Draw(best_img)
    # draw.polygon([(0, 0), (0, img_size[1]), img_size, (img_size[0], 0)], fill=(255, 255, 255, 255))
    best_img.save(os.path.join(results_folder, str(generation_n)+".png"))


def TargetFunc(individual,x_length,y_length):
    current_img = DrawImg(individual,img_size,x_length,y_length)
    sum = 0
    arrs = [np.array(x) for x in list(current_img.split())]  # split  intto R,G,B,A channel
    for i in range(4):
        sum += np.sum(np.abs(arrs[i] - target_pixels[i]))
    fitness = 1-sum/float(img_size[0]* img_size[1]*256*4)
    return fitness

def cal_fitness(population, x_length,y_length):
    n = population.shape[0]
    fitness_arr = np.zeros((n,))
    for i in range(n):
        fitness_arr[i] = TargetFunc(population[i], x_length, y_length)
    return fitness_arr


def selection(population,num, all_fitness,code_length):
    fitness_sum = np.sum(all_fitness)
    accP = np.cumsum(all_fitness / fitness_sum)
    n = num
    selected_population = np.zeros((n, code_length), dtype=np.int32)

    hasSelected = []
    for j in range(n):
        while 1:
            matrix = np.where(accP >= np.random.rand())
            if matrix[0][0] in hasSelected:
                continue
            hasSelected.append(matrix[0][0])
            break
        selected_population[j, :] = population[hasSelected[j], :]

    return selected_population

def crossover_mutation(population, kept_population,num,crossover_rate,variation_rate):
    pair_matrix = np.random.permutation(num)
    for j in range(int(num/2)):
        A = np.uint8(kept_population[pair_matrix[2 * j + 1], :])
        B = np.uint8(kept_population[pair_matrix[2 * j], :])
        if np.random.rand() < crossover_rate:
            crossover_index = np.random.rand(1,code_length) < 0.5
            crossover_index2 = ~crossover_index
            kept_population = np.concatenate([kept_population,
                                              np.uint8(np.logical_or(np.logical_and(A, crossover_index),
                                                                     np.logical_and(B, crossover_index2)))]
                                             )
            kept_population = np.concatenate([kept_population,
                                              np.uint8( np.logical_or(np.logical_and(A, crossover_index2),
                                                                      np.logical_and(B, crossover_index)))]
                                             )
    n = kept_population.shape[0]
    mutation_index = np.random.rand(n, code_length) < variation_rate
    kept_population = np.uint8(np.logical_xor(kept_population, mutation_index))

    p = np.argmax(all_fitness)
    kept_population = np.concatenate([kept_population,population[[p],]])
    all_fitness[p] = -1 # 置为负数，表示该个体已被选中
    p = np.argmax(all_fitness)
    kept_population = np.concatenate([kept_population,population[[p],:]])

    return kept_population


######################
if __name__ == "__main__":
    target_image = "data/firefox_768.png"
    results_folder = "results/firefox_768"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder,exist_ok=True)
    img = Image.open(target_image).resize((256, 256)).convert('RGBA')
    img_size = img.size
    target_pixels = [np.array(x) for x in list(img.split())]

    num = 40 #population size
    chromosomes_n = 100
    crossover_rate = 0.6
    variation_rate = 0.002

    x_code_length = int(np.ceil(np.log2(img_size[0]))) # length of encoded x
    y_code_length = int(np.ceil(np.log2(img_size[1]))) # length of encoded x
    point_code_lenght = x_code_length + y_code_length
    color_code_length = 4*8 # R,G,B, A
    chromosomes_l = int(3*point_code_lenght + color_code_length)
    code_length = int(chromosomes_n * chromosomes_l) # code length of one chromosome

    # Initialization
    population = np.random.randint(0,2,(num,code_length),dtype=np.int32)

    generation_n = 0
    while True:
        # Calculate fitness
        all_fitness = cal_fitness(population, x_code_length,y_code_length)
        print(f'Generation:{generation_n:3d} - '
              f'Top individual: {np.argmax(all_fitness)}, '
              f'Best fitness: {np.max(all_fitness)}')
        if generation_n % 100 == 0:
            best_indvi = population[np.argmax(all_fitness)]
            save_img(best_indvi,img_size,x_code_length,y_code_length,generation_n,results_folder)

        # Selection
        selected_population = selection(population, num, all_fitness,code_length)

        # Crossover and mutation
        new_population = crossover_mutation(population,selected_population,num,crossover_rate,variation_rate)

        del population
        gc.collect()
        population = new_population

        generation_n += 1



