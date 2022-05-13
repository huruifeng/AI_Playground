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

import matplotlib
import  matplotlib.pyplot as plt
import numpy as np

true_image = "data/firefox_252.png"
results_folder = "results/firefox_252"
if not os.path.exists(results_folder):
    os.makedirs(results_folder,exist_ok=True)

## Traget function
def f(x_true, x):
    err = x_true - x
    return err

def fitness(population,population_size,chromosome_size):
    fitness_value = np.zeros((population_size,))
    individual = np.zeros((population_size,))

    return individual,fitness_value

def selection(population_ranked,population_size):
    selected = []
    return selected

def crossover_mutation(selected_population,population_size,chromosome_size,mutate_rate):
    new_population = []
    return new_population


def run_GA(population,population_size,chromosome_size,generation_size,mutate_rate):
    for g in range(generation_size):
        # calculate the fitness of the individuals in current generation
        individual, fitness_value = fitness(population,population_size,chromosome_size)

        ## rank the indiidual based on the fitness_values in a decrease order
        index_ranked = np.argsort(fitness_value)
        population_ranked = population[index_ranked,:]

        # population selection
        selected_population = selection(population_ranked,population_size)

        # generate new population
        population = crossover_mutation(selected_population,population_size,chromosome_size,mutate_rate) # chromosome crossover


if __name__ == "__main__":
    ## GA parameters
    population_size = 100  # population size
    chromosome_size = 100  # number of gene on chrome.
    generation_size = 20000   # generation number
    mutate_rate = 0.001    # mutation rate

    # initialize the population
    population = []

    # run GA
    run_GA(population,population_size,chromosome_size,generation_size,mutate_rate)














