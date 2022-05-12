##########################
## Ruifeng Hu
## 05-10-2022
## Lexington, MA
## hurufieng.cn@hotmail.com
##########################

##--------------------------------------------
# Using the Genetic Algorithm(GA) to find out
#   the maximum value  of the function f(x)=x+10sin(5x)+7cos(4x)
#   where x in the range [0, 9]
#
#   Ground Truth:  f(x=7.857) = 24.8553627239529
#

##--------------------------------------------
import matplotlib
import  matplotlib.pyplot as plt
import numpy as np

x_max = 9.0  # max value of x
x_min = 0.0  # min calue of 0

## Traget function
def f(x):
    return x+10*np.sin(5*x)+7*np.cos(4*x)

def fitness():
    fitness_value = np.zeros((population_size,))
    individual = np.zeros((population_size,))

    # convert binary to decimal
    for i in range(population_size):
        for j in range(chromosome_size):
            individual[i] = individual[i] + population[i, j] * (2 ** j)

    # calculate fitness
    for i in range(population_size):
        x = x_min + (x_max-x_min)*float(individual[i])/(2**chromosome_size-1) # convert to [x_min,x_max]
        fitness_value[i] = f(x)

    return individual,fitness_value

def selection(index_ranked):
    population_ranked = population[index_ranked,:]
    selected = population_ranked[-int(population_size/2):,:]
    return selected

def crossover(selected_population):
    new_population = np.zeros((population_size,chromosome_size))
    for i in range(0,population_size,2):
        parent_1 = population[i,:]
        r = np.random.randint(int(population_size/2))
        parent_2 = selected_population[r,]

        cross_position = round(np.random.rand() * chromosome_size)
        if (cross_position == 0):
            continue

        # exchange the genes after cross_position
        new_population[i,:] = parent_1
        new_population[i+1,:] = np.concatenate([parent_1[:-cross_position] , parent_2[-cross_position:]])
    return new_population

def mutation(new_population,mutate_rate):
    for i in range(population_size):
        if np.random.rand() < mutate_rate:
            mutate_position = int(np.random.rand() * chromosome_size) # muation location
            if mutate_position == 0:
                continue
            new_population[i, mutate_position] = 1 - new_population[i, mutate_position]
    return new_population

def run_GA(population,population_size,chromosome_size,generation_size,mutate_rate):
    ## Record the generations
    fitness_best = []  # best fitness in each generation
    individual_best = []  # best individual in each generation
    for g in range(generation_size):
        individual, fitness_value = fitness() # calculate the fitness of the individuals in current generation

        ## rank the indiidual based on the fitness_values in a decrease order
        index_ranked = np.argsort(fitness_value)
        individual_ranked = individual[index_ranked] # Rank the individual based on the fitness value
        fitness_ranked = fitness_value[index_ranked]
        fitness_best.append(fitness_ranked[-1])
        x = x_min + (x_max-x_min)*float(individual_ranked[-1])/(2**chromosome_size-1)
        individual_best.append(x)
        print(f'Generation:{g:3d} - Best individual: {x:.4f}, Best fitness: {fitness_ranked[0]:.4f}')

        selected_population = selection(index_ranked) # population selection

        new_population = crossover(selected_population) # chromosome crossover
        population = mutation(new_population, mutate_rate) # gene mutations
    return fitness_best,individual_best

if __name__ == "__main__":
    ## GA parameters
    population_size = 100  # population size
    chromosome_size = 16  # number of gene on chrome.
    generation_size = 500  # generation number
    mutate_rate = 0.001  # mutation rate

    # initialize the population
    population = np.zeros((population_size, chromosome_size))
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i, j] = np.random.randint(0, 2)

    fitness_best, individual_best = run_GA(population,population_size,chromosome_size,generation_size,mutate_rate)

    ## plot ground truth
    fig = plt.figure()
    x = np.linspace(0, 10, 1000)
    y = x+10*np.sin(5*x)+7*np.cos(4*x)
    plt.plot(x, y)
    plt.show()














