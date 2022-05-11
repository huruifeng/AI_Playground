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
import numpy as np

## Traget function
def f(x):
    return x+10*np.sin(5*x)+7*np.cos(4*x)

## GA parameters
population_size = 100      # population size
chromosome_size = 16       # number of gene on chrome.
generation_size = 200      # generation number
cross_rate = 0.4           # crossover rate
mutate_rate = 0.01         # mutation rate

# initialize the population
population = np.zero((population_size,chromosome_size))
for i in range(population_size):
    for j in range(chromosome_size):
        population[i,j] = np.random.randint(0,2)

x_max = 9.0        # max value of x
x_min = 0.0        # min calue of 0

## Record the generations
fitness_best = []    # best fitness in each generation
individual_best=[]   # best individual in each generation

def fitness():
    fitness_value = np.zero((population_size,))
    individual = np.zeros((population_size,))

    # convert binary to decimal
    for i in range(population_size):
        for j in range(chromosome_size):
            individual[i] =   population[i,j] + 2**j

    # calculate fitness
    for i in range(population_size):
        x = x_min + (x_max-x_min)*float(individual[i])/(2**chromosome_size-1) # convert to [x_min,x_max]
        fitness_value[i] = f(x)

    return individual,fitness_value

def selection(index_ranked):
    population_ranked = population[index_ranked]


for g in range(generation_size):
    individual, fitness_value = fitness() # calculate the fitness of the individuals in current generation

    ## rank the indiidual based on the fitness_values in a decrease order
    index_ranked = np.argsort(fitness_value)[::-1]
    individual_ranked = individual[index_ranked] # Rank the individual based on the fitness value
    fitness_ranked = fitness_value[index_ranked]
    fitness_best.append(fitness_ranked[0])
    individual_best.append(individual_ranked[0])

    selected_population = selection(index_ranked) # population selection

    cross_population = crossover(selected_population, cross_rate) # chromosome crossover
    mutated_population = mutation(cross_population, mutate_rate) # gene mutations
















