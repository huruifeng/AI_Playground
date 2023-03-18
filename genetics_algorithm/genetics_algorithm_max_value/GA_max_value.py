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
# http://accu.cc/content/ga/sga/
##--------------------------------------------

import matplotlib.animation
import  matplotlib.pyplot as plt
import numpy as np

class GA(object):
    def __init__(self,pop_size=40,
                 generations=30,
                 gene_n = 32,
                 cross_rate=0.6,
                 mutate_rate=0.001,
                 x_bound=[0,9],
                 elitist = True
                 ):
        self.pop_size = pop_size
        self.generations = generations
        self.gene_n = gene_n
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.x_bound = x_bound
        self.elitist = elitist

        self.population = np.random.randint(2, size=(self.pop_size, self.gene_n))

    def f(self,x):
        return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

    def cal_fitness(self):
        temp_size = self.population.shape[0]
        fitness_value = np.zeros((temp_size,))
        individual = np.zeros((temp_size,))

        # convert binary to decimal
        for i in range(temp_size):
            for j in range(self.gene_n):
                individual[i] = individual[i] + self.population[i, j] * (2 ** j)

        # calculate fitness
        for i in range(temp_size):
            x = self.x_bound[0] + (self.x_bound[1]-self.x_bound[0])*float(individual[i])/(2**self.gene_n-1) # convert to [x_min,x_max]
            fitness_value[i] = self.f(x)

        return individual,fitness_value

    def selection(self,fitness_values):
        fitness_sum = np.sum(fitness_values)
        accP = np.cumsum(fitness_values / fitness_sum)
        selected_population = np.zeros((self.pop_size, self.gene_n), dtype=np.int32)
        for j in range(self.pop_size):
            idx = np.where(accP >= np.random.rand())
            selected_population[j, :] = self.population[idx[0][0], :]
        return selected_population

    def crossover(self,selected_population):
        for i in range(0, selected_population.shape[0], 2):
            if np.random.random() < self.cross_rate:
                a = selected_population[i,:]
                b = selected_population[i + 1,:]
                p = np.random.randint(1, self.gene_n)
                a[p:], b[p:] = b[p:], a[p:]
                selected_population[i,:] = a
                selected_population[i + 1,:] = b
        return selected_population

    def mutation(self,crossed_pop):
        mut = np.random.choice(np.array([0, 1]), crossed_pop.shape, p=[1 - self.mutate_rate, self.mutate_rate])
        mut_pop = np.where(mut == 1, 1 - crossed_pop, crossed_pop)
        return mut_pop

    def evolve(self):
        for g in range(self.generations):
            individuals,fitness_values = self.cal_fitness()
            individuals_x =  self.x_bound[0]+(self.x_bound[1]-self.x_bound[0])*(individuals/(2**self.gene_n))
            yield g, individuals_x, fitness_values

            best_indvi = self.population[np.argmax(fitness_values)]
            print(f'Generation:{g:3d} - '
                  f'Top individual: {individuals_x[np.argmax(fitness_values)]:.6f}, '
                  f'Best fitness: {np.max(fitness_values)}')

            selected_pop = self.selection(fitness_values)
            crossed_pop = self.crossover(selected_pop)
            mutated_pop = self.mutation(crossed_pop)
            self.population = mutated_pop
            if self.elitist:
                self.population = np.concatenate([self.population, [best_indvi]])


ga = GA()
gaiter = ga.evolve()

## Ground truth
fig, ax = plt.subplots()
ax.set_xlim(-0.2, 9.2)
ax.set_ylim(-20, 30)
x = np.linspace(*ga.x_bound, 500)
ax.plot(x, ga.f(x))

sca = ax.scatter([], [], marker="X", s=100, c='#00FF00', alpha=0.8)


def update(*args):
    g,individuals, fitness_values = next(gaiter)
    fx = individuals
    fv = fitness_values
    sca.set_offsets(np.column_stack((fx, fv)))
    ax.set_title("generation=" + str(g))
    # plt.savefig(f'best.png')


ani = matplotlib.animation.FuncAnimation(fig, update, interval=500, repeat=False)
writergif = matplotlib.animation.PillowWriter(fps=1)
ani.save('animation.gif', writer=writergif)











