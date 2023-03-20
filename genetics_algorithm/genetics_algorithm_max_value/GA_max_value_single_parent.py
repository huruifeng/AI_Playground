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
    def __init__(self,pop_size=10,
                 generations=30,
                 gene_n = 32,
                 mutate_rate=0.2,
                 x_bound=[0,9],
                 ):
        self.pop_size = pop_size
        self.generations = generations
        self.gene_n = gene_n
        self.mutate_rate = mutate_rate
        self.x_bound = x_bound

        self.population = []

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

    def mutation(self,parent):
        parent = np.tile(parent,(self.pop_size,1))
        mut = np.random.choice(np.array([0, 1]), (self.pop_size,self.gene_n), p=[1 - self.mutate_rate, self.mutate_rate])
        children = np.where(mut == 1, 1 - parent, parent)
        return children

    def evolve(self):
        parent_indiv = np.random.randint(2, size=(1,self.gene_n))
        self.population = parent_indiv
        parent_indiv_dec,parent_fitness = self.cal_fitness()
        parent_indiv_dec = parent_indiv_dec[0]
        parent_fitness = parent_fitness[0]
        parent_indiv_x = self.x_bound[0] + (self.x_bound[1] - self.x_bound[0]) * (parent_indiv_dec / (2 ** self.gene_n))

        for g in range(self.generations):
            self.population = self.mutation(parent_indiv)
            individuals_dec,fitness_values = self.cal_fitness()
            individuals_x = self.x_bound[0] + (self.x_bound[1] - self.x_bound[0]) * (individuals_dec / (2 ** self.gene_n))
            yield g, np.append(individuals_x,parent_indiv_x), np.append(fitness_values,parent_fitness)

            best_fitness = np.max(fitness_values)
            best_indiv = self.population[np.argmax(fitness_values)]
            best_dec = individuals_dec[np.argmax(fitness_values)]
            if best_fitness > parent_fitness:
                parent_fitness = best_fitness
                parent_indiv = best_indiv
                parent_indiv_dec = best_dec
            parent_indiv_x = self.x_bound[0] + (self.x_bound[1] - self.x_bound[0]) * (parent_indiv_dec / (2 ** self.gene_n))
            yield g, parent_indiv_x, parent_fitness

            print(f'Generation:{g:3d} - '
                  f'Top individual: {parent_indiv_x:.6f}, '
                  f'Best fitness: {parent_fitness}')




ga = GA()
gaiter = ga.evolve()

## Ground truth
fig, ax = plt.subplots()
ax.set_xlim(-0.2, 9.2)
ax.set_ylim(-20, 30)
x = np.linspace(*ga.x_bound, 500)
ax.plot(x, ga.f(x))

sca = ax.scatter([], [], marker="X", s=60, c='#00FF00', alpha=0.8)


def update(*args):
    g, individuals, fitness_values = next(gaiter)
    fx = individuals
    fv = fitness_values
    sca.set_offsets(np.column_stack((fx, fv)))
    ax.set_title("generation="+str(g))
    # plt.savefig(f'best.png')

ani = matplotlib.animation.FuncAnimation(fig, update, interval=500, repeat=False)
writergif = matplotlib.animation.PillowWriter(fps=1)
ani.save('animation_single.gif', writer=writergif)











