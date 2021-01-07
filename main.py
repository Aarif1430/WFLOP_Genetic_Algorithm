import os
from API.WindFlo import *
from GA_Optimizer.WindForm_GA import *

elite_rate = 0.2
cross_rate = 0.6
random_rate = 0.5
mutate_rate = 0.1

number_of_turbines = 25

population_size = 50  # how many layouts in a population
iteration_times = 100  # how many iterations in a genetic algorithm run


# wind farm size, cells
cols_cells = 12  # number of cells each row
rows_cells = 12  # number of cells each column
cell_width = 77.0 * 3  # unit : m

# pop_size: how many individuals in the population
# iteration: iteration times of the genetic algorithm
optimizer = GAOptimizer(rows=rows_cells, cols=cols_cells, N=number_of_turbines,
                        pop_size=population_size, iteration=iteration_times, cell_width=cell_width,
                        elite_rate=elite_rate, cross_rate=cross_rate, random_rate=random_rate,
                        mutate_rate=mutate_rate)

optimizer.gen_init_pop()

run_time, conversion_eff, final_positions = optimizer.evolve()
print('')
