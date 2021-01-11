import numpy as np

from API.misc import generic_plot, generic_bar, plot_turbines
from GA_Optimizer.WindForm_GA import GAOptimizer

# GA properties
elite_rate = 0.2
cross_rate = 0.6
random_rate = 0.5
mutate_rate = 0.1
population_size = 50  # how many layouts in a population
iteration_times = 50  # how many iterations in a genetic algorithm run

# Wind farm properties
cols_cells = 12  # number of cells each row
rows_cells = 12  # number of cells each column
cell_width = 77.0 * 3  # unit : m
number_of_turbines = 25


def run_optimizer():
    optimizer = GAOptimizer(rows=rows_cells, cols=cols_cells, no_of_turbines=number_of_turbines,
                            pop_size=population_size, iteration=iteration_times, cell_width=cell_width,
                            elite_rate=elite_rate, cross_rate=cross_rate, random_rate=random_rate,
                            mutate_rate=mutate_rate)

    optimizer.gen_init_pop()
    run_time, conversion_eff, final_positions, layout_power = optimizer.evolve()

    # Plot for single experiment
    generic_plot(range(len(conversion_eff)), conversion_eff)

    # Plot layout best positions
    plot_turbines(final_positions[0], final_positions[1], layout_power, scale=1.0e-3, title='P [kW]')


if __name__=='__main__':
    run_optimizer()
