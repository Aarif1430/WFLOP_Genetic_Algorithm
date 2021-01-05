import os
from API.WindFlo import *
from GA_Optimizer.WindForm_GA import *

elite_rate = 0.2
cross_rate = 0.6
random_rate = 0.5
mutate_rate = 0.1

wt_N = 25

population_size = 120  # how many layouts in a population
iteration_times = 3  # how many iterations in a genetic algorithm run

n_inits = 100  # number of initial populations n_inits >= run_times
run_times = 1  # number of different initial populations

# wind farm size, cells
cols_cells = 12  # number of cells each row
rows_cells = 12  # number of cells each column
cell_width = 77.0 * 3  # unit : m

# all data will be save in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Create a WindFarmGenetic object
# pop_size: how many individuals in the population
# iteration: iteration times of the genetic algorithm
wfg = WindFarmGenetic(rows=rows_cells, cols=cols_cells, N=wt_N, pop_size=population_size,
                      iteration=iteration_times, cell_width=cell_width, elite_rate=elite_rate,
                      cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)

wfg.gen_init_pop()

results_data_folder = "data/results"
if not os.path.exists(results_data_folder):
    os.makedirs(results_data_folder)

cg_result_folder = "{}/cg".format(results_data_folder)

result_arr = np.zeros((run_times, 2), dtype=np.float32)

# CGA: Conventional genetic algorithm
for i in range(0, run_times):  # run times
    print("run times {} ...".format(i))
    run_time, eta = wfg.genetic_alg(i, result_folder=cg_result_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
