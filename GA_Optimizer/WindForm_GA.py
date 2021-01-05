import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from API.WindFlo import *
from datetime import datetime


class WindFarmGenetic(object):
    elite_rate = 0.2    # elite rate: parameter for genetic algorithm
    cross_rate = 0.6    # crossover rate: parameter for genetic algorithm
    random_rate = 0.5   # random rate: parameter for genetic algorithm
    mutate_rate = 0.1   # mutation rate: parameter for genetic algorithm
    turbine = None
    pop_size = 0    # population size : how many individuals in a population
    N = 0           # number of wind turbines
    rows = 0        # how many cell rows the wind farm are divided into
    cols = 0        # how many colus the wind farm land are divided into
    iteration = 0   # how many iterations the genetic algorithm run
    cell_width = 0  # cell width
    cell_width_half = 0  # half cell width
    turb_ci = 4.0
    turb_co = 25.0
    rated_ws = 9.8
    rated_pwr = 3350000

    def __init__(self, rows=21, cols=21, N=0, pop_size=100, iteration=20, cell_width=0, elite_rate=0.2,
                 cross_rate=0.6, random_rate=0.5, mutate_rate=0.1):
        self.f_theta_v = np.array([[0.2], [0.3], [0.2], [0.1], [0.1], [0.1]], dtype=np.float32)
        self.velocity = np.array([13.0], dtype=np.float32)  # 1
        self.theta = np.array([0, np.pi / 3.0, 2 * np.pi / 3.0, 3 * np.pi / 3.0, 4 * np.pi / 3.0, 5 * np.pi / 3.0],
                              dtype=np.float32)  # 0.2, 0,3 0.2  0. 1 0.1 0.1
        self.turbine = GE_1_5_sleTurbine()
        self.rows = rows
        self.cols = cols
        self.N = N
        self.pop_size = pop_size
        self.iteration = iteration

        self.cell_width = cell_width
        self.cell_width_half = cell_width * 0.5

        self.elite_rate = elite_rate
        self.cross_rate = cross_rate
        self.random_rate = random_rate
        self.mutate_rate = mutate_rate

        self.init_pop = None
        self.init_pop_NA = None
        self.init_pop_nonezero_indices = None

    # calculate total rate power
    def cal_P_rate_total(self):
        f_p = 0.0
        for ind_t in range(len(self.theta)):
            for ind_v in range(len(self.velocity)):
                f_p += self.f_theta_v[ind_t, ind_v] * self.turbine.P_i_X(self.velocity[ind_v])
        return self.N * f_p

    def layout_power(self, velocity, N):
        power = np.zeros(N, dtype=np.float32)
        for i in range(N):
            power[i] = self.turbine.P_i_X(velocity[i])
        return power

    def gen_init_pop(self):
        self.init_pop = self.gen_pop(self.rows, self.cols, self.pop_size, self.N)
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1

    def gen_pop(self, rows, cols, n, N):
        np.random.seed(seed=int(time.time()))
        layouts = np.zeros((n, rows * cols), dtype=np.int32)
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0
        ind_pos = 0

        while ind_rows < n:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
            if np.sum(layouts[ind_rows, :]) == N:
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        return layouts

    def mutation(self, rows, cols, N, pop, pop_indices, pop_size, mutation_rate):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1
            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])

    def crossover(self, N, pop, pop_indices, pop_size, n_parents,
                               parent_layouts, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < pop_size:
            male = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            if male != female:
                cross_point = np.random.randint(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]

                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1
        return

    def select(self, pop, pop_indices, pop_size, elite_rate, random_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_pop_indices

    def fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = GaussianWake(trans_xy_position, N, self.turbine.rator_diameter)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]
            fitness_val[i] = np.sum(lp_power_accum)
        return fitness_val

    def genetic_alg(self, ind_time=0,result_folder=None):
        P_rate_total = self.cal_P_rate_total()
        start_time = datetime.now()
        print("Genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols), dtype=np.int32)  # best layout in each generation
        self.zeros = np.zeros((self.pop_size, self.N), dtype=np.int32)
        power_order = self.zeros  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(self.pop_size * self.elite_rate))  # elite number
        rN = int(int(np.floor(self.pop_size * self.mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = self.pop_size - eN - mN  # crossover number

        for gen in range(self.iteration):
            print("generation {}...".format(gen))
            fitness_value = self.fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                      N=self.N,
                                                      po=power_order)
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]

            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]

            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]

            n_parents, parent_layouts, parent_pop_indices = self.select(pop=pop, pop_indices=pop_indices,
                                                                        pop_size=self.pop_size,
                                                                        elite_rate=self.elite_rate,
                                                                        random_rate=self.random_rate)
            self.crossover(N=self.N, pop=pop, pop_indices=pop_indices, pop_size=self.pop_size,
                           n_parents=n_parents, parent_layouts=parent_layouts,
                           parent_pop_indices=parent_pop_indices)

            self.mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop,pop_indices=pop_indices,
                          pop_size=self.pop_size, mutation_rate=self.mutate_rate)

        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        eta_generations = np.copy(fitness_generations)
        eta_generations = eta_generations * (1.0 / P_rate_total)
        return run_time, eta_generations[self.iteration - 1]


class GE_1_5_sleTurbine:
    hub_height = 80.0  # unit (m)
    rator_diameter = 77.0  # unit m
    surface_roughness = 0.25 * 0.001  # unit mm surface roughness
    # surface_roughness = 0.25  # unit mm surface roughness
    rator_radius = 0

    entrainment_const = 0

    def __init__(self):
        self.rator_radius = self.rator_diameter / 2
        self.entrainment_const = 0.5 / np.log(self.hub_height / self.surface_roughness)
        return

    # power curve
    def P_i_X(self, v):
        if v < 2.0:
            return 0
        elif v < 12.8:
            return 0.3 * v ** 3
        elif v < 18:
            return 629.1
        else:
            return 0

