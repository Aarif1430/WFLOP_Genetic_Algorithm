import time
from API.WindFlo import *
from datetime import datetime


class GAOptimizer(object):
    def __init__(self, rows=21, cols=21, no_of_turbines=0, pop_size=100, iteration=20, cell_width=0, elite_rate=0.2,
                 cross_rate=0.6, random_rate=0.5, mutate_rate=0.1):
        self.velocity = np.array([13.0], dtype=np.float32)  # 1
        # Wind direction form W --> E i.e 90 degrees.
        self.theta = np.array([3 * np.pi / 6.0], dtype=np.float32)
        self.turbine = Turbine()
        self.rows = rows
        self.cols = cols
        self.no_of_turbines = no_of_turbines
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
    def get_total_power(self):
        f_p = 0.0
        f_p = self.no_of_turbines * self.turbine.get_power_output(self.velocity[0])
        return f_p

    def layout_power(self, wind_speed, no_of_turbines):
        power = np.zeros(no_of_turbines, dtype=np.float32)
        for i in range(no_of_turbines):
            power[i] = self.turbine.get_power_output(wind_speed[i])
        return power

    def gen_init_pop(self):
        self.init_pop = self.gen_pop(self.rows, self.cols, self.pop_size, self.no_of_turbines)
        # init_pop_nonzero_indices: Tracks the cells in layout where turbine is placed
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.no_of_turbines), dtype=np.int32)
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
        return pop_indices

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
        return pop_indices

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

    def fitness(self, pop, rows, cols, pop_size, N):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):
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

            speed_deficiency = simplified_gaussian_wake(xy_position, N, self.turbine.rator_diameter)

            actual_velocity = (1 - speed_deficiency) * self.velocity[0]
            # total power of a specific layout specific wind speed specific theta
            lp_power = self.layout_power(actual_velocity,N)
            lp_power_accum += lp_power
            fitness_val[i] = np.sum(lp_power_accum)
        return fitness_val, xy_position, lp_power_accum

    def evolve(self):
        conversion_eff = []
        xy_positions = 0
        lp_power = 0
        total_rated_power = self.get_total_power()
        start_time = datetime.now()
        print("Genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        self.zeros = np.zeros((self.pop_size, self.no_of_turbines), dtype=np.int32)
        power_order = self.zeros
        pop = np.copy(self.init_pop)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        for gen in range(self.iteration):
            print("generation {}...".format(gen))
            fitness_value, xy_positions, lp_power = self.fitness(pop=pop, rows=self.rows, cols=self.cols,
                                                                 pop_size=self.pop_size, N=self.no_of_turbines)
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
            self.crossover(N=self.no_of_turbines, pop=pop, pop_indices=pop_indices,
                           pop_size=self.pop_size, n_parents=n_parents,
                           parent_layouts=parent_layouts, parent_pop_indices=parent_pop_indices)

            self.mutation(rows=self.rows, cols=self.cols, N=self.no_of_turbines, pop=pop, pop_indices=pop_indices,
                          pop_size=self.pop_size, mutation_rate=self.mutate_rate)

            print("Conversion efficiency is: %f" % (fitness_generations[gen] / total_rated_power))
            conversion_eff.append((fitness_generations[gen] / total_rated_power))
        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        return run_time, conversion_eff, xy_positions, lp_power


class Turbine(object):
    """
        'manufacturer': 'Enercon',
        'name': 'E-126/4200 EP4',
        'turbine_type': 'E-126/4200',
        'nominal_power': 4200,
        Usage:
            from misc import *
            data=fetch_turbine_data_from_oedb()
            print(dict(t.iloc[1]))
    """
    hub_height = 99.0  # unit (m)
    rator_diameter = 127.0  # unit m
    surface_roughness = 0.25 * 0.001  # unit mm surface roughness
    rator_radius = 0
    power_density = 3
    power_curve_wind_speeds = np.arange(1, 26)
    # These values are from API fetch_turbine_data_from_oedb() which is in misc.py
    power_curve_values = [0.0, 0.0, 58.0, 185.0, 400.0, 745.0, 1200.0, 1790.0, 2450.0, 3120.0, 3660.0, 4000.0, 4150.0,
                          4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0, 4200.0,
                          4200.0]

    def __init__(self):
        self.rator_radius = self.rator_diameter / 2

    def get_power_output(self, wind_speed):
        power_output = np.interp(
            wind_speed,
            self.power_curve_wind_speeds,
            self.power_curve_values,
            left=0,
            right=0,
        )
        return power_output
