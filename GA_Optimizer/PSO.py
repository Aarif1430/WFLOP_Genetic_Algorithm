import random

import numpy as np

from API.WindFlo import simplified_gaussian_wake
from GA_Optimizer.WindForm_GA import Turbine

'''
WIP Module

Code for some PSO aspects were used from:
https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
'''


class Particle:  # a particle represents a turbine layout
    def __init__(self):
        self.position = []
        self.error = -1
        self.velocity = [random.uniform(-1, 1) for _ in range(num_dimensions)]  # start with random velocity
        # personal best position & error:
        self.best_position = []
        self.best_error = -1

    def evaluate(self):
        # TODO: get layout from particle's current position
        self.error = self.cost_function(np.zeros((1, self.turbine_count), dtype=np.int32))
        # check to see if the current position is an individual best
        if self.error < self.best_error or self.best_error == -1:
            self.best_position = self.position
            self.best_error = self.error

    def update_velocity(self, pos_best_g):
        w = 0.5  # inertia weighting (how much to weigh the previous velocity)
        c1 = 1  # cognitive weighting (how much the personal best affects velocity)
        c2 = 2  # social weighting (how much the global best affects velocity)

        for i in range(0, num_dimensions):
            vel_cognitive = c1 * random.random() * (self.best_position[i] - self.position[i])
            vel_social = c2 * random.random() * (pos_best_g[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_cognitive + vel_social

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]


class Swarm:  # population of particles (turbine layouts)
    def __init__(self, init, size, iters, bounds):
        best_global_error = -1
        best_global_position = []

        swarm = [Particle() for _ in range(size)]
        for _ in range(iters):
            for i in range(size):
                swarm[i].evaluate()

                # determine if current particle is the global best
                if swarm[i].error < best_global_error or best_global_error == -1:
                    best_global_position = list(swarm[i].position)
                    best_global_error = float(swarm[i].error)

                # iterate through swarm and update velocities and positions
                for j in range(0, size):
                    swarm[j].update_velocity(best_global_position)
                    swarm[j].update_position(bounds)


class Layout:
    def __init__(self, rows=21, cols=21, turbine_count=25, cell_width=231.):
        self.rows = rows
        self.cols = cols
        self.turbine_count = turbine_count
        self.cell_width = cell_width
        self.cell_width_half = cell_width * 0.5
        self.velocity = np.array([13.0], dtype=np.float32)
        self.theta = np.array([0, np.pi / 3.0, 2 * np.pi / 3.0, 3 * np.pi / 3.0, 4 * np.pi / 3.0, 5 * np.pi / 3.0],
                              dtype=np.float32)
        self.turbine = Turbine()
        self.f_theta_v = np.array([[0.2], [0.3], [0.2], [0.1], [0.1], [0.1]], dtype=np.float32)
        self.turbine_layout = self.gen_layout()

    def gen_layout(self):
        layout = np.zeros((1, self.rows * self.cols), dtype=np.int8)
        positionX = np.random.randint(0, self.cols, size=(self.turbine_count * 2))
        positionY = np.random.randint(0, self.rows, size=(self.turbine_count * 2))
        ind_rows = 0
        ind_pos = 0

        while ind_rows < 1:
            layout[ind_rows, positionX[ind_pos] + positionY[ind_pos] * self.cols] = 1
            if np.sum(layout[ind_rows, :]) == self.turbine_count:
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= self.turbine_count * 2:
                print("Not enough positions")
                break

        return layout

    def cost_function(self, power_order):
        xy_position = np.zeros((2, self.turbine_count), dtype=np.float32)
        cr_position = np.zeros((2, self.turbine_count), dtype=np.int32)
        ind_position = np.zeros(self.turbine_count, dtype=np.int32)
        ind_pos = 0
        for ind in range(self.rows * self.cols):
            if self.turbine_layout[0, ind] == 1:
                r_i = np.floor(ind / self.cols)
                c_i = np.floor(ind - r_i * self.cols)
                cr_position[0, ind_pos] = c_i
                cr_position[1, ind_pos] = r_i
                xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                ind_position[ind_pos] = ind
                ind_pos += 1
        lp_power_accum = np.zeros(self.turbine_count, dtype=np.float32)
        for ind_t in range(len(self.theta)):
            for ind_v in range(len(self.velocity)):
                trans_matrix = np.array(
                    [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                     [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                    np.float32)

                trans_xy_position = np.matmul(trans_matrix, xy_position)
                speed_deficiency = simplified_gaussian_wake(trans_xy_position, self.turbine_count,
                                                            self.turbine.rator_diameter)

                actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                lp_power = self.layout_power(actual_velocity)
                lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                lp_power_accum += lp_power

        sorted_index = np.argsort(lp_power_accum)
        power_order[0, :] = ind_position[sorted_index]
        fitness_val = np.sum(lp_power_accum)
        return fitness_val

    def layout_power(self, velocity):
        power = np.zeros(self.turbine_count, dtype=np.float32)
        for i in range(self.turbine_count):
            power[i] = self.turbine.get_power_output(velocity[i])
        return power


if __name__ == "__main__":
    row, col = 12, 12
    turbines = 25
    width = 77.0 * 3

    swarm_size = 100  # number of particles in swarm
    iterations = 5

    # TODO How to present wind turbine layouts to PSO?
    # With 144 potential turbine positions and 25 turbines to place there are 144C25 potential layouts

    initial = [int(row / 2), int(col / 2)]  # initial starting location
    boundaries = [(0, row), (0, col)]  # input bounds
    num_dimensions = len(initial)

    Swarm(initial, swarm_size, iterations, boundaries)
