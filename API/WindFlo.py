from __future__ import print_function   # For Python 3 compatibility
import numpy as np
import sys
from math import radians as DegToRad    # For converting degrees to radians

# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def GaussianWake(frame_coords, N, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in doi:10.1088/1742-6596/1037/4/042012
    turb_coords = np.asarray(list(zip(list(frame_coords[0]), list(frame_coords[1]))))
    turb_coords = np.recarray((25,), coordinate)
    turb_coords.x, turb_coords.y = frame_coords[0], frame_coords[1]
    num_turb = N

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)

    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = turb_coords.x[i] - turb_coords.x[j]   # Calculate the x-dist
            y = turb_coords.y[i] - turb_coords.y[j]   # And the y-offset
            if x > 0.:                   # If Primary is downwind of the Target
                sigma = k*x + turb_diam/np.sqrt(8.)  # Calculate the wake loss
                # Simplified Bastankhah Gaussian wake model
                exponent = -0.5 * (y/sigma)**2
                radical = 1. - CT/(8.*sigma**2 / turb_diam**2)
                loss_array[j] = (1.-np.sqrt(radical)) * np.exp(exponent)
            # Note that if the Target is upstream, loss is defaulted to zero
        # Total wake losses from all upstream turbs, using sqrt of sum of sqrs
        loss[i] = np.sqrt(np.sum(loss_array**2))

    return loss


if __name__ == "__main__":

    wind_dir = np.array([0, np.pi / 3.0, 2 * np.pi / 3.0, 3 * np.pi / 3.0, 4 * np.pi / 3.0, 5 * np.pi / 3.0],
                     dtype=np.float32)  # 0.2, 0,3 0.2  0. 1 0.1 0.1
    velocity = np.array([13.0], dtype=np.float32)  # 1
    f_theta_v = np.array([[0.2], [0.3], [0.2], [0.1], [0.1], [0.1]], dtype=np.float32)
    x_coords = [1270.5, 2194.5,  346.5, 1732.5, 2425.5,  346.5, 2194.5, 2425.5,
                                 115.5,  346.5,  808.5, 1039.5,  577.5, 1501.5, 1732.5, 1963.5,
                                1732.5, 2194.5, 2425.5,  346.5,  577.5, 1039.5, 1501.5, 2425.5,
                                1501.5]
    y_coords = [ 115.5,  115.5,  346.5,  577.5,  577.5,  808.5,  808.5,  808.5,
                                1039.5, 1039.5, 1039.5, 1039.5, 1270.5, 1270.5, 1270.5, 1270.5,
                                1732.5, 1732.5, 1732.5, 1963.5, 1963.5, 1963.5, 1963.5, 1963.5,
                                2194.5]

    turb_coords = np.asarray(list(zip(x_coords, y_coords)))
    turb_coords = np.recarray((25,), coordinate)
    turb_coords.x, turb_coords.y = x_coords, y_coords
    turb_coords = np.asarray([[1270.5, 2194.5, 346.5, 1732.5, 2425.5, 346.5, 2194.5, 2425.5,
                               115.5, 346.5, 808.5, 1039.5, 577.5, 1501.5, 1732.5, 1963.5,
                               1732.5, 2194.5, 2425.5, 346.5, 577.5, 1039.5, 1501.5, 2425.5,
                               1501.5], [115.5, 115.5, 346.5, 577.5, 577.5, 808.5, 808.5, 808.5,
                                         1039.5, 1039.5, 1039.5, 1039.5, 1270.5, 1270.5, 1270.5, 1270.5,
                                         1732.5, 1732.5, 1732.5, 1963.5, 1963.5, 1963.5, 1963.5, 1963.5,
                                         2194.5]])
    loss = GaussianWake(turb_coords,25,77.0)
    print(loss)
