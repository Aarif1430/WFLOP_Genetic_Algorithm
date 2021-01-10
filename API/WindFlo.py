from __future__ import print_function   # For Python 3 compatibility
import numpy as np
from API.misc import *
# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def simplified_gaussian_wake(frame_coords, N, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in doi:10.1088/1742-6596/1037/4/042012
    turb_coords = np.asarray(list(zip(list(frame_coords[0]), list(frame_coords[1]))))
    turb_coords = np.recarray((N,), coordinate)
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
    # Experiment related to wake_loss vs. Diameter
    diameters = np.arange(77, 121)
    wake_loss = []
    for dia in diameters:
        wake_loss.append(simplified_gaussian_wake(turb_coords,25,dia))
    wk = [np.mean(wake) for wake in wake_loss]
    generic_plot(diameters, wk)
    generic_plot(diameters, wk, 'Wake vs Diameter', 'Diameter', 'Wake_Loss', save=True)

    # Experiment related to wake_loss vs No. of Turbines
    no_of_turbines = np.arange(5, 26)
    wake_loss_ = []
    i = 5
    t_arr = np.copy(turb_coords)
    for turb in no_of_turbines:
        x_cor = t_arr[0][0:i]
        y_cor = t_arr[1][0:i]
        turb_coors = np.asarray([x_cor, y_cor])
        wake_loss_.append(simplified_gaussian_wake(turb_coors, turb, 77.0))
        t_arr = np.copy(turb_coords)
        i+=1
    wk = [np.mean(wake) for wake in wake_loss_]
    generic_plot(no_of_turbines, wk)
    generic_plot(no_of_turbines, wk, 'Wake vs No. of Turbines', 'No. of Turbines', 'Wake_Loss', save=True)

