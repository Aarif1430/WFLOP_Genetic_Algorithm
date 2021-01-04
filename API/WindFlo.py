import numpy as np
from math import radians as DegToRad    # For converting degrees to radians

# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


# choose wind distribution : 6 direction, 1 speed
def init_6_direction_1_speed_13():
    theta = np.array([0, np.pi / 3.0, 2 * np.pi / 3.0, 3 * np.pi / 3.0, 4 * np.pi / 3.0, 5 * np.pi / 3.0],
                          dtype=np.float32)  # 0.2, 0,3 0.2  0. 1 0.1 0.1
    velocity = np.array([13.0], dtype=np.float32)  # 1
    f_theta_v = np.array([[0.2], [0.3], [0.2], [0.1], [0.1], [0.1]], dtype=np.float32)
    return theta, velocity, f_theta_v


def WindFrame(turb_coords, wind_dir_deg):
    """Convert map coordinates to downwind/crosswind coordinates."""

    # Convert from meteorological polar system (CW, 0 deg.=N)
    # to standard polar system (CCW, 0 deg.=W)
    # Shift so North comes "along" x-axis, from left to right.
    wind_dir_deg = 270. - wind_dir_deg
    # Convert inflow wind direction from degrees to radians
    wind_dir_rad = DegToRad(wind_dir_deg)

    # Constants to use below
    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)
    # Convert to downwind(x) & crosswind(y) coordinates
    frame_coords = np.recarray(turb_coords.shape, coordinate)
    frame_coords.x = (turb_coords[0] * cos_dir) - (turb_coords[1] * sin_dir)
    frame_coords.y = (turb_coords[0] * sin_dir) + (turb_coords[1] * cos_dir)

    return frame_coords

def GaussianWake(frame_coords, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in <iea37-wakemodel.pdf>
    num_turb = len(frame_coords[0])

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)
    sorted_index = np.argsort(-frame_coords[1, :])
    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = np.absolute(frame_coords[0, sorted_index[i]] - frame_coords[0, sorted_index[j]])
            y = np.absolute(frame_coords[1, sorted_index[i]] - frame_coords[1, sorted_index[j]])
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


if __name__=='__main__':
    xy_positons = np.asarray([[346.5, 1039.5, 115.5, 2194.5, 1501.5, 1732.5, 115.5, 808.5,
      1501.5, 1963.5, 115.5, 1039.5, 1270.5, 1501.5, 2656.5, 346.5,
      1039.5, 115.5, 577.5, 2194.5, 2656.5, 346.5, 1039.5, 1501.5,
      1963.5],
     [115.5, 115.5, 346.5, 346.5, 577.5, 577.5, 808.5, 808.5,
      808.5, 808.5, 1039.5, 1039.5, 1039.5, 1039.5, 1039.5, 1270.5,
      1270.5, 1501.5, 1501.5, 1732.5, 1732.5, 2194.5, 2194.5, 2194.5,
      2194.5]])

    d = 77.0
    wind_dir_deg, _, _ = init_6_direction_1_speed_13()
    # frame_coords = WindFrame(xy_positons, wind_dir_deg[0])
    wake = GaussianWake(xy_positons, d)
    print('')
