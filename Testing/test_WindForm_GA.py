from GA_Optimizer.WindForm_GA import Turbine
from API.misc import *
import numpy as np


def _test_power_curve():
    turb = Turbine()
    output_powr = []
    test_speeds = np.arange(1, 30)
    for w_s in test_speeds:
        output_powr.append(turb.get_power_output(w_s))
    generic_plot(test_speeds, output_powr, 'Wind speed vs Power', 'Wind Speed', 'Power', save=True)


if __name__=='__main__':
    _test_power_curve()