from API.misc import *
from API.WindFlo import simplified_gaussian_wake


turb_coords = np.asarray([[1270.5, 2194.5, 346.5, 1732.5, 2425.5, 346.5, 2194.5, 2425.5,
                               115.5, 346.5, 808.5, 1039.5, 577.5, 1501.5, 1732.5, 1963.5,
                               1732.5, 2194.5, 2425.5, 346.5, 577.5, 1039.5, 1501.5, 2425.5,
                               1501.5], [115.5, 115.5, 346.5, 577.5, 577.5, 808.5, 808.5, 808.5,
                                         1039.5, 1039.5, 1039.5, 1039.5, 1270.5, 1270.5, 1270.5, 1270.5,
                                         1732.5, 1732.5, 1732.5, 1963.5, 1963.5, 1963.5, 1963.5, 1963.5,
                                         2194.5]])


def test_wake_loss_for_diameters():
    # Experiment related to wake_loss vs. Diameter
    diameters = np.arange(77, 121)
    wake_loss = []
    for dia in diameters:
        wake_loss.append(simplified_gaussian_wake(turb_coords,25,dia))
    wk = [np.mean(wake) for wake in wake_loss]
    generic_plot(diameters, wk, 'Wake vs Diameter', 'Diameter', 'Wake_Loss', save=True)


def test_wake_loss_for_no_of_turbines():
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
    generic_plot(no_of_turbines, wk, 'Wake vs No. of Turbines', 'No. of Turbines', 'Wake_Loss', save=True)


if __name__=='__main__':
    test_wake_loss_for_diameters()
    test_wake_loss_for_no_of_turbines()