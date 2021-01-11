import pandas as pd
import requests
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib import gridspec
import matplotlib.ticker as ticker
import numpy as np
import os


plt.style.use('ggplot')


def fetch_turbine_data_from_oedb(schema="supply", table="wind_turbine_library"):
    r"""
    Fetches turbine library from the OpenEnergy database (oedb).
    Parameters
    ----------
    schema : str
        Database schema of the turbine library.
    table : str
        Table name of the turbine library.
    Returns
    -------
    :pandas:`pandas.DataFrame<frame>`
        Turbine data of different turbines such as 'manufacturer',
        'turbine_type', 'nominal_power'.
    """
    # url of OpenEnergy Platform that contains the oedb
    oep_url = "http://oep.iks.cs.ovgu.de/"
    url = oep_url + "/api/v0/schema/{}/tables/{}/rows/?".format(schema, table)

    # load data
    result = requests.get(url)
    if not result.status_code == 200:
        raise ConnectionError(
            "Database (oep) connection not successful. \nURL: {2}\n"
            "Response: [{0}] \n{1}".format(
                result.status_code, result.text, url
            )
        )
    return pd.DataFrame(result.json())


def convert_data_to_list(data):
    if data is not None:
        data = data.strip('[]').split(',')
        return [float(num) for num in data]


def save_figure(plot, title, in_parent_dir=True):
    plots_folder = "Plots"
    if in_parent_dir:
        parent_dir = os.path.dirname(os.getcwd())
        if not os.path.exists(f'{parent_dir}/{plots_folder}'):
            os.makedirs(f'{parent_dir}/{plots_folder}')
        plot.savefig(f'{parent_dir}/{plots_folder}/{title}.png')
    else:
        if not os.path.exists(f'{plots_folder}'):
            os.makedirs(f'{plots_folder}')
        plot.savefig(f'{plots_folder}/{title}.png')


def plot_power_curve():
    data = fetch_turbine_data_from_oedb()
    speeds = data.iloc[1].power_curve_wind_speeds
    values = data.iloc[1].power_curve_values
    xs = convert_data_to_list(speeds)
    ys = convert_data_to_list(values)

    plt.plot(xs, ys)

    plt.vlines(3, 0, 4300, linestyles='--', alpha=0.5, color='b')
    plt.text(3.5, 1800, 'Cut-in speed', rotation=90)

    plt.vlines(14, 0, 4300, linestyles='--', alpha=0.5, color='b')
    plt.text(14.5, 1800, 'Cut-out speed', rotation=90)

    plt.title(f"Turbine Power Curve")
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Power (kW)')
    plt.tight_layout()

    save_figure(plt, 'Turbine Power Curve')


def fmt(x_in, pos):
    return str('{0:.2f}'.format(x_in))


def plot_turbines(xc, yc, var, plotVariable='V', scale=1.0, title=''):
    plt.figure(figsize=(8, 5), edgecolor='gray', linewidth=2)
    ax = plt.subplot(1, 1, 1)
    ax.title.set_text('Optimized turbine positions')
    ax.set_xlabel('x [m]', fontsize=16, labelpad=5)
    ax.set_ylabel('y [m]', fontsize=16, labelpad=5)

    ax.tick_params(axis='x', which='major', labelsize=15, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=15, pad=0)

    jet_map = plt.get_cmap('jet')
    scatterPlot = ax.scatter(xc, yc, c=var, marker='^', s=100, cmap=jet_map, alpha=1)

    if (max(var) - min(var)) > 0:
        colorTicks = np.linspace(min(var), max(var), 7, endpoint=True)
        colorBar = plt.colorbar(scatterPlot, pad=0.06, shrink=0.8, format=ticker.FuncFormatter(fmt), ticks=colorTicks)

        colorBar.ax.tick_params(labelsize=16)
        colorBar.ax.set_title(title, fontsize=16, ha='left', pad=15)
        colorBar.update_ticks()

    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.tight_layout()
    return ax


# Generic Plot function
def generic_plot(x, y, title='Avg. Fitness Vs. Generations', xlabel='Generations', ylabel='Avg. Fitness', save=False):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.7, color='#ff0000', linestyle='-')
    if save:
        save_figure(plt, title)
    plt.show()


def generic_bar(x, y, title='Avg. Fitness Vs. Generations', xlabel='Generations', ylabel='Avg. Fitness', save=False,
                in_parent=True):
    print(f"Plotting:\n{x}\nAgainst:\n{y}")
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=-45)
    plt.ylim(min(y)-0.2, max(y)+0.05)
    plt.tight_layout()
    if save:
        save_figure(plt, title, in_parent)
    plt.close()


if __name__ == '__main__':
    plot_power_curve()

