import pandas as pd
import requests
import matplotlib.pyplot as plt
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

    parent_dir = os.path.dirname(os.getcwd())
    plots_folder = "Plots"
    if not os.path.exists(f'{parent_dir}/{plots_folder}'):
        os.makedirs(f'{parent_dir}/{plots_folder}')
    plt.savefig(f'{parent_dir}/{plots_folder}/Turbine Power Curve.png')


plot_power_curve()
