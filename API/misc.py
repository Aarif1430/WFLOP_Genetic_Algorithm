import pandas as pd
import requests

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


def plot_power_curve():
    pass