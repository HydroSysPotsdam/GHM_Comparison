import matplotlib.pyplot as plt
import seaborn as sns
from helpers.easyit.easyit import load_data_all
from helpers import plotting_fcts
import os
import numpy as np
from scipy import stats
import pandas as pd
from pingouin import partial_corr

# This script plots different kinds of plots for different ISIMIP models.
# Most plots are scatter plots between two variables.

### load data ###
data_path = "model_outputs/2b/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
outputs = ["evap", "qr", "qs", "qsb", "qtot", "netrad", "potevap"]
forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "domains"] # domains contains pr, potevap, netrad

df = load_data_all(data_path, forcings, outputs, rmv_outliers=True)

df["netrad"] = df["netrad"] * 12.87 # transform radiation into mm/y
df["totrad"] = df["rlds"] + df["rsds"]
df["totrad"] = df["totrad"] / 2257 * 0.001 * (60 * 60 * 24 * 365) # transform radiation into mm/y

domains = ["wet warm", "dry warm", "wet cold", "dry cold"]

print("Finished loading data.")

### define functions ###

def scatterplots(df):

    # check if folder exists
    results_path = "results/lines/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # rename variables and models for paper
    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
    df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
    df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
    df = df.sort_values(by=["sort_helper", "ghm"])
    new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    df = df.replace(new_names, regex=True)

    # define variables
    x_name_list = ["Precipitation", "Net radiation", "totrad"]
    x_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    x_lim_list = [[-100, 3100], [-100, 2100], [2400, 10100]]
    y_name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    y_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    y_lim_list = [[-100, 2100], [-100, 2100], [-100, 2100]]

    # loop over variables
    for x_name, x_unit, x_lim in zip(x_name_list, x_unit_list, x_lim_list):

        for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

            sns.set_style("ticks",{'axes.grid' : True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in","ytick.direction": "in"})
            g = sns.FacetGrid(df, col="ghm", col_wrap=4, palette=palette)
            g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name,
                            domains="domain_days_below_1_0.08_aridity_netrad", alpha=0.5, s=1)
            d = "domain_days_below_1_0.08_aridity_netrad"
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm")
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm")
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold")
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold")

            g.set(xlim=x_lim, ylim=y_lim)
            if x_name == "Precipitation" or y_name == "Actual evapotranspiration":
                g.map(plotting_fcts.plot_origin_line, x_name, y_name)
            g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
            g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
            g.set_titles(col_template = '{col_name}')
            g.tight_layout()
            sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
            for axes in g.axes.ravel():
                axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)

            for ghm in ghms:
                print(x_name + " / " + y_name + " / " + ghm + " / total")
                print(stats.spearmanr(df.loc[(df["ghm"]==ghm), x_name], df.loc[(df["ghm"]==ghm), y_name], nan_policy='omit'))

            g.savefig(results_path + x_name + '_' + y_name + "_line_plot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished line plots.")


### run functions ###

scatterplots(df)
