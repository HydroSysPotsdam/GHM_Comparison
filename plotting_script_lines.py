import matplotlib.pyplot as plt
import seaborn as sns
from helpers.easyit.easyit import load_data_all
from helpers import plotting_fcts
import os
import numpy as np
from scipy import stats
import pandas as pd
from pingouin import partial_corr

# This script creates line plots for different ISIMIP models and datasets.

### load data ###
data_path = "model_outputs/2b/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
outputs = ["evap", "netrad", "potevap", "qr", "qtot"] # "qs", "qsb"]
forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "domains"] # domains contains pr, potevap, netrad

df = load_data_all(data_path, forcings, outputs, rmv_outliers=True)

df["netrad"] = df["netrad"] * 12.87 # transform radiation into mm/y
df["totrad"] = df["rlds"] + df["rsds"]
df["totrad"] = df["totrad"] / 2257 * 0.001 * (60 * 60 * 24 * 365) # transform radiation into mm/y

domains = ["wet warm", "dry warm", "wet cold", "dry cold"]

print("Finished loading data.")

### define functions ###

def plot_Budyko(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "Budyko_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls="dashed")

def plot_FLUXCOM(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "FLUXCOM_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='dotted')

def plot_FLUXNET(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#808080', "dry warm": '#808080',
                    "wet cold": '#808080', "dry cold": '#808080'}
    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "FLUXNET_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='solid')

def plot_Moeck(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "Moeck_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='solid')

def plot_MacDonald(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#808080', "dry warm": '#808080',
                    "wet cold": '#808080', "dry cold": '#808080'}
    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}

    df = pd.read_csv("data/" + "MacDonald_prepared.csv", sep=',')

    for i in [3]:
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='dashdot')

def plot_GSIM(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#826d8c', "dry warm": '#826d8c',
                    "wet cold": '#826d8c', "dry cold": '#826d8c'}
    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "GSIM_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='solid')

def plot_GRUN(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#808080', "dry warm": '#808080',
                    "wet cold": '#808080', "dry cold": '#808080'}
    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "GRUN_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='dotted')

def plot_CARAVAN(d, domain_list, x_name, y_name, axs, n):

    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    data_palette = {"wet warm": '#3B3B3B', "dry warm": '#3B3B3B',
                    "wet cold": '#3B3B3B', "dry cold": '#3B3B3B'}
    df = pd.read_csv("data/" + "CARAVAN_prepared.csv", sep=',')

    for i in range(0, 4):
        subgroup = df[d] == domain_list[i]
        x = df.loc[subgroup, x_name]
        y = df.loc[subgroup, y_name]
        plotting_fcts.plot_lines(x, y, axs[i], data_palette, domains, domain_list[i], n=n, ls='dashdot')


def multilineplots(df):

    # check if folder exists
    results_path = "results/lines/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # rename variables and models for paper
    df.rename(columns = {'pr':'$P$', 'netrad':'$N$',
                         'evap':'$E_a$', 'qr':'$R$', 'qtot':'$Q$'}, inplace = True)
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
    df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
    df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
    df = df.sort_values(by=["sort_helper", "ghm"])
    #new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    #df = df.replace(new_names, regex=True)

    # define variables
    x_name_list = ["$P$", "$N$"]
    x_unit_list = [" [mm/yr]", " [mm/yr]"]
    x_lim_list = [[[800, 3800], [100, 1500], [50, 800], [-100, 1600]],
                  [[700, 2500], [-200, 1500], [0, 1500], [500, 2700]]]
    #y_name_list = ["Actual evap.", "Gr. recharge", "Total runoff"]
    y_name_list = ["$E_a$", "$R$", "$Q$"]
    y_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    y_lim_list = [[[400, 1500], [-700/10, 700], [50, 550], [-1300/10, 1300]],
                  [[0, 1800], [-700/10, 700], [-120/10, 120], [-400/10, 400]],
                  [[200, 2500], [0, 1100], [-400/10, 400], [-700/10, 700]]]

    # loop over variables
    for x_name, x_unit, x_lim in zip(x_name_list, x_unit_list, x_lim_list):

        for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

            fig, axs = plt.subplots(1, 4, figsize=(8, 2)) #, facecolor='w', edgecolor='k'
            axs = axs.ravel()

            d = "domain_days_below_1_0.08_aridity_netrad"
            domain_list = ["wet warm", "wet cold", "dry cold", "dry warm"]
            #ghm_list = ["CLM4.5", "CWatM", "H08", "JULES-W1", "LPJmL", "MATSIRO", "PCR-GLOBWB", "WaterGAP2"]
            ghm_list = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]

            n = 11

            for i in range(0,4):

                for ghm in ghm_list:

                    subgroup = np.logical_and(df[d]==domain_list[i], df["ghm"]==ghm)
                    x = df.loc[subgroup, x_name]
                    y = df.loc[subgroup, y_name]
                    plotting_fcts.plot_lines(x, y, axs[i], palette, domains, domain_list[i], n=n)
                    axs[i].grid('major', alpha=0.5, linewidth=1)

                axs[i].set_xlabel(x_name + x_unit)
                if i == 0:
                    axs[i].set_ylabel(y_name + y_unit)
                axs[i].set_xlim(x_lim[i])
                axs[i].set_ylim(y_lim[i])
                #axs[i].set_title(domain_list[i])

                if x_name=="$P$" and y_name=="$E_a$":
                    plot_FLUXCOM(d, domain_list, "Precipitation GSWP3", "Actual evapotranspiration", axs, n)
                    #plot_FLUXNET(d, domain_list, "Precipitation", "Actual evapotranspiration", axs, n)
                    plot_Budyko(d, domain_list, "Precipitation", "Actual evapotranspiration", axs, n)
                    plotting_fcts.plot_origin_line_alt(axs[i])

                if x_name=="$P$" and y_name=="$R$":
                    plot_MacDonald(d, domain_list, "Precipitation", "Groundwater recharge", axs, 6)
                    plot_Moeck(d, domain_list, "Precipitation GSWP3", "Groundwater recharge", axs, 6)
                    plotting_fcts.plot_origin_line_alt(axs[i])

                if x_name=="$P$" and y_name=="$Q$":
                    plot_GSIM(d, domain_list, "Precipitation GSWP3", "Total runoff", axs, n)
                    plot_GRUN(d, domain_list, "Precipitation GSWP3", "Total runoff", axs, n)
                    #plot_CARAVAN(d, domain_list, "Precipitation", "Total runoff", axs, n)
                    plot_Budyko(d, domain_list, "Precipitation", "Total runoff", axs, n)
                    plotting_fcts.plot_origin_line_alt(axs[i])

                if x_name=="$N$" and y_name=="$E_a$":
                    plot_FLUXCOM(d, domain_list, "Net radiation", "Actual evapotranspiration", axs, n)
                    #plot_FLUXNET(d, domain_list, "Precipitation", "Actual evapotranspiration", axs, n)
                    plot_Budyko(d, domain_list, "Net radiation", "Actual evapotranspiration", axs, n)
                    plotting_fcts.plot_origin_line_alt(axs[i])

                if x_name=="$N$" and y_name=="$Q$":
                    plot_Budyko(d, domain_list, "Net radiation", "Total runoff", axs, n)
                    plotting_fcts.plot_origin_line_alt(axs[i])

            plt.tight_layout()
            fig.savefig(results_path + x_name + '_' + y_name + "_multiline_plot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished multi line plots.")


def multilineplots_alt(df):

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
    #new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    #df = df.replace(new_names, regex=True)

    # define variables
    x_name_list = ["Precipitation", "Net radiation"]
    x_unit_list = [" [mm/yr]", " [mm/yr]"]
    x_lim_list = [[-100, 3100], [-100, 2100]]
    y_name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    y_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    y_lim_list = [[-100, 2100], [-100, 2100], [-100, 2100]]

    # loop over variables
    for x_name, x_unit, x_lim in zip(x_name_list, x_unit_list, x_lim_list):

        for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

            fig, axs = plt.subplots(2, 2, figsize=(7, 7)) #, facecolor='w', edgecolor='k'
            axs = axs.ravel()

            d = "domain_days_below_1_0.08_aridity_netrad"
            domain_list = ["wet warm", "wet cold", "dry cold", "dry warm"]
            #ghm_list = ["CLM4.5", "CWatM", "H08", "JULES-W1", "LPJmL", "MATSIRO", "PCR-GLOBWB", "WaterGAP2"]
            ghm_list = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]

            n = 11

            for i in range(0,4):

                for ghm in ghm_list:

                    subgroup = np.logical_and(df[d]==domain_list[i], df["ghm"]==ghm)
                    x = df.loc[subgroup, x_name]
                    y = df.loc[subgroup, y_name]
                    plotting_fcts.plot_lines(x, y, axs[i], palette, domains, domain_list[i], n=n)
                    axs[i].grid('major')

                if i > 1:
                    axs[i].set_xlabel(x_name + x_unit)
                if i == 0 or i == 2:
                    axs[i].set_ylabel(y_name + y_unit)
                #axs[i].set_xlim(x_lim)
                #axs[i].set_ylim(y_lim)
                #axs[i].set_title(domain_list[i])
                plt.grid('major')

                if x_name=="Precipitation" and y_name=="Actual evapotranspiration":
                    plot_FLUXCOM(d, domain_list, "Precipitation GSWP3", y_name, axs, n)
                    #plot_FLUXNET(d, domain_list, "Precipitation", y_name, axs, n)
                    plot_Budyko(d, domain_list, "Precipitation", y_name, axs, n)

                if x_name=="Precipitation" and y_name=="Groundwater recharge":
                    plot_MacDonald(d, domain_list, "Precipitation", y_name, axs, n)
                    plot_Moeck(d, domain_list, "Precipitation GSWP3", y_name, axs, n)

                if x_name=="Precipitation" and y_name=="Total runoff":
                    #plot_GSIM(d, domain_list, "Precipitation", y_name, axs, n)
                    plot_GRUN(d, domain_list, "Precipitation GSWP3", y_name, axs, n)
                    plot_CARAVAN(d, domain_list, "Precipitation", y_name, axs, n)
                    plot_Budyko(d, domain_list, "Precipitation", y_name, axs, n)

                if x_name=="Net radiation" and y_name=="Actual evapotranspiration":
                    plot_FLUXCOM(d, domain_list, "Net radiation", y_name, axs, n)
                    #plot_FLUXNET(d, domain_list, "Precipitation", y_name, axs, n)
                    plot_Budyko(d, domain_list, "Net radiation", y_name, axs, n)

            fig.savefig(results_path + x_name + '_' + y_name + "_multiline_plot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished multi line plots.")


### run functions ###

multilineplots(df)
#multilineplots_alt(df) # alternative multiline plots
