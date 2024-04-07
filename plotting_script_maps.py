import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers.easyit.easyit import load_data_all
from helpers import plotting_fcts
import os
import xarray as xr

# This script plots different kinds of maps for different ISIMIP models.

### load data ###
data_path = "model_outputs/2b/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
outputs = ["evap", "netrad", "potevap", "qr", "qtot"] # "qs", "qsb"]
forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "domains"] # domains contains pr, potevap, netrad

df = load_data_all(data_path, forcings, outputs)
df["netrad"] = df["netrad"] / 2257 * 0.001 * (60 * 60 * 24 * 365) # transform radiation into mm/y

domains = ["wet warm", "dry warm", "wet cold", "dry cold"]

print("Finished loading data.")

### define functions ###

def globalmeans(df, ghms, domains):

    # check if folder exists
    results_path = "results/maps/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # grid cell areas
    area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
    df_area = area.to_dataframe().reset_index().dropna()
    plotting_fcts.plot_map(df_area["lon"], df_area["lat"], df_area["continentalarea"], " km^2", "Continental area", np.linspace(0, 4000, 11))
    ax = plt.gca()
    ax.coastlines(linewidth=0.5)
    plt.savefig(results_path + "continentalarea_map.png", dpi=600, bbox_inches='tight')
    plt.close()

    df = pd.merge(df, df_area, on=['lat', 'lon'], how='outer')#.dropna()

    # calculate means, std, and medians
    df_weighted = df.copy()
    df_weighted["pr"] = df_weighted["pr"] * df_weighted["continentalarea"]
    df_weighted["netrad"] = df_weighted["netrad"]* df_weighted["continentalarea"]
    df_weighted["evap"] = df_weighted["evap"]* df_weighted["continentalarea"]
    df_weighted["qr"] = df_weighted["qr"]* df_weighted["continentalarea"]
    df_weighted["qtot"] = df_weighted["qtot"]* df_weighted["continentalarea"]

    df_means = df_weighted.groupby(["ghm"]).sum()
    df_means = df_means.iloc[:,:-1].div(df_means["continentalarea"], axis=0)
    df_means["evap_pr"] = df_means["evap"]/df_means["pr"]
    df_means["qr_pr"] = df_means["qr"]/df_means["pr"]
    df_means["qtot_pr"] = df_means["qtot"]/df_means["pr"]
    df_means.loc['mean'] = df_means[0:7].mean()
    df_means.loc['std'] = df_means[0:7].std()
    df_means.loc['min'] = df_means[0:7].min()
    df_means.loc['max'] = df_means[0:7].max()
    df_means[["pr", "netrad", "evap", "qr", "qtot", "evap_pr", "qr_pr", "qtot_pr"]].transpose().to_csv(results_path + "means.csv", float_format='%.2f')

    for d in domains:
        df_tmp = df.loc[(df["domain_days_below_1_0.08_aridity_netrad"]==d)]
        print(d)
        print(df_tmp["continentalarea"].sum()/df["continentalarea"].sum())
        print(df_tmp["continentalarea"].count()/df["continentalarea"].count())

        df_tmp["pr"] = df_tmp["pr"] * df_tmp["continentalarea"]
        df_tmp["netrad"] = df_tmp["netrad"]* df_tmp["continentalarea"]
        df_tmp["evap"] = df_tmp["evap"]* df_tmp["continentalarea"]
        df_tmp["qr"] = df_tmp["qr"]* df_tmp["continentalarea"]
        df_tmp["qtot"] = df_tmp["qtot"]* df_tmp["continentalarea"]
        # calculate means, std, and medians
        df_means = df_tmp.groupby(["ghm"]).mean()
        df_means = df_means.iloc[:,:-1].div(df_means["continentalarea"], axis=0)
        df_means["evap_pr"] = df_means["evap"]/df_means["pr"]
        df_means["qr_pr"] = df_means["qr"]/df_means["pr"]
        df_means["qtot_pr"] = df_means["qtot"]/df_means["pr"]
        df_means.loc['mean'] = df_means[0:7].mean()
        df_means.loc['std'] = df_means[0:7].std()
        #df_means.loc['cov'] = df_means.loc['std']/df_means.loc['mean']
        df_means.loc['min'] = df_means[0:7].min()
        df_means.loc['max'] = df_means[0:7].max()
        df_means[["pr", "netrad", "evap", "qr", "qtot", "evap_pr", "qr_pr", "qtot_pr"]].transpose().to_csv(results_path + "means_" + d + ".csv", float_format='%.2f')

    print("Finished global means.")


def map_plots(df, ghms):

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    df["Evapotranspiration ratio"] = df["Actual evapotranspiration"]/df["Precipitation"]
    df["Recharge ratio"] = df["Groundwater recharge"]/df["Precipitation"]
    df["Runoff ratio"] = df["Total runoff"] / df["Precipitation"]

    new_names = {'clm45': 'CLM4.5', 'jules-w1': 'JULES-W1', 'lpjml': 'LPJmL', 'matsiro': 'MATSIRO',
                 'pcr-globwb': 'PCR-GLOBWB', 'watergap2': 'WaterGAP2', 'h08': 'H08', 'cwatm': 'CWatM'}
    df = df.replace(new_names, regex=True)
    ghms = ["CLM4.5", "CWatM", "H08", "JULES-W1", "LPJmL", "MATSIRO", "PCR-GLOBWB", "WaterGAP2"]

    # define variables
    name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    lim_list = [[0, 2000], [0, 500], [0, 2000]]

    name_list = ["Evapotranspiration ratio", "Recharge ratio", "Runoff ratio"]
    unit_list = [" [-]", " [-]", " [-]"]
    lim_list = [[0, 1.25], [0, 0.25], [0, 1.25]]

    # loop over variables
    for name, unit, lim in zip(name_list, unit_list, lim_list):

        # check if folder exists
        results_path = "results/maps/"
        if not os.path.isdir(results_path + name):
            os.makedirs(results_path + name)

        # loop over models
        for g in ghms:
            plotting_fcts.plot_map(df.loc[df["ghm"] == g, "lon"], df.loc[df["ghm"] == g, "lat"], df.loc[df["ghm"] == g, name],
                                   unit, name, np.linspace(lim[0], lim[1], 11))
            plotting_fcts.mask_greenland(data_path)
            ax = plt.gca()
            ax.set_title(g)
            plt.savefig(results_path + name + "/" + g + "_" + name + "_map.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished map plots.")


def aridity_map_plots(df, ghms):

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    # check if folder exists
    results_path = "results/maps/"
    if not os.path.isdir(results_path + "aridity"):
        os.makedirs(results_path + "aridity")

    # define variables
    df["Aridity"] = df["Net radiation"]/df["Precipitation"]
    name = "Aridity"
    unit = " [-]"
    lim = [0, 2]

    # loop over models
    for g in ghms:
        plotting_fcts.plot_map(df.loc[df["ghm"] == g, "lon"], df.loc[df["ghm"] == g, "lat"], df.loc[df["ghm"] == g, name],
                               var_unit=unit, var_name=name, bounds=np.linspace(lim[0], lim[1], 11))
        plotting_fcts.mask_greenland(data_path)
        ax = plt.gca()
        plt.savefig(results_path + name + "/" + g + "_" + name + "_map.png", dpi=600, bbox_inches='tight')
        plt.close()

    print("Finished aridity map plots.")


def outlier_plots(df, ghms):

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    # define variables
    name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    unit_list = [" [-]", " [-]", " [-]"]
    lim_list = [[0, 1], [0, 1], [0, 1]]

    for name, unit, lim in zip(name_list, unit_list, lim_list):

        # check if folder exists
        results_path = "results/maps/"
        if not os.path.isdir(results_path + name):
            os.makedirs(results_path + name)

        # loop over models
        for g in ghms:
            plotting_fcts.plot_outliers(df, g, name, unit)
            plotting_fcts.mask_greenland(data_path)
            ax = plt.gca()
            plt.savefig(results_path + name + "/" + g + "_" + name + "_outliers.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished outlier plots.")


def white_spaces_plots(df, ghms):

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    # define variables
    name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    lim_list = [[0, 1], [0, 1], [0, 1]] #
    #lim_list = [[0, .2], [0, .2], [0, .2]] #lim_list = [[0, 2], [0, 2], [0, 2]]
    #lim_list = [[0, 200], [0, 200], [0, 200]]
    #lim_list = [[0, 1000], [0, 1000], [0, 1000]]
    color_scale_list = ['Greens', 'Purples', 'Blues']

    # define plot type (coefficient of variation or standard deviation of ratio)
    var_type = 'CoV'#'mean'#'CoV' #'std' #'ratio'

    # check if folder exists
    results_path = "results/white_spaces/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    for name, lim, color_scale in zip(name_list, lim_list, color_scale_list):

        df_short = df[["lon", "lat", "Precipitation", name, "ghm"]]
        df_tmp = df_short.pivot_table(index=["lon", "lat", "Precipitation"], columns='ghm', values=name).reset_index()  # Reshape from long to wide
        columns_titles = [ "lon", "lat", "Precipitation", "domain_days_below_1_0.08_aridity_netrad"] + ghms
        df_tmp = df_tmp.reindex(columns=columns_titles)

        df_tmp['mean'] = df_tmp.iloc[:, 3:3+len(ghms)].mean(axis=1)
        df_tmp['std'] = df_tmp.iloc[:, 3:3+len(ghms)].std(axis=1)
        df_tmp['min'] = df_tmp.iloc[:, 3:3+len(ghms)].min(axis=1)
        df_tmp['max'] = df_tmp.iloc[:, 3:3+len(ghms)].max(axis=1)

        if var_type == 'CoV':
            df_tmp['var'] = df_tmp['std'] / df_tmp['mean']
            unit = " [-]"
        elif var_type == 'ratio':
            df_tmp['var'] = df_tmp['std'] / df_tmp['Precipitation']
            unit = " [-]"
        elif var_type == 'range':
            df_tmp['var'] = (df_tmp['max'] - df_tmp['min']) / df_tmp['mean']
            unit = " [-]"
        elif var_type == 'std':
            df_tmp['var'] = df_tmp['std']
            unit = " [mm/y]"
        elif var_type == 'mean':
            df_tmp['var'] = df_tmp['mean']
            unit = " [mm/y]"

        plotting_fcts.plot_map(df_tmp["lon"], df_tmp["lat"], df_tmp["var"],
                               unit, var_type + " (" + name + ")", bounds=np.linspace(lim[0], lim[1], 11),
                               colormap=color_scale, colormap_reverse=True)
        plotting_fcts.mask_greenland(data_path)
        ax = plt.gca()
        ax.coastlines(linewidth=0.25)
        plt.savefig(results_path + name + "_" + var_type + "_white_spaces.png", dpi=600, bbox_inches='tight')
        plt.close()

        print(name)
        print(var_type)
        print(str(np.round(df_tmp["var"].mean(),2)))
        print(str(np.round(df_tmp["var"].median(),2)))

    print("Finished white spaces plots.")

def most_deviating_model_plot(df, ghms):

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)
    ghms = ['CLM4.5', 'CWatM', 'H08', 'JULES-W1', 'LPJmL', 'MATSIRO', 'PCR-GLOBWB', 'WaterGAP2']

    # define variables
    name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]

    # check if folder exists
    results_path = "results/white_spaces/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    for name in name_list:

        df_short = df[["lon", "lat", "Precipitation", "domain_days_below_1_0.08_aridity_netrad", name, "ghm"]]
        df_tmp = df_short.pivot_table(index=["lon", "lat", "Precipitation", "domain_days_below_1_0.08_aridity_netrad"], columns='ghm', values=name).reset_index()  # Reshape from long to wide
        columns_titles = [ "lon", "lat", "Precipitation", "domain_days_below_1_0.08_aridity_netrad"] + ghms
        df_tmp.columns = columns_titles

        df_tmp['mean'] = df_tmp.iloc[:, 4:4+len(ghms)].mean(axis=1)
        df_tmp['std'] = df_tmp.iloc[:, 4:4+len(ghms)].std(axis=1)
        df_tmp['CoV'] = df_tmp["std"]/df_tmp["mean"]

        # get highest value
        diff = (df_tmp.iloc[:, 4:4+len(ghms)].values.T - df_tmp['mean'].values).T
        max_ind = np.argmax(abs(diff), axis=1)
        max_diff = np.max(abs(diff), axis=1)

        # get 2nd highest value
        diff_2 = (df_tmp.iloc[:, 4:4+len(ghms)].values.T - df_tmp['mean'].values).T
        for k in range(len(max_ind)):
            diff_2[k, max_ind[k]] = np.mean(diff_2[k, :])
        max_ind_2 = np.argmax(abs(diff_2), axis=1)
        max_diff_2 = np.max(abs(diff_2), axis=1)

        # check how different the first and second highest values are
        max_ind[(abs(max_diff - max_diff_2) / max_diff) < 0.2] = -1

        # if difference to mean is too small, value is not considered an outlier
        max_ind[(max_diff / df_tmp['mean'].values) < 0.2] = -2

        df_tmp["max_ind"] = max_ind
        df_tmp["max_diff"] = max_diff
        df_tmp["max_diff_2"] = max_diff_2
        df_tmp2 = df_tmp.loc[np.logical_and(df_tmp["CoV"] > 0.9, df_tmp["max_ind"] == -2)]

        plotting_fcts.plot_most_deviating_model(df_tmp, max_ind, ghms, name)
        plotting_fcts.mask_greenland(data_path)
        ax = plt.gca()

        # save stats about "outlier" models
        hist, bins = np.histogram(max_ind, bins=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        df_fractions = pd.DataFrame(hist)#.transpose()
        df_fractions.columns = ["count"]
        df_fractions["fraction"] = df_fractions["count"]/len(max_ind)
        df_fractions = df_fractions.transpose()
        df_fractions.columns = ["none"] + ["multiple"] + ghms
        df_fractions.to_csv(results_path + name + "_fractions.csv", float_format='%.2f')

        # save stats about "outlier" models
        for d in ["wet warm", "dry warm", "wet cold", "dry cold"]:
            hist, bins = np.histogram(max_ind[(df_tmp["domain_days_below_1_0.08_aridity_netrad"] == d).values], bins=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            df_fractions = pd.DataFrame(hist)#.transpose()
            df_fractions.columns = ["count"]
            df_fractions["fraction"] = df_fractions["count"]/len(max_ind[(df_tmp["domain_days_below_1_0.08_aridity_netrad"] == d).values])
            df_fractions = df_fractions.transpose()
            df_fractions.columns = ["none"] + ["multiple"] + ghms
            df_fractions.to_csv(results_path + name + "_fractions" + d + ".csv", float_format='%.2f')

        plt.savefig(results_path + name + "_most_deviating_model.png", dpi=600,  bbox_inches='tight')
        plt.close()

    print("Finished most deviating model plots.")


### run functions ###

# globalmeans(df, ghms, domains)
# map_plots(df, ghms)
# aridity_map_plots(df, ghms)
# outlier_plots(df, ghms)
white_spaces_plots(df, ghms)
most_deviating_model_plot(df, ghms)
