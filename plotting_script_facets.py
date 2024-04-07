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
outputs = ["evap", "netrad", "potevap", "qr", "qtot"] # "qs", "qsb"]
forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "domains"] # domains contains pr, potevap, netrad

df = load_data_all(data_path, forcings, outputs, rmv_outliers=True)

df["netrad"] = df["netrad"] * 12.87 # transform radiation into mm/y
df["totrad"] = df["rlds"] + df["rsds"]
df["totrad"] = df["totrad"] / 2257 * 0.001 * (60 * 60 * 24 * 365) # transform radiation into mm/y

domains = ["wet warm", "wet cold", "dry cold", "dry warm"]

print("Finished loading data.")

### define functions ###

def corrstats(df, ghms, domains):

    # check if folder exists
    results_path = "results/scatterplots/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # loop over all variable pairs
    for x in ["pr", "netrad"]:
        for y in ["evap", "qr", "qtot"]:

            stats_mat = np.zeros([len(ghms)+1,len(domains)+1])
            stats_mat_partial = np.zeros([len(ghms)+1,len(domains)+1])

            # loop over all models and domains
            for i in range(0, len(ghms)):
                for j in range(0, len(domains)):

                    # calculate rank correlations
                    input = df.loc[np.logical_and(df["ghm"]==ghms[i], df["domain_days_below_1_0.08_aridity_netrad"]==domains[j]), x]
                    output = df.loc[np.logical_and(df["ghm"]==ghms[i], df["domain_days_below_1_0.08_aridity_netrad"]==domains[j]), y]
                    r_sp, _ = stats.spearmanr(input, output, nan_policy='omit')

                    df_tmp = df.loc[np.logical_and(df["ghm"]==ghms[i], df["domain_days_below_1_0.08_aridity_netrad"]==domains[j])]

                    if x=='pr':
                        z = 'netrad'
                    elif x=='netrad':
                        z = 'pr'
                    r_partial_mat = partial_corr(data=df_tmp, x=x, y=y, covar=[z], method='spearman')
                    r_partial = r_partial_mat["r"].values[0]

                    stats_mat[i,j] = r_sp
                    stats_mat_partial[i,j] = r_partial

                stats_mat[i,len(domains)] = np.max(stats_mat[i,:-1]) - np.min(stats_mat[i,:-1])
                stats_mat_partial[i,len(domains)] = np.max(stats_mat_partial[i,:-1]) - np.min(stats_mat_partial[i,:-1])

            for j in range(0, len(domains)):
                stats_mat[len(ghms),j] = np.max(stats_mat[:-1,j]) - np.min(stats_mat[:-1,j])
                stats_mat_partial[len(ghms),j] = np.max(stats_mat_partial[:-1,j]) - np.min(stats_mat_partial[:-1,j])

            stats_mat[len(ghms),len(domains)] = np.max(stats_mat[:-1,:-1]) - np.min(stats_mat[:-1,:-1])
            stats_mat_partial[len(ghms),len(domains)] = np.max(stats_mat_partial[:-1,:-1]) - np.min(stats_mat_partial[:-1,:-1])

            df_stats = pd.DataFrame(stats_mat)
            df_stats.columns = domains + ["range"]
            df_stats = df_stats.transpose()
            df_stats.columns = ghms + ["range"]
            df_stats.to_csv(results_path + "corrstats_" + x + "_" + y + ".csv", float_format='%.2f')

            df_stats_partial = pd.DataFrame(stats_mat_partial)
            df_stats_partial.columns = domains + ["range"]
            df_stats_partial = df_stats_partial.transpose()
            df_stats_partial.columns = ghms + ["range"]
            df_stats_partial.to_csv(results_path + "corrstats_partial_" + x + "_" + y + ".csv", float_format='%.2f')

    print("Finished correlation statistics.")


def regressionstats(df, ghms, domains):

    from sklearn.linear_model import LinearRegression

    # check if folder exists
    results_path = "results/regression/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # loop over all variable pairs
    for x in ["pr", "netrad"]:
        for y in ["evap", "qr", "qtot"]:

            stats_mat = np.zeros([len(ghms)+1,len(domains)+1])
            stats_mat_partial = np.zeros([len(ghms)+1,len(domains)+1])

            # loop over all models and domains
            for i in range(0, len(ghms)):
                for j in range(0, len(domains)):

                    # calculate rank correlations
                    input = df.loc[np.logical_and(df["ghm"]==ghms[i], df["domain_days_below_1_0.08_aridity_netrad"]==domains[j]), x]
                    output = df.loc[np.logical_and(df["ghm"]==ghms[i], df["domain_days_below_1_0.08_aridity_netrad"]==domains[j]), y]
                    reg = LinearRegression().fit(input.array.reshape(-1, 1), output.array)

                    b0 = reg.intercept_
                    b1 = reg.coef_

                    stats_mat[i,j] = b1

                stats_mat[i,len(domains)] = np.max(stats_mat[i,:-1]) - np.min(stats_mat[i,:-1])

            for j in range(0, len(domains)):
                stats_mat[len(ghms),j] = np.max(stats_mat[:-1,j]) - np.min(stats_mat[:-1,j])

            stats_mat[len(ghms),len(domains)] = np.max(stats_mat[:-1,:-1]) - np.min(stats_mat[:-1,:-1])

            df_stats = pd.DataFrame(stats_mat)
            df_stats.columns = domains + ["range"]
            df_stats = df_stats.transpose()
            df_stats.columns = ghms + ["range"]
            df_stats.to_csv(results_path + "regressionstats_" + x + "_" + y + ".csv", float_format='%.2f')

    print("Finished regression statistics.")


def scatterplots(df):

    # check if folder exists
    results_path = "results/scatterplots/" #subsample/
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # rename variables and models for paper
    #df = df.sample(8*10000)
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
                            domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
            d = "domain_days_below_1_0.08_aridity_netrad"
            n = 11
            #g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm", n=n)
            #g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm", n=n)
            #g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold", n=n)
            #g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold", n=n)
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

            g.savefig(results_path + x_name + '_' + y_name + "_scatterplot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished scatter plots.")


def scatterplots_per_domain(df):

    # check if folder exists
    results_path = "results/scatterplots_per_domain/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # rename variables and models for paper
    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)
    new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    df = df.replace(new_names, regex=True)
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}

    # define variables
    x_name_list = ["Precipitation", "Net radiation", "totrad"]
    x_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    x_lim_list = [[-100, 3100], [-100, 2100], [2400, 10100]]
    y_name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff"]
    y_unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]"]
    y_lim_list = [[-100, 2100], [-100, 2100], [-100, 2100]]

    # loop over domains
    for d in ["wet warm", "dry warm", "wet cold", "dry cold"]:

        df_tmp = df.loc[df["domain_days_below_1_0.08_aridity_netrad"] == d]

        # loop over variables
        for x_name, x_unit, x_lim in zip(x_name_list, x_unit_list, x_lim_list):

            for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

                sns.set_style("ticks",{'axes.grid' : True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in","ytick.direction": "in"})
                g = sns.FacetGrid(df_tmp, col="ghm", col_wrap=4,
                                  palette=palette)
                g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name,
                                domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
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
                    print(stats.spearmanr(df_tmp.loc[(df_tmp["ghm"]==ghm), x_name], df_tmp.loc[(df_tmp["ghm"]==ghm), y_name], nan_policy='omit'))

                g.savefig(results_path + x_name + '_' + y_name + "_" + d + "_scatterplot.png", dpi=600, bbox_inches='tight')
                plt.close()

    print("Finished scatter plots per domain.")


def scatterplots_ensemble(df):

    # check if folder exists
    results_path = "results/scatterplots_ensemble/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # rename variables and models for paper
    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)
    new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    df = df.replace(new_names, regex=True)
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}

    # calculate mean
    df_means = df.groupby(["lat", "lon"]).mean().reset_index()
    df_dummy = df.loc[df["ghm"] == "PCR-GLOBWB", ["lat", "lon", "domain_days_below_1_0.08_aridity_netrad"]] # contains almost all cells
    df_means = pd.merge(df_means, df_dummy, on=['lat', 'lon'], how='outer').dropna()
    df_means["dummy"] = "Ensemble"

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

            sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
            g = sns.FacetGrid(df_means, col="dummy", col_wrap=4, palette=palette)
            g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name,
                            domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
            d = "domain_days_below_1_0.08_aridity_netrad"
            n = 11
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm", n=n)
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm", n=n)
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold", n=n)
            g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold", n=n)
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
                print(stats.spearmanr(df_means[x_name], df_means[y_name], nan_policy='omit'))

            g.savefig(results_path + x_name + '_' + y_name + "_ensemble_scatterplot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished ensemble scatter plots.")


def budyko_plots(df):

    # check if folder exists
    results_path = "results/budyko/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
    df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
    df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
    df = df.sort_values(by=["sort_helper", "ghm"])
    new_names = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WaterGAP2', 'h08':'H08', 'cwatm':'CWatM'}
    df = df.replace(new_names, regex=True)

    # define variables
    df["Aridity"] = df["Net radiation"]/df["Precipitation"]
    df["E v P"] = df["Actual evapotranspiration"]/df["Precipitation"]
    df["R v P"] = df["Groundwater recharge"]/df["Precipitation"]
    df["Q v P"] = df["Total runoff"]/df["Precipitation"]
    x_name = "Aridity"
    x_unit = " [-]"
    y_name_list = ["E v P", "R v P", "Q v P"]
    y_unit_list = [" [-]", " [-]", " [-]"]
    y_lim_list = [[-0.2, 1.8], [-0.2, 1.8], [-0.2, 1.8]]

    # loop over variables
    for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

        sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
        g = sns.FacetGrid(df, col="ghm", hue="domain_days_below_1_0.08_aridity_netrad", col_wrap=4, palette=palette)
        g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name,
                        domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
        g.set(xlim=(-1,7), ylim=y_lim)
        if y_name == "E v P":
            g.map(plotting_fcts.plot_Budyko_limits, x_name, y_name)
        g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
        g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
        g.set_titles(col_template = '{col_name}')
        g.tight_layout()
        sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
        for axes in g.axes.ravel():
            axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
        g.savefig(results_path + 'aridity' + '_' + y_name + "_budyko.png", dpi=600, bbox_inches='tight')
        plt.close()

    print("Finished Budyko plots.")


def histogram_plots(df):

    # check if folder exists
    results_path = "results/histograms/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    # define variables
    name_list = ["Actual evapotranspiration", "Groundwater recharge", "Total runoff", "Precipitation", "Net radiation"]
    unit_list = [" [mm/yr]", " [mm/yr]", " [mm/yr]", " [mm/yr]", " [mm/yr]"]
    lim_list = [[-100, 2100], [-100, 1100], [-100, 2100], [-100, 3100], [-100, 2100]]

    # loop over variables

    for name, unit, lim in zip(name_list, unit_list, lim_list):

        sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
        g = sns.FacetGrid(df, col="ghm", col_wrap=4)
        g.map(sns.histplot, name, bins=np.linspace(lim[0], lim[1], 50))
        g.set(xlim=lim)
        g.set(xlabel=name+unit)
        g.set_titles(col_template = '{col_name}')
        g.tight_layout()
        sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
        g.savefig(results_path + name + "_histogram.png", dpi=600, bbox_inches='tight')
        plt.close()

    print("Finished histograms.")


def coloured_scatterplots(df):

    # check if folder exists
    results_path = "results/scatterplots/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    df.rename(columns = {'pr':'Precipitation', 'netrad':'Net radiation',
                         'evap':'Actual evapotranspiration', 'qr':'Groundwater recharge', 'qtot':'Total runoff'}, inplace = True)

    # define variables
    y_name_list = ["Actual evapotranspiration"]
    y_unit_list = [" [mm/yr]"]
    y_lim_list = [[-100, 2100]]

    # loop over variables
    for y_name, y_unit, y_lim in zip(y_name_list, y_unit_list, y_lim_list):

        x_name = "Precipitation"
        x_unit = " [mm/y]"
        z_name = "Net radiation"
        z_unit = " [mm/y]"
        df["hue"] = np.round(df[z_name]/100)*100 # to have fewer unique values
        sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
        g = sns.FacetGrid(df, col="ghm", hue="hue", palette='viridis', col_wrap=4)
        g.map(sns.scatterplot, x_name, y_name, alpha=1, s=1)
        g.set(xlim=(-100, 3100), ylim=(-100, 2100))
        g.map(plotting_fcts.plot_origin_line, x_name, y_name)
        g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
        g.set_titles(col_template = '{col_name}')
        g.tight_layout()
        sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
        norm = plt.Normalize(df["hue"].min(), df["hue"].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        ax = plt.gca()
        ax.figure.colorbar(sm)
        g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_colour.png", dpi=600, bbox_inches='tight')
        plt.close()

    print("Finished coloured scatter plots.")


def regressionplots(df):

    # check if folder exists
    results_path = "results/regression/"
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
                            domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
            g.set(xlim=x_lim, ylim=y_lim)
            if x_name == "Precipitation" or y_name == "Actual evapotranspiration":
                g.map(plotting_fcts.plot_origin_line, x_name, y_name)
            #g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
            g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
            g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
            g.set_titles(col_template = '{col_name}')
            g.tight_layout()
            sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
            for axes in g.axes.ravel():
                axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)

            #for ghm in ghms:
            #    print(x_name + " / " + y_name + " / " + ghm + " / total")
            #    print(stats.spearmanr(df.loc[(df["ghm"]==ghm), x_name], df.loc[(df["ghm"]==ghm), y_name], nan_policy='omit'))

            g.savefig(results_path + x_name + '_' + y_name + "_scatterplot.png", dpi=600, bbox_inches='tight')
            plt.close()

    print("Finished regression plots.")


def latitude_plots(df):

    # note: changing var names in other plot affects df, so run this before the other plots

    # check if folder exists
    results_path = "results/latitude/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # make plot for pr and netrad
    sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
    g = sns.FacetGrid(df, col="ghm", col_wrap=4)
    g.map_dataframe(plotting_fcts.plot_latitudinal_averages_forcing)
    g.add_legend()
    g.tight_layout()
    x_name = 'netrad_pr'
    g.tight_layout()
    sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
    g.savefig(results_path + x_name + "_latitude.png", dpi=600, bbox_inches='tight')
    plt.close()

    # make plot for evap, qr, and qtot
    sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
    g = sns.FacetGrid(df, col="ghm", col_wrap=4)
    g.map_dataframe(plotting_fcts.plot_latitudinal_averages_fluxes)
    g.add_legend()
    g.tight_layout()
    x_name = 'evap_qr_qtot'
    g.tight_layout()
    sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
    g.savefig(results_path + x_name + "_latitude.png", dpi=600, bbox_inches='tight')
    plt.close()

    print("Finished latitude plots.")


### run functions ###

#latitude_plots(df)
#corrstats(df, ghms, domains)
#regressionstats(df, ghms, domains)
scatterplots(df)
#scatterplots_per_domain(df)
#scatterplots_ensemble(df)
#budyko_plots(df)
#histogram_plots(df)
#coloured_scatterplots(df)
#regressionplots(df)
