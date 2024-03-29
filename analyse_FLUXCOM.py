import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xarray as xr
from helpers import plotting_fcts
from helpers.weighted_mean import weighted_temporal_mean
from datetime import datetime as dt

# This script loads and analyses FLUXCOM data.

# prepare data
#data_path = "D:/Data/FLUXCOM/RS_METEO/ensemble/ALL/monthly/"
data_path = "D:/Data/FLUXCOM/RS/ensemble/720_360/monthly/"

# check if folder exists
results_path = "results/fluxcom/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
name_list = ["H", "LE", "Rn"]
#var_list = ["H.RS_METEO.EBC-ALL.MLM-ALL.METEO-ALL.720_360.monthly.",
#            "LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-ALL.720_360.monthly.",
#            "Rn.RS_METEO.EBC-NONE.MLM-ALL.METEO-ALL.720_360.monthly."]
var_list = ["H.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.",
            "LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.",
            "Rn.RS.EBC-NONE.MLM-ALL.METEO-NONE.720_360.monthly."]
years = ["2001", "2002", "2003", "2004", "2005",
         "2006", "2007", "2008", "2009", "2010",
         "2011", "2012", "2013", "2014", "2015"]

# get multi annual averages
def re(path,name):
    data = xr.open_dataset(path)
    d = weighted_temporal_mean(data,name)
    d.name = name
    return d

df_tot = pd.DataFrame(columns = ["lat", "lon"])
for name, var in zip(name_list, var_list):

    # get annual averages
    data = []
    for y in years:
        path = data_path + var + y + ".nc"
        data.append(re(path,name))

    # get average of all years
    data_all_years = xr.concat(data,"time")
    data_avg = data_all_years.mean("time")

    # transform into dataframe
    df = data_avg.to_dataframe().reset_index()
    df[name] = df[name] * (10**6/86400)*12.87 # MJ m^-2 d^-1 into W m^-2 into mm/y
    df_tot = pd.merge(df_tot, df, on=['lat', 'lon'], how='outer')

"""
# add land area
d = xr.open_dataset("D:/Data/FLUXCOM/ancillary/landfraction.720_360.nc")
df_land = d.to_dataframe().reset_index().dropna()
df_tot = pd.merge(df_tot, df_land, on=['lat', 'lon'], how='outer')
"""
"""
# add vegetated area
d = xr.open_dataset("D:/Data/FLUXCOM/ancillary/vegfraction.720_360.nc")
df_veg = d.to_dataframe().reset_index()#.dropna()
df_tot = pd.merge(df_tot, df_veg, on=['lat', 'lon'], how='inner')
df_tot[(df_tot["vegfraction"]<0.95)] = np.nan

# plot map
var = "vegfraction"
plotting_fcts.plot_map(df_tot["lon"], df_tot["lat"], df_tot[var], " [mm/y]", var, np.linspace(0, 1, 11))
#plotting_fcts.mask_greenland("2b/aggregated/")
ax = plt.gca()
ax.coastlines(linewidth=0.5)
plt.savefig(results_path + var + "_map.png", dpi=600, bbox_inches='tight')
plt.close()
"""

var = "Rn"
plotting_fcts.plot_map(df_tot["lon"], df_tot["lat"], df_tot[var], " [mm/y]", var, np.linspace(0, 2000, 11))
#plotting_fcts.mask_greenland("2b/aggregated/")
ax = plt.gca()
ax.coastlines(linewidth=0.5)
plt.savefig(results_path + var + "_map.png", dpi=600, bbox_inches='tight')
plt.close()

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()
df_tot = pd.merge(df_tot, df_area, on=['lat', 'lon'], how='outer')

# scatter plot
df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
df = pd.merge(df_tot, df_domains, on=['lat', 'lon'], how='outer')
df = df.dropna()
df.rename(columns={'pr_median': 'Precipitation HadGEM2-ES', 'pr_gswp3': 'Precipitation GSWP3', 'netrad_median': 'Net radiation ISIMIP',
                   'evap': 'Actual ET', 'qr': 'Groundwater recharge', 'qtot': 'Total runoff',
                   'LE': 'Actual evapotranspiration', 'H': 'Sensible heat', 'Rn': 'Net radiation'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

# global averages need to be weighted due to different grid cell areas
df_weighted = df.copy().dropna()
print(np.round((df_weighted["Sensible heat"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(),2))
print(np.round(df["Sensible heat"].mean(),2))
print(np.round((df_weighted["Actual evapotranspiration"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(),2))
print(np.round(df["Actual evapotranspiration"].mean(),2))
print(np.round((df_weighted["Net radiation"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(),2))
print(np.round(df["Net radiation"].mean(),2))

domains = ["wet warm", "dry warm", "wet cold", "dry cold"]
for d in domains:
    df_tmp = df.loc[(df["domain_days_below_1_0.08_aridity_netrad"] == d)]
    print(d)
    print(np.round((df_tmp["Sensible heat"] * df_tmp["continentalarea"]).sum() / df_tmp["continentalarea"].sum(), 2))
    print(np.round((df_tmp["Actual evapotranspiration"] * df_tmp["continentalarea"]).sum() / df_tmp["continentalarea"].sum(), 2))
    print(np.round((df_tmp["Net radiation"] * df_tmp["continentalarea"]).sum() / df_tmp["continentalarea"].sum(), 2))

# from FLUXCOM paper
# vegetated = 0.765, cold deserts = 0.108, hot deserts = 0.1265
# global mean net radiation as 75.49 Wm-2, sensible heat as 32.39 Wm−2, latent heat  as 39.14 Wm−2
# hot deserts: 5.9356 MJ m−2 day−1 for Rn and 5.8264 MJ m−2 day−1 for H
# cold deserts: Rn as −0.1826 MJ m−2 day−1 and H was der of −33.2 Wm−2 (−2.8685 MJ m−2 day−1)
weights = np.array([0.765, 0.108, 0.1265])
mean_Rn = np.array([0, -0.1826*(10**6/86500)*12.87, 5.9356*(10**6/86500)*12.87])
mean_H = np.array([0, -2.8685*(10**6/86500)*12.87, 5.8264*(10**6/86500)*12.87])
mean_LE = np.array([0, 0.001, 0.001]) # assuming desert LE is minimal
global_Rn = 75.49*12.87
global_H = 32.39*12.87
global_LE = 39.14*12.87
mean_LE[0] = (global_LE - mean_LE[1]*weights[1] - mean_LE[2] * weights[2])/weights[0]
mean_Rn[0] = (global_Rn - mean_Rn[1]*weights[1] - mean_Rn[2] * weights[2])/weights[0]
mean_H[0] = (global_H - mean_H[1]*weights[1] - mean_H[2] * weights[2])/weights[0]
print('Recalculated averages from FLUXCOM paper:')
print(np.round(mean_LE[0],2))
print(np.round(mean_Rn[0],2))
print(np.round(mean_H[0],2))

# scatter plots
x_name = "Precipitation GSWP3"
y_name = "Actual evapotranspiration"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
d = "domain_days_below_1_0.08_aridity_netrad"
n = 11
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold", n=n)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_FLUXCOM.png", dpi=600, bbox_inches='tight')
plt.close()

sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_FLUXCOM.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Actual evapotranspiration"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
d = "domain_days_below_1_0.08_aridity_netrad"
n = 11
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold", n=n)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_FLUXCOM.png", dpi=600, bbox_inches='tight')
plt.close()

sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_FLUXCOM.png", dpi=600, bbox_inches='tight')
plt.close()

# count grid cells per climate region (dropna because nan values were not part of the scatter plot)
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts())
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts()/df.dropna()["domain_days_below_1_0.08_aridity_netrad"].count())

# save data to df
df.to_csv("data/" + "FLUXCOM_prepared.csv", index=False)
