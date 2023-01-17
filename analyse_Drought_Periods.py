import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xarray as xr
from helpers import plotting_fcts
from datetime import datetime as dt
import cartopy as cart
from helpers.weighted_mean import weighted_temporal_mean

# This script loads and analyses GRUN data.

# prepare data
data_path = "D:/Data/GRUN/"

# check if folder exists
results_path = "results/drought_periods/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
file_path = "GRUN_v1_GSWP3_WGS84_05_1902_2014.nc"

# T data
tas = xr.open_dataset(r'./data/tas_gswp3-ewembi_1971_1980.nc4')
#tas = weighted_temporal_mean(tas, "tas")
#tas.name = "tas"
tas = tas.resample(time="1Y").mean()
for decade in ['1981_1990', '1991_2000', '2001_2010']:
    tmp = xr.open_dataset(r'./data/tas_gswp3-ewembi_' + decade + '.nc4')
    #tmp = weighted_temporal_mean(tmp, "tas")
    #tmp.name = "tas"
    tmp = tmp.resample(time="1Y").mean()
    tas = xr.merge([tas, tmp])


# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()

tas_mean = tas.mean("time")
df_tas_mean = tas_mean.to_dataframe().reset_index().dropna()

# plot maps
years = np.linspace(1971,2010,40)
for year in years:
    tas_tmp = tas.sel(time=slice(dt(int(year), 1, 1), dt(int(year), 12, 31)))
    # transform into dataframe
    df_tas = tas_tmp.to_dataframe().reset_index().dropna()
    df_tas['tas_anomaly'] = (df_tas['tas'] - df_tas_mean['tas'])#
    # df_tas.columns = ['lat', 'lon', 'tas_gswp3']
    df_tas = pd.merge(df_tas, df_area, on=['lat', 'lon'], how='outer')
    var = "tas_anomaly"
    plotting_fcts.plot_map(df_tas["lon"], df_tas["lat"], df_tas[var], " [-]", var, np.linspace(-3, 3, 11), colormap='RdBu', colortype='Diverging', colormap_reverse=True)
    ax = plt.gca()
    ax.add_feature(cart.feature.OCEAN, zorder=100)
    ax.coastlines(linewidth=0.5)
    df_weighted = df_tas.copy().dropna()
    tas_mean_tmp = (df_weighted["tas"] * df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum()
    plt.title(str(np.round(tas_mean_tmp)))
    plt.savefig(results_path + var + "_" + str(int(year)) + "_map.png", dpi=600, bbox_inches='tight')
    plt.close()

# P data
pr = xr.open_dataset(r'./data/pr_gswp3-ewembi_1971_1980.nc4')
#pr = weighted_temporal_mean(pr, "pr")
#pr.name = "pr"
pr = pr.resample(time="1Y").sum()
for decade in ['1981_1990', '1991_2000', '2001_2010']:
    tmp = xr.open_dataset(r'./data/pr_gswp3-ewembi_' + decade + '.nc4')
    #tmp = weighted_temporal_mean(tmp, "pr")
    #tmp.name = "pr"
    tmp = tmp.resample(time="1Y").sum()
    pr = xr.merge([pr, tmp])

# get multi annual averages
#var = "Runoff"
#data = xr.open_dataset(data_path + file_path)
#d = weighted_temporal_mean(data, "Runoff")
#d.name = "Runoff"
#d = d.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
#d = d.mean("time")
#print(d.mean().values)

# transform into dataframe
#df = d.to_dataframe().reset_index().dropna()
#df[var] = df[var] * 365 # mm/d to mm/y
#df.columns = ['lat', 'lon', 'qtot']

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()

pr_mean = pr.mean("time")
df_pr_mean = pr_mean.to_dataframe().reset_index().dropna()

# plot maps
years = np.linspace(1971,2010,40)
for year in years:
    pr_tmp = pr.sel(time=slice(dt(int(year), 1, 1), dt(int(year), 12, 31)))
    # transform into dataframe
    df_pr = pr_tmp.to_dataframe().reset_index().dropna()
    df_pr['pr_anomaly'] = (df_pr['pr'] - df_pr_mean['pr']) * 86400 * 0.001 * 1000  # to mm/y /df_pr_mean['pr'] #
    # df_pr.columns = ['lat', 'lon', 'pr_gswp3']
    df_pr = pd.merge(df_pr, df_area, on=['lat', 'lon'], how='outer')
    var = "pr_anomaly"
    plotting_fcts.plot_map(df_pr["lon"], df_pr["lat"], df_pr[var], " [-]", var, np.linspace(-500, 500, 11), colormap='RdBu', colortype='Diverging')
    ax = plt.gca()
    ax.add_feature(cart.feature.OCEAN, zorder=100)
    ax.coastlines(linewidth=0.5)
    df_weighted = df_pr.copy().dropna()
    pr_mean_tmp = (df_weighted["pr"] * 86400 * 0.001 * 1000 * df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum()
    plt.title(str(np.round(pr_mean_tmp)))
    plt.savefig(results_path + var + "_" + str(int(year)) + "_map.png", dpi=600, bbox_inches='tight')
    plt.close()

"""

# scatter plot
df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
df = pd.merge(df, df_domains, on=['lat', 'lon'], how='outer')
df = pd.merge(df, df_pr, on=['lat', 'lon'], how='outer')
df.rename(columns={'pr_median': 'Precipitation', 'pr_gswp3': 'Precipitation GSWP3', 'netrad_median': 'Net radiation',
                   'evap': 'Actual ET', 'qr': 'Groundwater recharge', 'qtot': 'Total runoff'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

df_weighted = df.copy().dropna()
print((df_weighted["Total runoff"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df["Total runoff"].mean())
print((df_weighted["Precipitation"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df["Precipitation"].mean())
#print((df_weighted["Precipitation GSWP3"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
#print(df["Precipitation GSWP3"].mean())

x_name = "Precipitation GSWP3"
y_name = "Total runoff"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_GRUN.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Total runoff"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_GRUN.png", dpi=600, bbox_inches='tight')
plt.close()

# count grid cells per climate region (dropna because nan values were not part of the scatter plot)
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts())
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts()/df.dropna()["domain_days_below_1_0.08_aridity_netrad"].count())
"""
