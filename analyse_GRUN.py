import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xarray as xr
from helpers import plotting_fcts
from datetime import datetime as dt
from helpers.weighted_mean import weighted_temporal_mean

# This script loads and analyses GRUN data.

# prepare data
data_path = "D:/Data/GRUN/"

# check if folder exists
results_path = "results/grun/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
file_path = "GRUN_v1_GSWP3_WGS84_05_1902_2014.nc"

# get multi annual averages
var = "Runoff"
data = xr.open_dataset(data_path + file_path)
d = weighted_temporal_mean(data, "Runoff")
d.name = "Runoff"
d = d.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
d = d.mean("time")
print(d.mean().values)

# transform into dataframe
df = d.to_dataframe().reset_index().dropna()
df[var] = df[var] * 365 # mm/d to mm/y
df.columns = ['lat', 'lon', 'qtot']

# plot map
var = "qtot"
plotting_fcts.plot_map(df["lon"], df["lat"], df[var], " [mm/y]", var, np.linspace(0, 2000, 11))
ax = plt.gca()
ax.coastlines(linewidth=0.5)
plt.savefig(results_path + var + "_map.png", dpi=600, bbox_inches='tight')
plt.close()

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()
df = pd.merge(df, df_area, on=['lat', 'lon'], how='outer')

# scatter plot
df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
df = pd.merge(df, df_domains, on=['lat', 'lon'], how='outer')
#df = pd.merge(df, df_pr, on=['lat', 'lon'], how='outer')
df = df.dropna()
df.rename(columns={'pr_median': 'Precipitation HadGEM2-ES', 'pr_gswp3': 'Precipitation GSWP3', 'netrad_median': 'Net radiation',
                   'evap': 'Actual ET', 'qr': 'Groundwater recharge', 'qtot': 'Total runoff'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

df_weighted = df.copy().dropna()
print(np.round((df_weighted["Total runoff"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(), 2))
print(np.round(df["Total runoff"].mean(), 2))
print(np.round((df_weighted["Precipitation GSWP3"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(), 2))
print(np.round(df["Precipitation GSWP3"].mean(), 2))
#print(np.round((df_weighted["Precipitation GSWP3"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum(), 2))
#print(dnp.round(f["Precipitation GSWP3"].mean(), 2))

domains = ["wet warm", "dry warm", "wet cold", "dry cold"]
for d in domains:
    df_tmp = df.loc[(df["domain_days_below_1_0.08_aridity_netrad"] == d)]
    print(d)
    print(np.round((df_tmp["Total runoff"]*df_tmp["continentalarea"]).sum()/df_tmp["continentalarea"].sum(), 2))
    print(np.round((df_tmp["Precipitation GSWP3"]*df_tmp["continentalarea"]).sum()/df_tmp["continentalarea"].sum(), 2))

x_name = "Precipitation GSWP3"
y_name = "Total runoff"
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

sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_GRUN.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Total runoff"
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

# save data to df
df.to_csv("data/" + "GRUN_prepared.csv", index=False)
