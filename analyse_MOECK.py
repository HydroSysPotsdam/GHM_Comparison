import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
from helpers import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
import xarray as xr
from datetime import datetime as dt

# This script loads and analyses Moeck recharge data.

# prepare data
data_path = "data/"

# check if folder exists
results_path = "results/moeck/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

"""
# test other P data
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

pr = pr.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
pr = pr.mean("time")
# transform into dataframe
df_pr = pr.to_dataframe().reset_index().dropna()
df_pr['pr_gswp3'] = df_pr['pr']*86400*0.001*1000 # to mm/y
#df_pr.columns = ['lat', 'lon', 'pr_gswp3']
"""

# load and process data
df = pd.read_csv(data_path + "global_groundwater_recharge_moeck-et-al.csv", sep=',')

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest) # merge the datasets by index (for this, it is good to use '.join()' -function)

"""
# test other P dataset
coord_list = [(x,y) for x,y in zip(df['lon'], df['lat'])]
pr_path = "D:/Data/CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
# open all datasets
pr = rio.open(pr_path, masked=True)
df['pr'] = [x for x in pr.sample(coord_list)]
# transform to np.arrays and rescale variables
df['pr'] = np.concatenate(df['pr'].to_numpy())
# remove nodata values
df.loc[df["pr"] > 50000, "pr"] = np.nan  # 65535
# transform values
df['pr'] = df['pr'] * 0.1
"""

# scatter plot
#df = pd.merge(df, df_pr, on=['lat', 'lon'], how='outer')
df.rename(columns={'pr_median': 'Precipitation HadGEM2-ES', 'netrad_median': 'Net radiation', 'pr_gswp3': 'Precipitation GSWP3',
                   'evap': 'Actual Evapotranspiration', 'qr': 'Groundwater recharge ISIMIP', 'qtot': 'Total runoff',
                   'Groundwater recharge [mm/y]': 'Groundwater recharge'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

x_name = "Precipitation GSWP3"
# because many R values are paired with exactly the same P value, this will help binning the data
df[x_name] = (1+0.001*np.random.rand(len(df[x_name])))*df[x_name]
# this will randomly place R values into different bins and thus leads to slightly different plots
y_name = "Groundwater recharge"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=5)
n = 6
d = "domain_days_below_1_0.08_aridity_netrad"
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
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Moeck.png", dpi=600, bbox_inches='tight')
plt.close()

sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=5)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_Moeck.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Groundwater recharge"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=5)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Moeck.png", dpi=600, bbox_inches='tight')
plt.close()

# count grid cells per climate region
print(df["domain_days_below_1_0.08_aridity_netrad"].value_counts())
print(df["domain_days_below_1_0.08_aridity_netrad"].value_counts()/df["domain_days_below_1_0.08_aridity_netrad"].count())

# mean value
print("mean value:")
print(str(np.round(df["Groundwater recharge"].mean(),2)))

# rank correlations between forcing variables
domains = ["wet warm", "dry warm", "wet cold", "dry cold"]
print("Rank correlations between precipitation and net radiation")
for d in domains:
    r_sp, _ = stats.spearmanr(df.loc[df["domain_days_below_1_0.08_aridity_netrad"] == d, "Precipitation GSWP3"],
                              df.loc[df["domain_days_below_1_0.08_aridity_netrad"] == d, "Net radiation"], nan_policy='omit')
    corr_str = d + ": " + r' $\rho_s$ ' "= " + str(np.round(r_sp, 2))
    print(corr_str)

# save data to df
df.to_csv("data/" + "Moeck_prepared.csv", index=False)
