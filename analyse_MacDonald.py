import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from helpers import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from datetime import datetime as dt

# This script loads and analyses MacDonald recharge data.

# prepare data
data_path = "data/"

# check if folder exists
results_path = "results/macdonald/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(data_path + "Recharge_data_Africa_BGS.csv", sep=';')

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.Long, df.Lat)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest) # merge the datasets by index (for this, it is good to use '.join()' -function)

# scatter plot
df.rename(columns={'LTA_P_mmpa': 'Precipitation', 'netrad_median': 'Net radiation', 'pr_median': 'Precipitation HadGEM2-ES', 'pr_gswp3': 'Precipitation GSWP3',
                   'evap': 'Actual Evapotranspiration', 'qr': 'Groundwater recharge ISIMIP', 'qtot': 'Total runoff',
                   'Recharge_mmpa': 'Groundwater recharge'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}

df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

x_name = "Precipitation"
y_name = "Groundwater recharge"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=25)
n = 6
d = "domain_days_below_1_0.08_aridity_netrad"
#g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet warm", n=n)
g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry warm", n=n)
#g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="wet cold", n=n)
#g.map_dataframe(plotting_fcts.plot_lines_group, x_name, y_name, palette, domains=d, domain="dry cold", n=n)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_MacDonald.png", dpi=600, bbox_inches='tight')
plt.close()

sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=25)
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_MacDonald.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Groundwater recharge"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=25)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_MacDonald.png", dpi=600, bbox_inches='tight')
plt.close()

# count grid cells per climate region
print(df["domain_days_below_1_0.08_aridity_netrad"].value_counts())
print(df["domain_days_below_1_0.08_aridity_netrad"].value_counts()/df["domain_days_below_1_0.08_aridity_netrad"].count())

# save data to df
df.to_csv("data/" + "MacDonald_prepared.csv", index=False)
