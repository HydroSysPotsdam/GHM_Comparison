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

df_tot = df_tot.dropna()

# plot map
var = "Rn"
plotting_fcts.plot_map(df_tot["lon"], df_tot["lat"], df_tot[var], " [mm/y]", var, np.linspace(0, 2000, 11))
#plotting_fcts.mask_greenland("2b/aggregated/")
ax = plt.gca()
ax.coastlines(linewidth=0.5)
plt.savefig(results_path + var + "_map.png", dpi=600, bbox_inches='tight')
plt.close()

# test other P data
pr = xr.open_dataset(r'./data/pr_gswp3-ewembi_1971_1980.nc4')
#pr = weighted_temporal_mean(pr, "pr")
#pr.name = "pr"
pr = pr.resample(time="1Y").sum()
for decade in ['1981_1990', '1991_2000', '2001_2010', '2011_2016']:
    tmp = xr.open_dataset(r'./data/pr_gswp3-ewembi_' + decade + '.nc4')
    #tmp = weighted_temporal_mean(tmp, "pr")
    #tmp.name = "pr"
    tmp = tmp.resample(time="1Y").sum()
    pr = xr.merge([pr, tmp])

pr = pr.sel(time=slice(dt(2001, 1, 1), dt(2015, 12, 31)))
pr = pr.mean("time")

# transform into dataframe
df_pr = pr.to_dataframe().reset_index().dropna()
df_pr['pr_gswp3'] = df_pr['pr']*86400*0.001*1000 # to mm/y
df_tot = pd.merge(df_tot, df_pr, on=['lat', 'lon'], how='outer')

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()
df_tot = pd.merge(df_tot, df_area, on=['lat', 'lon'], how='outer')

# global averages need to be weighted due to different grid cell areas
df_weighted = df_tot.copy().dropna()
print((df_weighted["H"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_tot["H"].mean())
print((df_weighted["LE"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_tot["LE"].mean())
print((df_weighted["Rn"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_tot["Rn"].mean())

# scatter plot
df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')
df = pd.merge(df_tot, df_domains, on=['lat', 'lon'], how='outer')
df.rename(columns={'pr_median': 'Precipitation', 'pr_gswp3': 'Precipitation GSWP3', 'netrad_median': 'Net radiation ISIMIP',
                   'evap': 'Actual ET', 'qr': 'Groundwater recharge', 'qtot': 'Total runoff',
                   'LE': 'Actual evapotranspiration', 'H': 'Sensible heat', 'Rn': 'Net radiation'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

x_name = "Precipitation GSWP3"
y_name = "Actual evapotranspiration"
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
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_FLUXCOM.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation ISIMIP"
y_name = "Actual evapotranspiration"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
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

# count grid cells per climate region (dropna because nan values were not part of the scatter plot)
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts())
print(df.dropna()["domain_days_below_1_0.08_aridity_netrad"].value_counts()/df.dropna()["domain_days_below_1_0.08_aridity_netrad"].count())
