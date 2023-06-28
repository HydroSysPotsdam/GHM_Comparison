import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from helpers import plotting_fcts
import xarray as xr

# This script calculates functional relationships between precipitation / net radiation and
# actual evapotranspiration / total runoff based on Budyko's equation.

# prepare data
data_path = "model_outputs/2b/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
df_greenland = pd.read_csv(data_path + "greenland.csv", sep=',') # greenland mask for plot

# check if folder exists
results_path = "results/budyko_theory/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

df_merged = pd.read_csv("model_outputs/2b/aggregated/domains.csv", sep=',')

def calc_Budyko(AI):
    # "Original" Budyko curve (Budyko, 1974)
    # AI = PET/P (or PET equivalent)
    # EI = AET/P
    EI = np.sqrt(AI*np.tanh(1./AI)*(1 - np.exp(-AI))) # AET/P
    return EI

# calculate evaporative fraction
sources = ["netrad"] # "potevap"
for source in sources:
    df_merged["evap_Budyko_"+source] = calc_Budyko(df_merged["aridity_"+source])*df_merged["pr_median"]
    df_merged["qtot_Budyko_"+source] = (1-calc_Budyko(df_merged["aridity_"+source]))*df_merged["pr_median"]

# get long format
df = pd.DataFrame()
for source in sources:
    df_tmp = df_merged[["lat", "lon", "pr_median", source+"_median", "domain_days_below_1_0.08_aridity_netrad", "aridity_"+source,"evap_Budyko_"+source,"qtot_Budyko_"+source]]
    df_tmp.columns = ["lat", "lon", "pr", "netrad", "domain_days_below_1_0.08_aridity_netrad", "aridity", "evap_Budyko", "qtot_Budyko"]
    df_tmp["source"] = source
    df = pd.concat([df, df_tmp])

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index() #.dropna()
df = pd.merge(df, df_area, on=['lat', 'lon'], how='outer')

# global averages need to be weighted due to different grid cell areas
df_weighted = df.loc[(df["source"]=="netrad")].copy().dropna()
print((df_weighted["evap_Budyko"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_weighted["evap_Budyko"].mean())
print((df_weighted["qtot_Budyko"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_weighted["qtot_Budyko"].mean())
print((df_weighted["pr"]*df_weighted["continentalarea"]).sum()/df_weighted["continentalarea"].sum())
print(df_weighted["pr"].mean())
print(df.loc[(df["source"]=="netrad"), "evap_Budyko"].mean())
print(df.loc[(df["source"]=="netrad"), "qtot_Budyko"].mean())
print(df.loc[(df["source"]=="netrad"), "qtot_Budyko"].mean() + df.loc[(df["source"]=="netrad"), "evap_Budyko"].mean())

# scatter plot
df.rename(columns={'pr': 'Precipitation', 'netrad': 'Net radiation',
                   'evap_Budyko': 'Actual evapotranspiration', 'qr': 'Groundwater recharge', 'qtot_Budyko': 'Total runoff'}, inplace=True)
df["dummy"] = ""
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
#palette = {"wet warm": '#4477AA', "dry warm": '#EE6677', "wet cold": '#66CCEE', "dry cold": '#CCBB44'}

df["sort_helper"] = df["domain_days_below_1_0.08_aridity_netrad"]
df["sort_helper"] = df["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
df = df.sort_values(by=["sort_helper"])

x_name = "Precipitation"
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
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Precipitation"
y_name = "Actual evapotranspiration"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
d = "domain_days_below_1_0.08_aridity_netrad"
g.set(xlim=[-100, 3100], ylim=[-100, 2100])
g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
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
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Actual evapotranspiration"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
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
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Precipitation"
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
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Precipitation"
y_name = "Total runoff"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
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
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
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
#g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_corr_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_scatterplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Net radiation"
y_name = "Total runoff"
x_unit = " [mm/yr]"
y_unit = " [mm/yr]"
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", col_wrap=4, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", alpha=1.0, s=1)
g.set(xlim=[-100, 2100], ylim=[-100, 2100])
#g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.map_dataframe(plotting_fcts.add_regression_domains, x_name, y_name, domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .715), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + x_name + '_' + y_name + "_regressionplot_Budyko_theory.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished scatterplot.")

# count grid cells per climate region (dropna because nan values were not part of the scatter plot)
print(df.dropna().loc[(df["source"]=="netrad", "domain_days_below_1_0.08_aridity_netrad")].value_counts())
print(df.dropna().loc[(df["source"]=="netrad", "domain_days_below_1_0.08_aridity_netrad")].value_counts()/
      df.dropna().loc[(df["source"]=="netrad", "domain_days_below_1_0.08_aridity_netrad")].count())

# save data to df
df.to_csv("data/" + "Budyko_prepared.csv", index=False)
