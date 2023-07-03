import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from helpers import plotting_fcts
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import xarray as xr
import matplotlib.ticker as mticker

# This script calculates the climate regions (here domains) in different ways
# and saves the one that is used to a csv file.

# prepare data
data_path = "model_outputs/2b/aggregated/"  # "outputs2a/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
df_greenland = pd.read_csv(data_path + "greenland.csv", sep=',') # greenland mask for plot

### netrad ###
# adjust variables and colours
var_name_list = ["netradiation_watergap", "netradiation_cwatm", "netradiation_h08",
                 "netradiation_jules", "netradiation_matsiro_withoutlakes", "netradiation_clm45_minus"]
var_unit_list = [' [mm/y]', ' [mm/y]', ' [mm/y]', ' [mm/y]', ' [mm/y]', ' [mm/y]']

# check if folder exists
results_path = "results/domains/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

df_merged = pd.read_csv(data_path + "tasmin.csv", sep=',')
df_tmp = pd.read_csv(data_path + "pr.csv", sep=',')
df_tmp.columns = ["lat", "lon", "pr_median"] # dummy pr value, used only in domain file
df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer')
# new code: add GSWP3
df_tmp = pd.read_csv("results/gswp3/30y_average_GSWP3.csv", sep=',')
df_tmp.columns = ["lon", "lat", "pr_gswp3"] #
df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer').dropna()
for var_name in var_name_list:  # loop over all models
    df_tmp = pd.read_csv(data_path + var_name + ".csv", sep=',')
    df_tmp.columns = ["lat", "lon", var_name]
    df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer')

n=5 # existing columns
df_merged["netrad_median"] = df_merged.iloc[:,n:n+len(var_name_list)].median(axis=1)*12.87 # median to account for outliers
df_merged["aridity_netrad"] = df_merged["netrad_median"]/df_merged["pr_median"]
df_merged["aridity_netrad_gswp3"] = df_merged["netrad_median"]/df_merged["pr_gswp3"]
df_aridity_netrad = df_merged[["lat","lon","pr_median","pr_gswp3","netrad_median","aridity_netrad","aridity_netrad_gswp3"]]

### potevap ###
ghms = ["watergap2", "h08", "lpjml", "pcr-globwb", "matsiro"] # "matsiro" is ET

df_merged = pd.read_csv(data_path + "tasmin.csv", sep=',')
df_tmp = pd.read_csv(data_path + "pr.csv", sep=',')
df_tmp.columns = ["lat", "lon", "pr_median"] # dummy pr value, used only in domain file
df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer')
# new code: add GSWP3
df_tmp = pd.read_csv("results/gswp3/30y_average_GSWP3.csv", sep=',')
df_tmp.columns = ["lon", "lat", "pr_gswp3"] #
df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer').dropna()
for ghm in ghms:  # loop over all models
    df_tmp = pd.read_csv(data_path + ghm + "/potevap.csv", sep=',')
    df_tmp.columns = ["lat", "lon", "potevap_"+ghm]
    df_merged = pd.merge(df_merged, df_tmp, on=['lat', 'lon'], how='outer')

n=5 # existing columns
df_merged["potevap_median"] = df_merged.iloc[:,n:n+len(ghms)].median(axis=1) # median to account for outliers
df_merged["aridity_potevap"] = df_merged["potevap_median"]/df_merged["pr_median"]
df_merged["aridity_potevap_gswp3"] = df_merged["potevap_median"]/df_merged["pr_gswp3"]
df_aridity_potevap = df_merged[["lat","lon","potevap_median","aridity_potevap","aridity_potevap_gswp3"]]

### temperature ###
df_temperature = pd.read_csv(data_path + "days_below_6.7.csv", sep=',')
df_tmp = pd.read_csv(data_path + "days_below_2.85.csv", sep=',')
df_temperature = pd.merge(df_temperature, df_tmp, on=['lat', 'lon'], how='outer')
df_tmp = pd.read_csv(data_path + "days_below_1.csv", sep=',')
df_temperature = pd.merge(df_temperature, df_tmp, on=['lat', 'lon'], how='outer')
df_temperature.columns = ["lat", "lon", "days_below_6.7", "days_below_2.85", "days_below_1"]

# get single dataframe
df_domains = pd.merge(df_aridity_netrad, df_aridity_potevap, on=['lat', 'lon'], how='outer')
df_domains = pd.merge(df_domains, df_temperature, on=['lat', 'lon'], how='outer')
df_domains = df_domains.dropna().reset_index()

print("Finished data preparation.")

# create figure
# 4 cases: aridity from netrad and potevap and wet/cold for 1 month below 1 degree and 3 months below 6.7 degrees
days_below = ["days_below_1", "days_below_6.7"]
thresh = [1/12, 3/12]

for i in [0,1]:

    for aridity in ["aridity_netrad", "aridity_potevap", "aridity_netrad_gswp3", "aridity_potevap_gswp3"]: #

        plt.rcParams['axes.linewidth'] = 0.1
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        str_tmp = "domain_" + str(days_below[i]) + "_" + str(np.round(thresh[i],2)) + "_" + aridity
        df_domains[str_tmp] = np.nan
        df_domains.loc[np.logical_and(df_domains[days_below[i]] <= thresh[i]*365.27, df_domains[aridity] <= 1), str_tmp] = "wet warm" # energy-limited and warm
        df_domains.loc[np.logical_and(df_domains[days_below[i]] <= thresh[i]*365.27, df_domains[aridity] > 1), str_tmp] = "dry warm" # water-limited and warm
        df_domains.loc[np.logical_and(df_domains[days_below[i]] > thresh[i]*365.27, df_domains[aridity] <= 1), str_tmp] = "wet cold" # energy-limited and cold
        df_domains.loc[np.logical_and(df_domains[days_below[i]] > thresh[i]*365.27, df_domains[aridity] > 1), str_tmp] = "dry cold" # water-limited and cold

        frac_1 = len(df_domains.loc[df_domains[str_tmp] == "wet warm"])/len(df_domains[str_tmp].dropna())
        frac_2 = len(df_domains.loc[df_domains[str_tmp] == "dry warm"])/len(df_domains[str_tmp].dropna())
        frac_3 = len(df_domains.loc[df_domains[str_tmp] == "wet cold"])/len(df_domains[str_tmp].dropna())
        frac_4 = len(df_domains.loc[df_domains[str_tmp] == "dry cold"])/len(df_domains[str_tmp].dropna())
        print(frac_1 + frac_2 + frac_3 + frac_4)

        print(df_domains.loc[df_domains[str_tmp] == "wet warm", "aridity_netrad"].median())
        print(df_domains.loc[df_domains[str_tmp] == "wet cold", "aridity_netrad"].median())
        print(df_domains.loc[df_domains[str_tmp] == "dry cold", "aridity_netrad"].median())
        print(df_domains.loc[df_domains[str_tmp] == "dry warm", "aridity_netrad"].median())

        df_domains["sort_helper"] = df_domains[str_tmp]
        df_domains["sort_helper"] = df_domains["sort_helper"].replace({'wet warm': 0, 'wet cold': 1, 'dry cold': 2, 'dry warm': 3})
        df_domains = df_domains.sort_values(by=["sort_helper"])

        sc = ax.scatter(df_domains.loc[df_domains[str_tmp] == "wet warm", "lon"], df_domains.loc[df_domains[str_tmp] == "wet warm", "lat"],
                        marker='s', s=.35,  c='#018571', edgecolors='none', transform=ccrs.PlateCarree(),
                        label="Wet & warm (" + str(np.round(frac_1*100)/100) + ")")
        sc = ax.scatter(df_domains.loc[df_domains[str_tmp] == "wet cold", "lon"], df_domains.loc[df_domains[str_tmp] == "wet cold", "lat"],
                        marker='s', s=.35,  c='#80cdc1', edgecolors='none', transform=ccrs.PlateCarree(),
                        label="Wet & cold (" + str(np.round(frac_3*100)/100) + ")")
        sc = ax.scatter(df_domains.loc[df_domains[str_tmp] == "dry cold", "lon"], df_domains.loc[df_domains[str_tmp] == "dry cold", "lat"],
                        marker='s', s=.35,  c='#dfc27d', edgecolors='none', transform=ccrs.PlateCarree(),
                        label="Dry & cold (" + str(np.round(frac_4*100)/100) + ")")
        sc = ax.scatter(df_domains.loc[df_domains[str_tmp] == "dry warm", "lon"],
                        df_domains.loc[df_domains[str_tmp] == "dry warm", "lat"],
                        marker='s', s=.35, c='#a6611a', edgecolors='none', transform=ccrs.PlateCarree(),
                        label="Dry & warm (" + str(np.round(frac_2 * 100) / 100) + ")")

        box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-70)
        x0, y0, x1, y1 = box.bounds
        ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
        plt.gca().outline_patch.set_visible(False)
        plt.legend(markerscale=8, fontsize=6)
        plotting_fcts.mask_greenland(data_path)
        ax = plt.gca()

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

        plt.savefig(results_path + days_below[i] + "_" + str(np.round(thresh[i],2)) + "_" + aridity + "_map.png", dpi=600,  bbox_inches='tight')
        plt.close()

print("Finished maps.")

# save dataframes
df_domains = df_domains.drop(["days_below_1", "days_below_2.85", "days_below_6.7"], axis=1)
df_domains.to_csv("model_outputs/2b/aggregated/domains.csv", index=False)

# create dummy netrad file for models without netrad
df_netrad = df_domains[["lat", "lon", "netrad_median"]]
df_netrad.columns = ["lat", "lon", "netrad"]
df_netrad["netrad"] = df_netrad["netrad"] / 12.87
df_netrad.to_csv("model_outputs/2b/aggregated/netrad_median.csv", index=False)

# create dummy potevap file for models without potevap
df_potevap = df_domains[["lat", "lon", "potevap_median"]]
df_potevap.columns = ["lat", "lon", "potevap"]
df_potevap.to_csv("model_outputs/2b/aggregated/potevap_median.csv", index=False)

# grid cell areas
area = xr.open_dataset("model_outputs/2b/aggregated/watergap_22d_continentalarea.nc4", decode_times=False)
df_area = area.to_dataframe().reset_index().dropna()
df_domains = pd.merge(df_domains, df_area, on=['lat', 'lon'], how='outer').dropna()
print("Fraction of grid cells")
print("Wet warm")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="wet warm", "continentalarea"].count()) /
          df_domains["continentalarea"].count()))
print("Wet cold")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="wet cold", "continentalarea"].count()) /
          df_domains["continentalarea"].count()))
print("Dry cold")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="dry cold", "continentalarea"].count()) /
          df_domains["continentalarea"].count()))
print("Dry warm")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="dry warm", "continentalarea"].count()) /
          df_domains["continentalarea"].count()))
print("Total")
print(str(np.round(df_domains["continentalarea"].count())))

print("Fraction of continental area")
print("Wet warm")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="wet warm", "continentalarea"].sum()) /
          df_domains["continentalarea"].sum()))
print("Wet cold")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="wet cold", "continentalarea"].sum()) /
          df_domains["continentalarea"].sum()))
print("Dry cold")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="dry cold", "continentalarea"].sum()) /
          df_domains["continentalarea"].sum()))
print("Dry warm")
print(str(np.round(df_domains.loc[df_domains["domain_days_below_1_0.08_aridity_netrad"]=="dry warm", "continentalarea"].sum()) /
          df_domains["continentalarea"].sum()))
print("Total")
print(str(np.round(df_domains["continentalarea"].sum())))


# scatter plot
palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df_domains, col="domain_days_below_1_0.08_aridity_netrad", col_wrap=2, palette=palette)
g.map_dataframe(plotting_fcts.plot_coloured_scatter_random_domains, "pr_median", "netrad_median", domains="domain_days_below_1_0.08_aridity_netrad", alpha=1, s=1)
g.map(plotting_fcts.plot_origin_line, "pr_median", "netrad_median")
g.map_dataframe(plotting_fcts.add_corr_domains, "pr_median", "netrad_median", domains="domain_days_below_1_0.08_aridity_netrad", palette=palette)
g.set(xlabel="Precipitation [mm/y]", ylabel="Net radiation [mm/y]")
g.set_titles(col_template='{col_name}')
g.set(xlim=[0, 3000], ylim=[0, 3000])
g.tight_layout()
sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
for axes in g.axes.ravel():
    axes.legend(loc=(.0, .85), handletextpad=0.0, frameon=False, fontsize=9, labelspacing=0)
g.savefig(results_path + "pr" + '_' + "netrad" + "_scatterplot.png", dpi=600, bbox_inches='tight')
plt.close()
