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

# check if folder exists
results_path = "results/gswp3/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

###
# load and process data
for decade in ['1971_1980', '1981_1990', '1991_2000', '2001_2010']:
    tmp = xr.open_dataset(r'./data/pr_gswp3-ewembi_' + decade + '.nc4')
    tmp = tmp.resample(time="1Y").sum()*86400*0.001*1000
    for i in range(0,len(tmp.time)):
        year = pd.to_datetime(tmp.time[i].values).year
        pr_year = tmp.sel(time=slice(dt(year, 1, 1), dt(year, 12, 31)))
        pr_year.to_netcdf(results_path + "average_GSWP3_" + str(year))

# load and process data
pr = xr.open_dataset(r'./data/pr_gswp3-ewembi_1941_1950.nc4')
#pr = weighted_temporal_mean(pr, "pr")
#pr.name = "pr"
pr = pr.resample(time="1Y").sum()
for decade in ['1951_1960', '1961_1970', '1971_1980', '1981_1990', '1991_2000', '2001_2010']:
    tmp = xr.open_dataset(r'./data/pr_gswp3-ewembi_' + decade + '.nc4')
    #tmp = weighted_temporal_mean(tmp, "pr")
    #tmp.name = "pr"
    tmp = tmp.resample(time="1Y").sum()
    pr = xr.merge([pr, tmp])

pr = pr.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
#pr = pr.sel(time=slice(dt(1945, 1, 1), dt(1974, 12, 31)))
pr = pr.mean("time")*86400*0.001*1000 # to mm/y

df_pr = pr.to_dataframe().reset_index().dropna()

pr.to_netcdf(results_path + "30y_average_GSWP3")
df_pr.to_csv(results_path + "30y_average_GSWP3.csv", index=False)

