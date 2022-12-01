import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt


files= [
"pr_watch-wfdei_1971_1980.nc4",
"pr_watch-wfdei_1981_1990.nc4",
"pr_watch-wfdei_1991_2000.nc4",
"pr_watch-wfdei_2001_2010.nc4",
"pr_watch-wfdei_2011_2016.nc4"
]

reference_date = '1/1/1901'
m_start = (1901 - 1661) * 12
m_end = (2005 - 1661) * 12
#unit in kg m-2 s-1 each day
conv = 86400*0.001*1000

def re(path):
    date = xr.open_dataset(path)
    date = date * conv # convert to mm/day
    d = date.resample(time="1M").sum() # resample to monthly data
    return d

data = []
for f in files:
    data.append(re(f))

date = xr.concat(data, "time")
date = date.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))

date.to_netcdf("precip_monthly_1975-2005.nc4")

