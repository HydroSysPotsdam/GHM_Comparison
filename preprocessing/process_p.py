import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt


files = ["pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_landonly_19710101-19801231.nc4","pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_landonly_19810101-19901231.nc4","pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_landonly_19910101-20001231.nc4","pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_landonly_20010101-20051231.nc4"]

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

