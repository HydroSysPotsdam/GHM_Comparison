import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt

w = "" #mean dealy t

files = [
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19010101-19101231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19110101-19201231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19210101-19301231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19310101-19401231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19410101-19501231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19510101-19601231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19610101-19701231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19710101-19801231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19810101-19901231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19910101-20001231.nc4",
"tas" + w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_20010101-20051231.nc4"]

reference_date = '1/1/1901'
m_start = (1901 - 1661) * 12
m_end = (2005 - 1661) * 12
#unit in kg m-2 s-1 each day
#conv = 86400*0.001*1000

temperatures = [1, 2.85, 6.7]

def re(path):
    date = xr.open_dataset(path)
    date = date - 272.15
    return date

data = []
for f in files:
    data.append(re(f))

date = xr.concat(data, "time")
date = date.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))

for t in temperatures:
    ds = date.where(date.tas <= t).groupby('time.year').count(dim='time')
    ds = ds.mean("year")
    d = ds.to_dataframe()
    print(d.head())
    d.to_csv("days_below_" + str(t) + ".csv")

