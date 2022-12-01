import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt

w = "rlds"

files = [
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19010101-19101231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19110101-19201231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19210101-19301231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19310101-19401231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19410101-19501231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19510101-19601231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19610101-19701231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19710101-19801231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19810101-19901231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_19910101-20001231.nc4",
w + "_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_20010101-20051231.nc4"]

reference_date = '1/1/1901'
m_start = (1901 - 1661) * 12
m_end = (2005 - 1661) * 12
#unit in kg m-2 s-1 each day
#conv = 86400*0.001*1000

def weighted_temporal_mean(ds, var):
    """
    weight by days in each mont adapted from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out

def re(path):
    date = xr.open_dataset(path)
    #d = weighted_temporal_mean(date, w)
    d = date.resample(time="1M").mean() # resample to monthly data
    return d

data = []
for f in files:
    data.append(re(f))

date = xr.concat(data, "time")
date = date.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))

date.to_netcdf(w + "_monthly_1975-2005.nc4")

