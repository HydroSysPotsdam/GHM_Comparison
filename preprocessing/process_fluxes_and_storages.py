import xarray as xr
import pandas as pd
from datetime import datetime as dt
from os.path import exists as file_exists
import os
import regionmask

from functools import reduce

# signatures and the variables needed to compute them

# timestep daily files are in 10 pieces; first file 1861 - 1870, 1871 - 1880
# All variables avilable in ISIMIP are listed here
fluxes = [
        {"name": "dis", "description": "Discharge", "timestep": "daily", "unit": "m3 s-1"},
        {"name": "evap", "description": "Evapotranspiration", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "lai", "description": "Leaf Area Index", "timestep": "monthly", "unit": "1"},
        {"name": "maxdis", "description": "Monthly max of daily discharge", "timestep": "monthly", "unit": "m3 s-1"},
        {"name": "mindis", "description": "Monthly min of daily discharge", "timestep": "monthly", "unit": "m3 s-1"},
        {"name": "qs", "description": "Surface runoff", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "qsb", "description": "Subsurface runoff", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "qtot", "description": "Runoff (qs + qsb) ", "timestep": "daily", "unit": "kg m-2 s-1"},
        
        # WaterUse
        {"name": "adomww", "description": "Domestic water withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "airrww", "description": "Actual irrigation withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "amanww", "description": "Actual manufacturing withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "pdomuse", "description": "Potential domestic water consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "adomuse", "description": "Actual domestic water consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "pirruse", "description": "Potential irrigation water use", "timestep": "monthly", "unit": "kg m-2 s-1" },
        {"name": "pirrusegreen", "description": "Potential green water consumption for irrigated cropland", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "pirrww", "description": "Potential rrigation withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "pmanuse", "description": "Potential manufactoring water use", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "admeluse", "description": "", "timestep": "annual"},
        {"name": "airruse", "description": "Actual irrigation consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "airrusegreen", "description": "Actual green water use for irrigation of cropland", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "airainfusegreen", "description": "Actual green water on rainfed cropland", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "ainduse", "description": "Actual industrial water use", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "aindww", "description": "Actual industrial water withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "aliveuse", "description": "Actual livestock consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "aliveww", "description": "Actual livestocl withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "atotuse", "description": "total water consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "ptotuse", "description": "Total potential consumption", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "ptotww", "description": "Total potential water withdrawl", "timestep": "monthly", "unit": "kg m-2 s-1"},

        # Groundwater 
        {"name": "qg", "description": "Groundwater runoff", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "qr", "description": "Groundwater recharge", "timestep": "monthly", "unit": "kg m-2 s-1"},
        {"name": "qint", "description": "", "timestep": "monthly"},
        
        # Other
        {"name": "rainf", "description": "", "timestep": "monthly"},
        {"name": "snd", "description": "Snow depth", "timestep": "monthly", "unit": "m"},
        {"name": "snowf", "description": "", "timestep": "monthly"},
        {"name": "thawdepth", "description": "Annual max faily thaw depth", "timestep": "annual", "unit": "m"},
        {"name": "tsl", "description": "Soil temperature", "timestep": "daily", "unit": "K"},
        {"name": "tsl", "description": "Soil temperature", "timestep": "monthly", "unit": "K"},
        {"name": "potevap", "description": "Potential evap", "timestep": "monthly", "unit": "kg m-2 s-1"},
        ]

storages = [ 
        {"name": "rootmoist", "description": "Soil moiture root zone", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "soilmoist", "description": "Soil moisture", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "soilmoistfroz", "description": "Frozen soil moisture", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "swe", "description": "Snow water equivalent", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "tws", "description": "Total water storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "canopystor", "description": "Canopy storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "lakestor", "description": "Lake storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "wetlandstor", "description": "Wetland storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "reservoirstor", "description": "Seservoir storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "riverstor", "description": "River storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "groundwstor", "description": "Groundwater storage", "timestep": "monthly", "unit": "kg m-2"} 
        ]



# This script processes years from s - e

path = "../hydro/data/ISIMIP/2b/outputs"
time_s = "1975"
time_e = "2005"
start_y = 1975
end_y = 2005
reference_date = '1/1/1975'

# Histrocial data goes from 1861 to 2005
# Not all models have outputs for all fluxes -> this is ignored here


# example file string  clm45_hadgem2-es_ewembi_historical_2005soc_co2_dis_global_daily_1861_1870.nc4

soc =  {"clm45": "2005soc",
        "h08": "histsoc", # also 2005
        "jules-w1": "nosoc",
        "lpjml": "histsoc",
        "matsiro": "histsoc",
        "pcr-globwb": "histsoc",
        "watergap2": "histsoc",
        "cwatm": "histsoc"
        }

# Selected for this study
ghms = ["clm45", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2","h08"]
#ghms = ["cwatm"] #! process only alone becaause of necessary switches
gcm = "hadgem2-es"

# months since 1661 (ISMIP uses 1601) -> issue: causes out of bounds in datetime
m_start = (start_y - 1661) * 12
m_end = (2005 - 1661) * 12
# Thus hist data = 01/01/1861 = 2400 months since 1661
d_start = (start_y - 1661) * 360
d_end = (2005 - 1661) * 360


# isimip flux unit: kg m-2 s-1 = kg pro m2 per second
# gives mm/month:
conv = 86400*0.001*1000*30

# continental areas in km2
filen = "watergap_22d_continentalarea.nc4"
areas = xr.open_dataset(filen, decode_times=False) 
areas = areas.mean("time")

def load_precip():
    f = 'precip_monthly_1975-2005.nc4'
    date = xr.open_dataset("../hydro/data/ISIMIP/2b/climate/" + f)
    d = date.resample(time="1Y").sum() # calculate mm/year
    d = d.mean("time")
    return d

def load_inputs(variables, storages=False):
    global m_start
    global m_end
    global start_y
    global end_y
    global reference_date
    read_precip = True
    data_all_models = {}
    # iterating all hydrological models
    for ghm in ghms:
        data = {}
        # iterate all possible fluxes or storages
        for v in variables:
            if v["name"] == "pr":
                #print("Should P data be read? {}".format(read_precip))
                if read_precip:
                    data["pr"] = load_precip()
                    print("Read precip")
                    read_precip = False
                    continue
                else:
                    continue
            if v["name"] == "qtot" and ghm == "cwatm":
                continue # not uploaded to isimip ignore
            start = "1861"
            end = "2005"
            if ghm == "cwatm":
                start = "1979"
                end = "2005"
                start_y = 1979
                end_y = 2005
                m_start = 0
                m_end = 324
                reference_date = '1/1/1979'

            print("Opening variable {} from model {} at {} timesteps at year {} - {}".format(v["name"], ghm, v["timestep"], start, end))
            p = path + '/' + ghm + '_' + gcm + '_ewembi_historical_' + soc[ghm] + '_co2_' + v["name"] + '_global_' + v["timestep"] + '_' + start + '_' + end + '.nc4'

            if ghm == "cwatm":
                p = path + '/' + ghm + '_' + gcm + '_historical_' + soc[ghm] + '_et-pm_' + v["name"] + '_' + v["timestep"] + '_' + start + '_' + end + '.nc4'
            if v["timestep"] == "monthly":
                try:
                    date = xr.open_dataset(p, decode_times=False)
                    print("Success in reading file")
                    print("Calculating mean")
                    # Convert to a unit that makes sense
                    if not storages:
                        date = date * conv # mm/month only convert fluxes
                    # The following is necessary to deal with strange time format of ISIMIP
                    if not ghm == "cwatm": # no need to cut
                        date = date.sel(time=slice(m_start, m_end))
                    #units, year = date.time.attrs['units'].split('since')
                    date['time'] = pd.date_range(start=reference_date, periods=date.sizes['time'], freq='MS')
                    # Select a time range 
                    date = date.sel(time=slice(dt(start_y, 1, 1), dt(end_y, 12, 31)))
                    # Resample to yearly values
                    if not storages:
                        d = date.resample(time="1Y").sum()
                    else:
                        d = date.resample(time="1Y").mean()
                    # Calculate the mean over time period
                    d = d.mean("time")
                    data[v["name"]] = d
                    date = None #force python to free up space
                    d = None
                    print("done")

                except FileNotFoundError:
                    print("Does not exist")
            elif v["timestep"] == "annual":
                try:
                    date = xr.open_dataset(p, decode_times=False)
                    print("Success in reading file")
                    print("Calculating mean")
                    # Convert here if necessary
                    #date = date * X
                    # The following is necessary to deal with strange time format of ISIMIP
                    #date = date.sel(time=slice(m_start, m_end))
                    #units, year = date.time.attrs['units'].split('since')
                    date['time'] = pd.date_range(start=reference_date, periods=date.sizes['time'], freq='Y')
            
                    # Select a time range 
                    date = date.sel(time=slice(dt(start_y, 1, 1), dt(end_y, 12, 31)))
                    # Calculate the mean over time period
                    d = d.mean("time")
                    data[v["name"]] = d
                    date = None #force python to free up space
                    d = None
                    print("done")
                except FileNotFoundError:
                    print("Does not exist")
            else:
                # daily timestep
                tmp_d = []
                start_year = int(time_s) 
                if start_year % 10 != 1:
                    # start year is not a ten -> files only start 01
                    start_year = start_year - (start_year % 10) + 1
                for i in range(3):
                    ts = start_year + i * 10
                    te = ts + 10 - 1 # end of 10 year period
                    if te > 2005:
                        te = 2005 # last year goes only till 2005
                    p = path + '/' + ghm + '_' + gcm + '_ewembi_historical_' + soc[ghm] + '_co2_' + v["name"] + '_global_' + v["timestep"] + '_' + str(ts) + '_' + str(te) + '.nc4'    
                    print(p)
                    try:
                        date = xr.open_dataset(p, decode_times=False)
                        print("Success in reading file")
                        print("Calculating mean")
                        # Convert to a unit that makes sense
                        if v["name"] == "dis":
                            date = date * 86400 # s -> d
                        else:
                            date = date * 86400*0.001*1000 # mm/day 
                        # The following is necessary to deal with strange time format of ISIMIP
                        #date = date.sel(time=slice(m_start, m_end))
                        #units, year = date.time.attrs['units'].split('since')
                        date['time'] = pd.date_range(start="1/1/" + str(ts), periods=date.sizes['time'], freq='D')
                        tmp_d.append(date)
                    except FileNotFoundError:
                        print("Does not exist")
                d = xr.concat(tmp_d, dim="time")
                # resample as yearly values
                d = date.resample(time="1Y").sum()
                d = d.sel(time=slice(dt(start_y, 1, 1), dt(end_y, 12, 31)))
                # Calculate the mean over time period
                d = d.mean("time")
                if v["name"] == "dis":
                    # "dis" is in m3 s-1; area in km2
                    # m3 y-1 / m2 = m/y
                    # m * 1000 = mm
                    d["area"] = areas.continentalarea
                    d["dis"] = (d.dis / (d.area * 1000000) ) * 1000
                data[v["name"]] = d
                d = None
                print("done")
        # data contains all variables for a GHM as yearly means
        if data is not None:
            data_all_models[ghm] = data
    return data_all_models


# get only a subset of the storages
s =  [ 
        {"name": "swe", "description": "Snow water equivalent", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "lakestor", "description": "Lake storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "wetlandstor", "description": "Wetland storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "reservoirstor", "description": "Seservoir storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "riverstor", "description": "River storage", "timestep": "monthly", "unit": "kg m-2"},
        {"name": "groundwstor", "description": "Groundwater storage", "timestep": "monthly", "unit": "kg m-2"} 
        ]

# Landmask of ISIMIP to cut data to
p = "../hydro/data/ISIMIP/2b/ISIMIP2b_landseamask_generic.nc4"

mask = xr.open_dataset(p) 

# Read either the fluxes or the storages
dfs = load_inputs(fluxes)
#dfs = load_inputs(s, True)

# The actual loop to get the data and write them to file
for model in dfs:
    dfList = list(dfs[model].values())
    if not os.path.exists(model):
        os.makedirs(model)
    for var in dfList:
        try:
            data_l = var.where(mask.LSM == 1)
            # Mask out Greenland
            mask_regions = regionmask.defined_regions.ar6.land.mask_3D(data_l)
            mask_not_greenland = mask_regions.loc[mask_regions.names != 'Greenland/Iceland']
            d = data_l.where(mask_not_greenland)
            d = d.drop(["abbrevs", "names"])
            df = d.to_dataframe().dropna(how='all')
            df.reset_index(inplace=True)
            df.drop(["time", "region"], axis=1, inplace=True)
            p = model + "/" + df.columns[2] + ".csv"
            print("Writing model {} with var {}".format(model, df.columns[2]))
            df.to_csv(p, index=False)
        except ValueError:
            continue

