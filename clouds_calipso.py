import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import xesmf as xe

# =========== LOAD ERA5, MAR_noBS, MAR_BS, Cloudsat =======================


def preprocess(ds):
    data = ds.sel(lat=slice(-90, -40))
    return data


file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly/'
file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'

file_str_ERA = '/projects/NS9600K/shofer/blowing_snow/MAR/ERA5/SINGLELEVS/'

file_str_calipso = '/projects/NS9600K/shofer/blowing_snow/sat_data/cloudsat/'

file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'
# Cloud cover years: 2006/07 to 2011/02
year_s = '2006-07-01'
year_e = '2011-02-18'

# Open the no driftig snow file
ds_bs_CC = (xr.open_dataset(
    file_str + 'mon-CC-MAR_ERA5-1979-2019.nc').sel(TIME=slice(year_s, year_e)).rename(
        {'X': 'x', 'Y': 'y'}))
ds_nobs_CC = (xr.open_dataset(
    file_str_nobs + 'mon-CC-MAR_ERA5-1979-2019.nc').sel(TIME=slice(year_s, year_e)).rename(
        {'X': 'x', 'Y': 'y'}))

# .isel(x=slice(10, -10), y=slice(10, -10))
data_ERA = xr.open_mfdataset(
    file_str_ERA + 'ERA5_*.nc', combine='by_coords')
data_ERA = data_ERA.rename({'latitude': 'lat', 'longitude': 'lon'}).sel(
    time=slice(year_s, year_e))
cloud_data_ERA = data_ERA.tcc

ds_cloudsat = xr.open_dataset(
    file_str_calipso + 'cf_2deg_cloudsat_calipso_total_annual.nc')

# ==========================================================================
# CREATE the ICE MASK
# =========================================================

MAR_grid = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values),
                      'ICE': (['y', 'x'], MAR_grid.ICE.values),
                      'AIS': (['y', 'x'], MAR_grid.AIS.values),
                      'GROUND': (['y', 'x'], MAR_grid.GROUND.values),
                      'AREA': (['y', 'x'], MAR_grid.AREA.values),
                      'ROCK': (['y', 'x'], MAR_grid.ROCK.values)},
                     coords={'x': (['x'], ds_nobs_CC.x),
                             'y': (['y'], ds_nobs_CC.y)})

ais = ds_grid['AIS'].where(ds_grid)['AIS'] > 0  # Only AIS=1, other islands  =0
# Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
ice = ds_grid['ICE'].where(ds_grid['ICE'] > 30)
# Combine ais + ice/100 * factor area for taking into account the projection
ice_msk = (ais * ice * ds_grid['AREA'] / 100)

grd = ds_grid['GROUND'].where(ds_grid['GROUND'] > 30)
grd_msk = (ais * grd * ds_grid['AREA'] / 100)

lsm = (ds_grid['AIS'] < 1)
ground = (ds_grid['GROUND'] * ds_grid['AIS'] > 30)

shf = (ds_grid['ICE'] / ds_grid['ICE']).where((ds_grid['ICE'] > 30) &
                                              (ds_grid['GROUND'] < 50) & (ds_grid['ROCK'] < 30) & (ais > 0))
shelf = (shf > 0)

x2D, y2D = np.meshgrid(ds_grid['x'], ds_grid['y'])
sh = ds_grid['SH']

dh = (ds_grid['x'].values[0] - ds_grid['x'].values[1]) / 2.


# =========== COMPUTE THE MEAN OVER THE SAME TIME PERIOD ==================
BS = ds_bs_CC.mean(dim='TIME')
# Add LAT LON to MAR data
BS['lat'] = ds_grid.LAT
BS['lon'] = ds_grid.LON
BS['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)

NOBS = ds_nobs_CC.mean(dim='TIME')
NOBS['lat'] = ds_grid.LAT
NOBS['lon'] = ds_grid.LON
NOBS['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)


ERA = cloud_data_ERA.mean(dim='time')

cloudsat = ds_cloudsat.cf


# =========== REGRID TO THE SAME GRID =====================================
# This creates the output grid, atm I think can be done with any variable
# as long as lat lon grid is present
ds_out = xe.util.grid_global(2, 2).isel(y=slice(0, 15))

# Can be any MAR input grid as long as lat lon is present (rename!)
# REGRID ERA
ds_in = ERA
regridder_ERA = xe.Regridder(ds_in, ds_out, 'bilinear')
regrid_ERA = regridder_ERA(ERA)


# MAR
regridder_BS = xe.Regridder(BS, ds_out, 'bilinear')
regrid_BS = regridder_BS(BS)
regrid_NOBS = regridder_BS(NOBS)

# CLOUDSAT

ds_in = cloudsat
regridder_cloudsat = xe.Regridder(ds_in, ds_out, 'bilinear')
regrid_cloudsat = regridder_cloudsat(cloudsat.transpose())


# ========== COMPUTE THE DIFFERENCES BETWEEN CLOUDSAT and ERA,MAR, MARBS ==

diff_test = regrid_BS.CC - regrid_cloudsat
diff_test_nobs = regrid_NOBS.CC - regrid_cloudsat
