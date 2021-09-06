import xarray as xr

import glob
import datetime as dt

import cartopy.feature as feat

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
# file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/new/3D_monthly/'


year_s = '2000-01-01'
year_e = '2019-12-31'
files = sorted(glob.glob(file_str + 'mon*.nc'))

i = 0
for file in files:
    ds = xr.open_dataset(file)
    mean_ds = ds.sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
    save_str = file_str + 'mean-' + \
        files[i].split('/')[-1][:-21] + 'MAR_ERA5_2000-2019.nc'
    mean_ds.to_netcdf(save_str)
    print('Working on: {}'.format(save_str))
    i += 1

file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'

year_s = '2000-01-01'
year_e = '2019-12-31'
files = sorted(glob.glob(file_str_nobs + 'mon*.nc'))

i = 0
for file in files:
    ds = xr.open_dataset(file)
    mean_ds = ds.sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
    save_str = file_str_nobs + 'mean-' + \
        files[i].split('/')[-1][:-21] + 'MAR_ERA5_2000-2019.nc'
    mean_ds.to_netcdf(save_str)
    print('Working on: {}'.format(save_str))
    i += 1
