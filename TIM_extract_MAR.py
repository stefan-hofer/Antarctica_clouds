import xarray as xr


# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
# file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly/'
file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'
file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'


# Cloud cover years: 2006/07 to 2011/02
year_s = '2006-07-01'
year_e = '2011-02-18'
# Open the no driftig snow file
ds_bs_CC = (xr.open_dataset(
    file_str + 'mon-CC-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_CC = (xr.open_dataset(
    file_str_nobs + 'mon-CC-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))

ds_bs_CC.to_netcdf(file_str + 'CC_BS_200607-201102.nc')
ds_nobs_CC.to_netcdf(file_str + 'CC_noBS_200607-201102.nc')
# For IWP/LWP
# 2006: 06 - 12
# 2007: 01 - 12
# 2008: 01 - 12
# 2009: 01 - 12
# 2010: 01 - 12
# 2011: 02 - 04
# 2012: 05 - 12
# 2013: 01 - 12
# 2014: 01 - 12
# 2015: 01 - 12
# 2016: 01 - 12
# 2017: 01 - 12

year_s = '2006-07-01'
year_e = '2010-12-18'
# Open the no driftig snow file
ds_bs_CWP_one = (xr.open_dataset(
    file_str + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_CWP_one = (xr.open_dataset(
    file_str_nobs + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_bs_IWP_one = (xr.open_dataset(
    file_str + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_IWP_one = (xr.open_dataset(
    file_str_nobs + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))


ds_bs_CWP_one.to_netcdf(file_str + 'LWP_BS_200607-201012.nc')
ds_nobs_CWP_one.to_netcdf(file_str + 'LWP_noBS_200607-201012.nc')

ds_bs_IWP_one.to_netcdf(file_str + 'IWP_BS_200607-201012.nc')
ds_nobs_IWP_one.to_netcdf(file_str + 'IWP_noBS_200607-201012.nc')

year_s = '2011-02-01'
year_e = '2011-04-18'
# Open the no driftig snow file
ds_bs_CWP_two = (xr.open_dataset(
    file_str + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_CWP_two = (xr.open_dataset(
    file_str_nobs + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_bs_IWP_two = (xr.open_dataset(
    file_str + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_IWP_two = (xr.open_dataset(
    file_str_nobs + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))

ds_bs_CWP_two.to_netcdf(file_str + 'LWP_BS_201102-201104.nc')
ds_nobs_CWP_two.to_netcdf(file_str + 'LWP_noBS_201102-201104.nc')

ds_bs_IWP_two.to_netcdf(file_str + 'IWP_BS_201102-201104.nc')
ds_nobs_IWP_two.to_netcdf(file_str + 'IWP_noBS_201102-201104.nc')

year_s = '2012-05-01'
year_e = '2018-12-18'
# Open the no driftig snow file
ds_bs_CWP_three = (xr.open_dataset(
    file_str + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_CWP_three = (xr.open_dataset(
    file_str_nobs + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_bs_IWP_three = (xr.open_dataset(
    file_str + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))
ds_nobs_IWP_three = (xr.open_dataset(
    file_str_nobs + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e))

ds_bs_CWP_three.to_netcdf(file_str + 'LWP_BS_201205-201812.nc')
ds_nobs_CWP_three.to_netcdf(file_str + 'LWP_noBS_201205-201812.nc')

ds_bs_IWP_three.to_netcdf(file_str + 'IWP_BS_201205-201812.nc')
ds_nobs_IWP_three.to_netcdf(file_str + 'IWP_noBS_201205-201812.nc')
