"""
Created on Tue Mar 24 08:57:52 2020

@author: ckittel

Time series of SMB and comps + mean values
"""

import xarray as xr
import xesmf as xe
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import pandas as pd
import glob
import datetime as dt
import matplotlib.path as mpath

import cartopy.feature as feat

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
# file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly/'
file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'
file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'


ME_bs = (xr.open_dataset(
    file_str + 'mon-ME-MAR_ERA5-1979-2019.nc')).isel(SECTOR1_1=0)

ME_nobs = (xr.open_dataset(file_str_nobs +
                           'mon-ME-MAR_ERA5-1979-2019.nc')).isel(SECTOR1_1=0)
# =============================================================================
# FUNCTIONS
# =============================================================================


def preprocess(ds):
    '''
    Avoid reading/opening the whole ds. => Selection of interesting variable
    Also try to remove SECTOR dimension
    (Sector=1 corresponds to the ice sheet in MAR (2= tundra or rocks))
    '''
    ds_new = ds[['LWD', 'SWD', 'SHF', 'LHF', 'AL2']]
    try:
        ds_new = ds_new.sel(SECTOR=1)
    except:
        ds_new = ds_new

    ds_new['SWN'] = (1-ds_new['AL2'])*ds_new['SWD']

    return ds_new


def SMBcomponents_to_gt(SMB_array, variable, mask, data_start=1980, data_end=2100):
    '''This function returns a daily time series of absolute SMB
    component values, expressed as gigatons.
    35 000 m = MAR resolution
    1e12 factor to convert kg/m² in Gt
    '''

    data = SMB_array[variable] * mask.values
    # Make sure only wanted time frame is used
    data = data.loc[str(data_start) +
                    '-01-01':str(data_end) + '-12-31']
    # Convert to gigatons and sum up spatially over the AIS
    sum_spatial = data.sum(dim=['X', 'Y']
                           ) * ((35000 * 35000) / (1e12))

    return sum_spatial


def annual_sum(data):
    '''This function returns the annual sum
    '''
    annual_sum = data.groupby('TIME.year').sum(dim='TIME')
    return annual_sum


def spatial_mean(SMB_array, variable, mask, data_start=1980, data_end=2100):
    '''This function returns the spatial mean
    '''
    data = SMB_array[variable] * mask.values
    data = data.loc[str(data_start) +
                    '-01-01':str(data_end) + '-12-31']
    mean_spatial = data.mean(dim=['X', 'Y'])

    return mean_spatial


def annual_mean(data):
    annual_mean = data.groupby('TIME.year').mean(dim='TIME')
    return annual_mean


# =============================================================================
# # CREATE the ICE MASK
# =============================================================================
test = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf')

# Only AIS=1, other islands  =0
ais = test['AIS'].where(test['AIS'] > 0)
# Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
ice_msk = (ais/ais).where(test['ICE'] > 30)
# ice_msk = (ais*ice*test['AREA']/ 100)    #Combine ais + ice/100 * factor area for taking into account the projection

grd_msk = (ais/ais).where(test['GROUND'] > 30)

shf_msk = (ais/ais).where((test['ICE'] > 30) &
                          (test['GROUND'] < 50) & (test['ROCK'] < 30))
