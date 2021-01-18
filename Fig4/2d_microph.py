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

# # Wessel file folders
# file_str = '/uio/kant/geo-metos-u1/shofer/data/3D_monthly/'
# file_str_nobs = '/uio/kant/geo-metos-u1/shofer/data/3D_monthly_nDR/'
# file_str_zz = '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/'

year_s = '2000-01-01'
year_e = '2019-12-31'
# Open the no driftig snow file
ds_bs_CC = (xr.open_dataset(
    file_str + 'mon-CC-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_COD = (xr.open_dataset(
    file_str + 'mon-COD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_IWP = (xr.open_dataset(
    file_str + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_LWP = (xr.open_dataset(
    file_str + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

for ds in [ds_bs_CC, ds_bs_COD, ds_bs_IWP, ds_bs_LWP]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000
# ========================================
ds_nobs_CC = (xr.open_dataset(
    file_str_nobs + 'mon-CC-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_COD = (xr.open_dataset(
    file_str_nobs + 'mon-COD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_IWP = (xr.open_dataset(
    file_str_nobs + 'mon-IWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWP = (xr.open_dataset(
    file_str_nobs + 'mon-CWP-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
for ds in [ds_nobs_CC, ds_nobs_COD, ds_nobs_IWP, ds_nobs_LWP]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000
# ============================================
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
                     coords={'x': (['x'], ds_nobs_CC.X),
                             'y': (['y'], ds_nobs_CC.Y)})

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

x2D, y2D = np.meshgrid(MSK['x'], ds_grid['y'])
sh = ds_grid['SH']

dh = (ds_grid['x'].values[0] - ds_grid['x'].values[1]) / 2.

# =======================================================================


diff_CC = (ds_bs_CC - ds_nobs_CC).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(10, -10), y=slice(10, -10))
diff_COD = (ds_bs_COD - ds_nobs_COD).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(10, -10), y=slice(10, -10))
diff_IWP = (ds_bs_IWP - ds_nobs_IWP).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(10, -10), y=slice(10, -10))
diff_LWP = (ds_bs_LWP - ds_nobs_LWP).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(10, -10), y=slice(10, -10))

# Weighted by difference in Cloud cover bc values are just monthly means
diff_weighted_LWP = diff_LWP.CWP - (diff_CC.CC * diff_LWP.CWP)
diff_weighted_IWP = diff_IWP.IWP - (diff_CC.CC * diff_IWP.IWP)
diff_weighted_COD = diff_COD.COD - (diff_CC.CC * diff_COD.COD)

# Extract values for paper
# Grounded ice
diff_weighted_LWP.where(ground == 1).mean() * 1000
# Shelves
diff_weighted_LWP.where(shelf == 1).mean() * 1000
# Ocean
diff_weighted_LWP.where((ais == 0) & (shelf == 0)).mean() * 1000
# Grounded ice
diff_weighted_IWP.where(ground == 1).mean() * 1000
# Shelves
diff_weighted_IWP.where(shelf == 1).mean() * 1000
# Ocean
diff_weighted_IWP.where((ais == 0) & (shelf == 0)).mean() * 1000
# Grounded ice
diff_weighted_COD.where(ground == 1).mean()
# Shelves
diff_weighted_COD.where(shelf == 1).mean()
# Ocean
diff_weighted_COD.where((ais == 0) & (shelf == 0)).mean()
# Grounded ice
diff_CC.CC.where(ground == 1).mean() * 100
# Shelves
diff_CC.CC.where(shelf == 1).mean() * 100
# Ocean
diff_CC.CC.where((ais == 0) & (shelf == 0)).mean() * 100

diff_CC['LAT'] = ds_grid.LAT
diff_CC['LON'] = ds_grid.LON

diff_COD['LAT'] = ds_grid.LAT
diff_COD['LON'] = ds_grid.LON

diff_IWP['LAT'] = ds_grid.LAT
diff_IWP['LON'] = ds_grid.LON

diff_LWP['LAT'] = ds_grid.LAT
diff_LWP['LON'] = ds_grid.LON


# Plotting routines
plt.close('all')
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(7, 10))
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0], projection=proj)
ax2 = fig.add_subplot(spec2[0, 1], projection=proj)
# plt.setp(ax2.get_yticklabels(), visible=False)
ax3 = fig.add_subplot(spec2[1, 0], projection=proj)
ax4 = fig.add_subplot(spec2[1, 1], projection=proj)


ax = [ax1, ax2, ax3, ax4]
names = ['CC', 'COD', 'LWP', 'IWP']
for i in range(4):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax[i].add_feature(feat.LAND)
    # ax[i].add_feature(feat.OCEAN)

    ax[i].gridlines()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax[i].set_boundary(circle, transform=ax[i].transAxes)

# PLOT EVERYTHING

(diff_CC.CC * 100).plot.pcolormesh('x', 'y', transform=proj, ax=ax1,
                                   robust=True, cbar_kwargs={
                                       'label': r'$\Delta$ CC (%)', 'shrink': 1, 'orientation': 'horizontal',
                                       'ticks': [-20, -10, 0, 10, 20], 'pad': 0.05})
diff_weighted_COD.plot.pcolormesh('x', 'y', transform=proj, ax=ax2, robust=True, cbar_kwargs={
                                  'label': r'$\Delta$ COD', 'shrink': 1, 'orientation': 'horizontal',
                                  'pad': 0.05, 'fraction': 0.15})
(diff_weighted_LWP * 1000).plot.pcolormesh('x', 'y', transform=proj, ax=ax3,
                                           robust=True, cbar_kwargs={
                                               'label': r'$\Delta$ LWP (g/kg)', 'shrink': 1, 'orientation': 'horizontal',
                                               'ticks': [-2, -1, 0, 1, 2], 'pad': 0.05})
(diff_weighted_IWP * 1000).plot.pcolormesh('x', 'y', transform=proj, ax=ax4,
                                           robust=True, cbar_kwargs={
                                               'label': r'$\Delta$ IWP (g/kg)', 'shrink': 1, 'orientation': 'horizontal',
                                               'ticks': [-20, -10, 0, 10, 20], 'pad': 0.05})
#
# cont = ax[0].pcolormesh(diff_CC['x'], diff_CC['y'],
#                         (diff_CC.CC)*100,
#                         transform=proj, cmap='Reds')
# cont2 = ax[1].pcolormesh(diff_COD['x'], diff_COD['y'],
#                          diff_COD.COD,
#                          transform=proj, cmap='RdBu_r')
# cont3 = ax[2].pcolormesh(diff_LWP['x'], diff_LWP['y'],
#                          diff_LWP.CWP, transform=proj, cmap='RdBu_r')
# cont4 = ax[3].pcolormesh(diff_IWP['x'], diff_IWP['y'],
#                          diff_IWP.IWP, transform=proj, cmap='RdBu_r')
# cont2 = ax[1].pcolormesh(trend_CC['lon'], trend_CC['lat'],
#                          trend_CC.slope*30, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
letters = ['A', 'B', 'C', 'D']
for i in range(4):
    ax[i].add_feature(feat.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=16)
    ax[i].text(0.04, 1.02, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})


# fig.canvas.draw()

fig.tight_layout()
# fig.colorbar(cont2, ax=ax[1], ticks=list(
#     np.arange(-15, 15.5, 3)), shrink=0.8)
# cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                    orientation = 'horizontal', fraction = 0.13, pad = 0.01, shrink = 0.8)
# cbar.set_label(
#    'Average DJF cloud cover 2002-2015 (%)', fontsize=18)
fig.savefig('/uio/kant/geo-metos-u1/shofer/data/microphysics.png',
            format='PNG', dpi=300)
fig.savefig('/projects/NS9600K/shofer/blowing_snow/microphysics.png',
            format='PNG', dpi=300)
