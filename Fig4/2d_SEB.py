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

year_s = '2000-01-01'
year_e = '2019-12-31'
# Open the no driftig snow file
ds_bs_SWD = (xr.open_dataset(
    file_str + 'mon-SWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWU = (xr.open_dataset(
    file_str + 'mon-SWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_LWD = (xr.open_dataset(
    file_str + 'mon-LWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_LWU = (xr.open_dataset(
    file_str + 'mon-LWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_AL = (xr.open_dataset(
    file_str + 'mon-AL-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWN = (ds_bs_SWD.SWD - ds_bs_SWU.SWU)
ds_bs_SWN.name = 'SWN'
ds_bs_LWN = (ds_bs_LWD.LWD - ds_bs_LWU.LWU)
ds_bs_LWN.name = 'LWN'

# ========================================
ds_nobs_SWD = (xr.open_dataset(
    file_str_nobs + 'mon-SWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWU = (xr.open_dataset(
    file_str_nobs + 'mon-SWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWD = (xr.open_dataset(
    file_str_nobs + 'mon-LWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWU = (xr.open_dataset(
    file_str_nobs + 'mon-LWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_AL = (xr.open_dataset(
    file_str_nobs + 'mon-AL-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWN = (ds_nobs_SWD.SWD - ds_nobs_SWU.SWU)
ds_nobs_SWN.name = 'SWN'
ds_nobs_LWN = (ds_nobs_LWD.LWD - ds_nobs_LWU.LWU)
ds_nobs_LWN.name = 'LWN'
# ============================================
MAR_grid = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], ds_nobs_SWD.X),
                             'y': (['y'], ds_nobs_SWD.Y)})

# ==========================================================================
# CREATE the ICE MASK
MSK = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf', decode_times=False)
MSK = MSK.rename_dims({'x': 'X', 'y': 'Y'})  # change dim from ferret
MSK = MSK.swap_dims({'X': 'x', 'Y': 'y'})  # change dim from ferret

ais = MSK['AIS'].where(MSK['AIS'] > 0)  # Only AIS=1, other islands  =0
# Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
ice = MSK['ICE'].where(MSK['ICE'] > 30)
# Combine ais + ice/100 * factor area for taking into account the projection
ice_msk = (ais * ice * MSK['AREA'] / 100)

grd = MSK['GROUND'].where(MSK['GROUND'] > 30)
grd_msk = (ais * grd * MSK['AREA'] / 100)

lsm = (MSK['AIS'] < 1)
ground = (MSK['GROUND'] * MSK['AIS'] > 30)

shf = (MSK['ICE'] / MSK['ICE']).where((MSK['ICE'] > 30) &
                                      (MSK['GROUND'] < 50) & (MSK['ROCK'] < 30) & (ais > 0))
shelf = (shf > 0)

x2D, y2D = np.meshgrid(MSK['x'], MSK['y'])
sh = MSK['SH']

dh = (MSK['x'].values[0] - MSK['x'].values[1]) / 2.

# ==========================================================


diff_SWD = (ds_bs_SWD - ds_nobs_SWD).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(5, -5), y=slice(5, -5))
diff_LWD = (ds_bs_LWD - ds_nobs_LWD).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(5, -5), y=slice(5, -5))
diff_SWN = (ds_bs_SWN - ds_nobs_SWN).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(5, -5), y=slice(5, -5))
diff_LWN = (ds_bs_LWN - ds_nobs_LWN).rename(
    {'X': 'x', 'Y': 'y'}).isel(x=slice(5, -5), y=slice(5, -5))

diff_SWD['LAT'] = ds_grid.LAT
diff_SWD['LON'] = ds_grid.LON

diff_LWD['LAT'] = ds_grid.LAT
diff_LWD['LON'] = ds_grid.LON

diff_SWN['LAT'] = ds_grid.LAT
diff_SWN['LON'] = ds_grid.LON

diff_LWN['LAT'] = ds_grid.LAT
diff_LWN['LON'] = ds_grid.LON

abs_diff_external = (
    diff_SWD.SWD + diff_LWD.LWD)
abs_diff = (diff_SWN + diff_LWN)


# Plotting routines
plt.close('all')
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(7, 10))
spec2 = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0], projection=proj)
ax2 = fig.add_subplot(spec2[0, 1], projection=proj)
# plt.setp(ax2.get_yticklabels(), visible=False)
ax3 = fig.add_subplot(spec2[1:, :], projection=proj)

# PLOT EVERYTHING


ax = [ax1, ax2, ax3]
names = ['SWD', 'LWD', 'Net radiation']
for i in range(3):
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

cmap = 'YlGnBu_r'
cont = ax[0].pcolormesh(diff_SWD['x'] * 1000, diff_SWD['y'] * 1000,
                        diff_SWD.SWD,
                        transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont2 = ax[1].pcolormesh(diff_LWD['x'] * 1000, diff_LWD['y'] * 1000,
                         diff_LWD.LWD,
                         transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont3 = ax[2].pcolormesh(abs_diff['x'] * 1000, abs_diff['y'] * 1000,
                         abs_diff, transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
# cont2 = ax[1].pcolormesh(trend_CC['lon'], trend_CC['lat'],
#                          trend_CC.slope*30, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
letters = ['A', 'B', 'C']
for i in range(3):
    ax[i].add_feature(feat.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=16)
    ax[i].text(0.04, 1.02, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})
# fig.canvas.draw()

cb = fig.colorbar(cont, ax=ax[2], ticks=list(
    np.arange(-4, 4.5, 1)), shrink=0.8)
cb.set_label(r'$\Delta$ Radiative Flux $(Wm^{-2})$', fontsize=16)
cb.ax.tick_params(labelsize=11)
fig.tight_layout()
# fig.colorbar(cont2, ax=ax[1], ticks=list(
#     np.arange(-15, 15.5, 3)), shrink=0.8)
# cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                    orientation = 'horizontal', fraction = 0.13, pad = 0.01, shrink = 0.8)
# cbar.set_label(
#    'Average DJF cloud cover 2002-2015 (%)', fontsize=18)

fig.savefig('/projects/NS9600K/shofer/blowing_snow/SEB.png',
            format='PNG', dpi=300)
