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

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/new/3D_monthly/'
file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'
file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'
station_dir = '/projects/NS9600K/shofer/blowing_snow/observations/new/LAT_LON_stations.csv'
station_dir_xy = '/projects/NS9600K/shofer/blowing_snow/observations/new/x_y_stations.csv'
# ================================================================================================

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
# ds_bs_AL = (xr.open_dataset(
#    file_str + 'mon-AL-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWN = (ds_bs_SWD.SWD - ds_bs_SWU.SWU)
ds_bs_SWN.name = 'SWN'
ds_bs_LWN = (ds_bs_LWD.LWD - ds_bs_LWU.LWU)
ds_bs_LWN.name = 'LWN'

for ds in [ds_bs_SWD, ds_bs_SWU, ds_bs_LWD, ds_bs_LWU, ds_bs_SWN, ds_bs_LWN]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000

# ========================================
ds_nobs_SWD = (xr.open_dataset(
    file_str_nobs + 'mon-SWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWU = (xr.open_dataset(
    file_str_nobs + 'mon-SWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWD = (xr.open_dataset(
    file_str_nobs + 'mon-LWD-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWU = (xr.open_dataset(
    file_str_nobs + 'mon-LWU-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
# ds_nobs_AL = (xr.open_dataset(
#    file_str_nobs + 'mon-AL-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWN = (ds_nobs_SWD.SWD - ds_nobs_SWU.SWU)
ds_nobs_SWN.name = 'SWN'
ds_nobs_LWN = (ds_nobs_LWD.LWD - ds_nobs_LWU.LWU)
ds_nobs_LWN.name = 'LWN'

for ds in [ds_nobs_SWD, ds_nobs_SWU, ds_nobs_LWD, ds_nobs_LWU, ds_nobs_SWN, ds_nobs_LWN]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000
# ============================================
ds = (xr.open_dataset(file_str + 'year-QS-MAR_ERA5-1980-2019.nc')
      ).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

ds_nobs = (xr.open_dataset(file_str_nobs + 'year-QS-MAR_ERA5-1979-2020.nc')
           ).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
stations = pd.read_csv(station_dir, delimiter=',',
                       names=['Name', 'Lat', 'Lon'])
station_xy = pd.read_csv(station_dir_xy, delimiter=';')

for ds in [ds_nobs, ds]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000

diff = ((ds.QS * 1000) - (ds_nobs.QS * 1000)).rename({'X': 'x', 'Y': 'y'})
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
                      'SOL': (['y', 'x'], MAR_grid.SOL.values),
                      'GROUND': (['y', 'x'], MAR_grid.GROUND.values),
                      'AREA': (['y', 'x'], MAR_grid.AREA.values),
                      'ROCK': (['y', 'x'], MAR_grid.ROCK.values)},
                     coords={'x': (['x'], ds_nobs_LWD.X),
                             'y': (['y'], ds_nobs_LWD.Y)})

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


# ==========================================================


diff_SWD = (ds_bs_SWD - ds_nobs_SWD).rename(
    {'X': 'x', 'Y': 'y'}).where((ground > 0) | (shelf > 0)).isel(x=slice(10, -10), y=slice(10, -10))
diff_LWD = (ds_bs_LWD - ds_nobs_LWD).rename(
    {'X': 'x', 'Y': 'y'}).where((ground > 0) | (shelf > 0)).isel(x=slice(10, -10), y=slice(10, -10))
diff_SWN = (ds_bs_SWN - ds_nobs_SWN).rename(
    {'X': 'x', 'Y': 'y'}).where((ground > 0) | (shelf > 0)).isel(x=slice(10, -10), y=slice(10, -10))
diff_LWN = (ds_bs_LWN - ds_nobs_LWN).rename(
    {'X': 'x', 'Y': 'y'}).where((ground > 0) | (shelf > 0)).isel(x=slice(10, -10), y=slice(10, -10))


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


# Grounded ice
LWP_one = diff_SWD.SWD.where(ground == 1).mean()
# Shelves
LWP_two = diff_SWD.SWD.where(shelf == 1).mean()
# Ocean
LWP_three = diff_SWD.SWD.where((ais == 0) & (shelf == 0)).mean()

print('{}: Grounded={:.2f}, Shelves={:.2f}, Ocean={:.2f}'.format(
    'SWD', LWP_one.values, LWP_two.values, LWP_three.values))
# Grounded ice
IWP_one = diff_LWD.LWD.where(ground == 1).mean()
# Shelves
IWP_two = diff_LWD.LWD.where(shelf == 1).mean()
# Ocean
IWP_three = diff_LWD.LWD.where((ais == 0) & (shelf == 0)).mean()

print('{}: Grounded={:.2f}, Shelves={:.2f}, Ocean={:.2f}'.format(
    'LWD', IWP_one.values, IWP_two.values, IWP_three.values))
# Grounded ice
COD_one = abs_diff.where(ground == 1).mean()
# Shelves
COD_two = abs_diff.where(shelf == 1).mean()
# Ocean
COD_three = abs_diff.where((ais == 0) & (shelf == 0)).mean()

print('{}: Grounded={:.2f}, Shelves={:.2f}, Ocean={:.2f}'.format(
    'Net diff', COD_one.values, COD_two.values, COD_three.values))

# diff in QS
diff_new = diff.where((ground > 0) | (shelf > 0)).isel(
    x=slice(10, -10), y=slice(10, -10))


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
cont = ax[0].pcolormesh(diff_SWD['x'], diff_SWD['y'],
                        diff_SWD.SWD,
                        transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont2 = ax[1].pcolormesh(diff_LWD['x'], diff_LWD['y'],
                         diff_LWD.LWD,
                         transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont3 = ax[2].pcolormesh(abs_diff['x'], abs_diff['y'],
                         abs_diff, transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
# cont2 = ax[1].pcolormesh(trend_CC['lon'], trend_CC['lat'],
#                          trend_CC.slope*30, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
letters = ['A', 'B', 'C']
for i in range(3):
    xr.plot.contour(ds_grid.SOL, levels=1, colors='black',
                    linewidths=0.4, transform=proj, ax=ax[i])
    xr.plot.contour(ground, levels=1, colors='black', linewidths=0.4, ax=ax[i])
    # ax[i].add_feature(feat.COASTLINE.with_scale(
    #     '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=16)
    ax[i].text(0.04, 1.02, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})
# fig.canvas.draw()

cb = fig.colorbar(cont3, ax=ax[2], ticks=list(
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
names = ['SWD', 'LWD', 'Net Radiation', 'QS']
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

cmap = 'YlGnBu_r'
cont = ax[0].pcolormesh(diff_SWD['x'], diff_SWD['y'],
                        diff_SWD.SWD,
                        transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont2 = ax[1].pcolormesh(diff_LWD['x'], diff_LWD['y'],
                         diff_LWD.LWD,
                         transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont3 = ax[2].pcolormesh(abs_diff['x'], abs_diff['y'],
                         abs_diff, transform=proj, vmin=-4, vmax=4, cmap='RdBu_r')
cont4 = ax[3].pcolormesh(diff_new['x'], diff_new['y'],
                         diff_new, transform=proj, vmin=-25, vmax=25, cmap='RdBu_r')

ax[3].plot(station_xy.x * 1000, station_xy.y * 1000,
           'o', color='#014d4e', transform=proj)

#          robust = True, cbar_kwargs = {
# 'label': r'$\Delta$ Snow Particle Ratio (g/kg)', 'shrink': 1, 'orientation': 'horizontal',
# 'ticks': [-30, -20, -10, 0, 10, 20, 30], 'pad': 0.05, 'extend': 'both'}, vmin = -35, vmax = 35)


letters = ['A', 'B', 'C', 'D']
for i in range(4):
    xr.plot.contour(ds_grid.SOL, levels=1, colors='black',
                    linewidths=0.4, transform=proj, ax=ax[i])
    xr.plot.contour(ground, levels=1, colors='black',
                    linewidths=0.4, ax=ax[i])
    # ax[i].add_feature(feat.COASTLINE.with_scale(
    #     '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=16)
    ax[i].text(0.04, 1.02, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})
# fig.canvas.draw()
for i in range(0, 3):
    cb = fig.colorbar(cont3, ax=ax[i], ticks=list(
        np.arange(-4, 4.5, 1)), shrink=0.8, orientation='horizontal')
    cb.set_label(r'$\Delta$ Radiative Flux $(Wm^{-2})$', fontsize=13)
    cb.ax.tick_params(labelsize=11)

cb = fig.colorbar(cont4, ax=ax[3], ticks=[-20, -10,
                                          0, 10, 20], shrink=0.8, orientation='horizontal')
cb.set_label(r'$\Delta$ Snow Particle Ratio (g/kg)', fontsize=13)
cb.ax.tick_params(labelsize=11)
fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/SEB_with_stations.png',
            format='PNG', dpi=300)
