import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.gridspec as gridspec
import xesmf as xe

import cartopy.feature as feat

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
                      'SOL': (['y', 'x'], MAR_grid.SOL.values),
                      'ROCK': (['y', 'x'], MAR_grid.ROCK.values)},
                     coords={'x': (['x'], ds_nobs_CC.x * 1000),
                             'y': (['y'], ds_nobs_CC.y * 1000)})

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

msk = ds_grid['SOL'].where(ds_grid['SOL'] == 4)
# =========== COMPUTE THE MEAN OVER THE SAME TIME PERIOD ==================
BS = ds_bs_CC.mean(dim='TIME')
# Add LAT LON to MAR data
BS['lat'] = ds_grid.LAT
BS['lon'] = ds_grid.LON
BS['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)
BS['shelf'] = shelf
BS['ground'] = ground
BS['msk'] = msk
BS['SOL'] = ds_grid.SOL


NOBS = ds_nobs_CC.mean(dim='TIME')
NOBS['lat'] = ds_grid.LAT
NOBS['lon'] = ds_grid.LON
NOBS['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)
NOBS['shelf'] = shelf
NOBS['ground'] = ground
NOBS['msk'] = msk
NOBS['SOL'] = ds_grid.SOL


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

diff_test = (regrid_BS.CC - regrid_cloudsat).where((regrid_BS.shelf > 0)
                                                   | (regrid_BS.ground > 0))
diff_test_nobs = (regrid_NOBS.CC - regrid_cloudsat).where((regrid_BS.shelf > 0)
                                                          | (regrid_BS.ground > 0))
diff_test_ERA = (regrid_ERA - regrid_cloudsat).where((regrid_BS.shelf > 0)
                                                     | (regrid_BS.ground > 0))

# ============ PLOTTING ROUTINE ========================
# Plotting routines
plt.close('all')
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(14, 7))
spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0], projection=proj)
ax2 = fig.add_subplot(spec2[0, 1], projection=proj)
# plt.setp(ax2.get_yticklabels(), visible=False)
ax3 = fig.add_subplot(spec2[0, 2], projection=proj)

# PLOT EVERYTHING


ax = [ax1, ax2, ax3]
names = ['No Blowing Snow', 'Blowing Snow', 'ERA5']
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
cont = ax[0].pcolormesh(diff_test_nobs['lon'], diff_test_nobs['lat'],
                        diff_test_nobs,
                        transform=ccrs.PlateCarree(), vmin=-1, vmax=1, cmap='RdBu_r')
cont2 = ax[1].pcolormesh(diff_test['lon'], diff_test['lat'],
                         diff_test,
                         transform=ccrs.PlateCarree(), vmin=-1, vmax=1, cmap='RdBu_r')
cont3 = ax[2].pcolormesh(diff_test_ERA['lon'], diff_test_ERA['lat'],
                         diff_test_ERA, transform=ccrs.PlateCarree(), vmin=-1, vmax=1, cmap='RdBu_r')
# cont2 = ax[1].pcolormesh(trend_CC['lon'], trend_CC['lat'],
#                          trend_CC.slope*30, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')

#    xr.plot.contour(ground, levels=1, colors='black', linewidths=0.2)
#    xr.plot.contour('x','y',ds_grid.SOL, levels=1, colors='black', linewidths=0.2, ax=ax[1], transform=proj)

letters = ['A', 'B', 'C']
for i in range(3):
    xr.plot.contour(ds_grid.SOL, levels=1, colors='black',
                    linewidths=0.4, transform=proj, ax=ax[i])
    xr.plot.contour(ground, levels=1, colors='black', linewidths=0.4, ax=ax[i])
    # ax[i].add_feature(feat.COASTLINE.with_scale(
    #    '50m'), zorder=1, edgecolor='black')

    ax[i].set_title(names[i], fontsize=16)
    ax[i].text(0.04, 1.02, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})
# fig.canvas.draw()
fig.tight_layout()

cb = fig.colorbar(cont3, ax=ax, ticks=list(
    np.arange(-1, 1.5, 0.2)), shrink=0.85, orientation='horizontal')
cb.set_label('Cloud Cover Difference (%)', fontsize=16)
cb.ax.tick_params(labelsize=11)

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.24)
# fig.colorbar(cont2, ax=ax[1], ticks=list(
#     np.arange(-15, 15.5, 3)), shrink=0.8)
# cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                    orientation = 'horizontal', fraction = 0.13, pad = 0.01, shrink = 0.8)
# cbar.set_label(
#    'Average DJF cloud cover 2002-2015 (%)', fontsize=18)

fig.savefig('/projects/NS9600K/shofer/blowing_snow/Calipso_difference.png',
            format='PNG', dpi=300)
