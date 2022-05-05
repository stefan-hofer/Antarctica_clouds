
import numpy as np
from cmcrameri import cm
import cartopy.feature as feat
import datetime as dt
import xarray as xr
import xesmf as xe
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import pandas as pd
import glob
import matplotlib.path as mpath


directory = '/projects/NS9600K/shofer/blowing_snow/sat_data/CALIOP_blowing_snow/'
ds = xr.open_mfdataset(
    directory + 'CAL_LID_L2_BlowingSnow_Antarctica-Standard-V1*.nc')
ann_mean = ds.mean(dim='time')
apr_oct = ds["bs_freq"].sel(time=slice(2007 - 2015)).where((ds.time.dt.month >= 4) & (
    ds.time.dt.month <= 10)).mean(dim='time')

file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'
# ==========================================================================
# CREATE the ICE MASK
# =========================================================

MAR_grid = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'lat': (['y', 'x'], MAR_grid.LAT.values),
                      'lon': (['y', 'x'], MAR_grid.LON.values),
                      'ICE': (['y', 'x'], MAR_grid.ICE.values),
                      'AIS': (['y', 'x'], MAR_grid.AIS.values),
                      'SOL': (['y', 'x'], MAR_grid.SOL.values),
                      'GROUND': (['y', 'x'], MAR_grid.GROUND.values),
                      'AREA': (['y', 'x'], MAR_grid.AREA.values),
                      'ROCK': (['y', 'x'], MAR_grid.ROCK.values)},
                     coords={'x': (['x'], MAR_grid.X.values * 1000),
                             'y': (['y'], MAR_grid.Y.values * 1000)})

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

# ====================================================================================
# ==============================================================================
# Regridding operations
# ==============================================================================
# This creates the output grid, atm I think can be done with any variable
# as long as lat lon grid is present
ds_out = ds_grid
# Can be any MAR input grid as long as lat lon is present (rename!)
# REGRID MAR
ds_in = ds
regridder_MAR = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=True)

regridded_ds = regridder_MAR(ds)  # whole bs dataset regridded to MAR grid
regridded_BS = regridder_MAR(ann_mean)
regridded_BS_seasonal = regridder_MAR(apr_oct)
# PLOT THE MAP
plt.close('all')
proj = ccrs.SouthPolarStereo()
fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={
    'projection': proj}, figsize=(7, 10))
# fig = plt.figure(figsize=(8, 8))
# spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
# ax1 = fig.add_subplot(spec2[0, 0], projection=proj)


ax = [axs]
for i in range(1):
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

cont = ax[0].pcolormesh(ds_grid['x'], ds_grid['y'],
                        regridded_BS['bs_freq'].where(ais == 1), transform=proj, cmap=cm.bilbao, vmin=0, vmax=50)
# axs.add_feature(feat.COASTLINE)
xr.plot.contour(ds_grid.SOL, levels=1, colors='black',
                linewidths=0.4, transform=proj, ax=ax[i])
xr.plot.contour(ground, levels=1, colors='black', linewidths=0.4, ax=ax[i])

cb = fig.colorbar(cont, ax=ax[0], ticks=list(
    np.arange(0, 50, 10)), shrink=0.8)
# cb = fig.colorbar(cont, ax=ax[0], ticks=list(
#     np.arange(-30, 35, 10)), shrink=0.8, orientation='horizontal')
# cb.set_label(r'$\Delta$ Snow Content (g/kg)', fontsize=16)


fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map.png',
            format='PNG', dpi=300)
fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map.pdf',
            format='PDF')

Snow_Blow_Frequency_Grid
Units: count
Format: Float_32
Fill Value:
    Valid Range: N / A
    Definition: 360x180x9 grid
    Contains the monthly blowing snow layer occurrence in each 1x1 degree grid
    Blow_Snow_Frequency_Grid(0) = occurrence of blowing snow layers
    Blow_Snow_Frequency_Grid(1) = occurrence of blowing snow layers where the backscatter ratio of the top altitude bin within layer to first bin above ground < 0.5
    Blow_Snow_Frequency_Grid(2) = occurrence of blowing snow layers where the ratio of average backscatter strength within blowing snow layer to the blowing snow threshold > 1.5 and layer depth > 50 m
    Blow_Snow_Frequency_Grid(3) = occurrence of blowing snow layers where the 10 m wind speed > 6 m / s and layer depth > 50 m
    Blow_Snow_Frequency_Grid(4) = occurrence of blowing snow layers where the surface elevation(as determined from DEM) > 0.0 and 10 m wind speed > 7 m / s
    Blow_Snow_Frequency_Grid(5) = occurrence of blowing snow layers where the layer depth > 30m and < 350 m
    Blow_Snow_Frequency_Grid(6) = occurrence of blowing snow layers where the average layer 532 depolarization > 0.25 and layer average color ratio > 0.8
    Blow_Snow_Frequency_Grid(7) = occurrence of blowing snow layers where one of the following conditions is true(as stated in (1), (4), & (6)) the backscatter ratio of the top altitude bin within layer to first bin above ground < 0.5 OR the surface elevation(as determined from DEM) > 0.0 and 10 m wind speed > 7 m / s OR the average layer 532 depolarization > 0.25 and layer average color ratio > 0.8
    Blow_Snow_Frequency_Grid(8) = occurrence of the Blowing Snow Confidence Flag > 4


# lat = dset_geo["Latitude"][:,0]
# lon = dset_geo["Longitude"][:,0]
