import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.gridspec as gridspec
import cartopy.feature as feat

from cmcrameri import cm


file_dir = '/projects/NS9600K/shofer/blowing_snow/MAR/new/3D_monthly/'
file_dir_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'
station_dir = '/projects/NS9600K/shofer/blowing_snow/observations/new/LAT_LON_stations.csv'
station_dir_xy = '/projects/NS9600K/shofer/blowing_snow/observations/new/x_y_stations.csv'

file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'

year_s = '2000-01-01'
year_e = '2019-12-31'

ds = (xr.open_dataset(file_dir + 'year-QS-MAR_ERA5-1980-2019.nc')
      ).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

ds_nobs = (xr.open_dataset(file_dir_nobs + 'year-QS-MAR_ERA5-1979-2020.nc')
           ).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

erosion = (xr.open_dataset(file_dir + 'year-ER-MAR_ERA5-1979-2019.nc')
           ).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
erosion['X'] = erosion['X'] * 1000
erosion['Y'] = erosion['Y'] * 1000


stations = pd.read_csv(station_dir, delimiter=',',
                       names=['Name', 'Lat', 'Lon'])
station_xy = pd.read_csv(station_dir_xy, delimiter=';')

for ds in [ds_nobs, ds]:
    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000

diff = ((ds.QS * 1000) - (ds_nobs.QS * 1000)).rename({'X': 'x', 'Y': 'y'})
erosion_new = erosion.isel(SECTOR1_1=0).rename({'X': 'x', 'Y': 'y'})
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
                     coords={'x': (['x'], ds_nobs.X),
                             'y': (['y'], ds_nobs.Y)})

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

diff_new = diff.where((ground > 0) | (shelf > 0)).isel(
    x=slice(10, -10), y=slice(10, -10))

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

diff_new.plot.pcolormesh('x', 'y', transform=proj, ax=ax[0], cmap='RdBu_r',
                         robust=True, cbar_kwargs={
    'label': r'$\Delta$ Snow Particle Ratio (g/kg)', 'shrink': 1, 'orientation': 'horizontal',
    'ticks': [-30, -20, -10, 0, 10, 20, 30], 'pad': 0.05, 'extend': 'both'}, vmin=-35, vmax=35)

erosion_new.ER.plot.pcolormesh('x', 'y', transform=proj, ax=ax[0], cmap=cm.bilbao,
                               robust=True, cbar_kwargs={
    'label': 'Snow Erosion (mmWE)', 'shrink': 1, 'orientation': 'horizontal', 'pad': 0.05})
ax[0].set_title('')

xr.plot.contour(ds_grid.SOL, levels=1, colors='black',
                linewidths=0.4, transform=proj, ax=ax[i])
xr.plot.contour(ground, levels=1, colors='black', linewidths=0.4, ax=ax[i])

plt.plot(station_xy.x * 1000, station_xy.y * 1000,
         'o', color='#014d4e', transform=proj)  # d0fefe
# cb = fig.colorbar(cont, ax=ax[0], ticks=list(
#     np.arange(-30, 35, 10)), shrink=0.8, orientation='horizontal')
# cb.set_label(r'$\Delta$ Snow Content (g/kg)', fontsize=16)

fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map_erosion.png',
            format='PNG', dpi=300)
fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map_erosion.pdf',
            format='PDF')
