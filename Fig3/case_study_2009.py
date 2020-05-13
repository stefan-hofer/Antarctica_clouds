import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units

from functions import thetae_mar

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

# Open the no driftig snow file
ds_nobs = xr.open_dataset(
    file_str + 'MAR35_nDS_Oct2009.nc')
# Calculate overall CC from CC3D
ds_nobs['CC'] = ds_nobs.CC3D.max(dim='ATMLAY')

# Open the drifting snow file
ds_bs = xr.open_dataset(
    file_str + 'MAR35_DS_Oct2009.nc')
# Calculate overall CC from CC3D
ds_bs['CC'] = ds_bs.CC3D.max(dim='ATMLAY')
# Calculate ThetaE for ds case
thetae_ds = thetae_mar(ds_bs)
thetae_da = xr.DataArray(thetae_ds, coords=[ds_bs.TIME, ds_bs.Y, ds_bs.X],
                         dims = ['TIME', 'Y', 'X'])
# Add thetae to main dataset
ds_bs['thetae'] = thetae_da

# Calculate ThetaE for nods case
thetae_nods = thetae_mar(ds_nobs)
thetae_da_nods = xr.DataArray(thetae_nods, coords=[ds_bs.TIME, ds_bs.Y, ds_bs.X],
                         dims = ['TIME', 'Y', 'X'])
# Add thetae to main dataset
ds_nobs['thetae'] = thetae_da_nods

# Read in the grid
MAR_grid = xr.open_dataset(
    file_str + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})

ds_wind = xr.Dataset(ds_bs[['UU', 'VV', 'UV']].sel(TIME='2009-10-14').isel(TIME=0),
                coords={'lat': (('Y', 'X'), ds_grid.LAT), 'lon': (('Y', 'X'), ds_grid.LON)})
ds_mpy = ds_wind.metpy.parse_cf()

# Difference for 14th of October
lat_min = -90
lat_max = -60
lon_min = -180  # 70
lon_max = 180


diff = ds_bs.sel(TIME='2009-10-14') - ds_nobs.sel(TIME='2009-10-14')

ds = xr.Dataset(diff[['CC', 'IWP', 'CWP', 'QQ', 'LQI', 'LQS', 'RH', 'TT', 'thetae']].isel(TIME=0),
                coords={'lat': (('Y', 'X'), ds_grid.LAT), 'lon': (('Y', 'X'), ds_grid.LON)})
ds_wind = xr.Dataset(ds_bs[['UU', 'VV', 'UV']].sel(TIME='2009-10-14').isel(TIME=0),
                coords={'lat': (('Y', 'X'), ds_grid.LAT), 'lon': (('Y', 'X'), ds_grid.LON)})

# Plotting
names = ['Wind', 'Spec.Humidity', 'Low level Qs',
         'Cloud Cover', 'Ice water path', 'Liquid water path']
# Compare trends between 2002 and 2015
proj = ccrs.SouthPolarStereo()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(
    14, 8), subplot_kw={'projection': proj})
ax = axs.ravel().tolist()

for i in range(6):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    ax[i].add_feature(cartopy.feature.LAND)
    ax[i].add_feature(cartopy.feature.OCEAN)

    ax[i].gridlines()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax[i].set_boundary(circle, transform=ax[i].transAxes)


# cmap = 'YlGnBu_r'
ds_wind.UV.isel(ATMLAY=19).plot.pcolormesh('lon', 'lat', ax=ax[0], transform=ccrs.PlateCarree(),
                                               robust=True, cbar_kwargs={'shrink': 0.7})
# regrid shape controls the density of the wind barbs
# ax[0].barbs(ds_mpy.X.values, ds_mpy.Y.values, ds_wind.UU.isel(ATMLAY=19).values,
#             ds_wind.isel(ATMLAY=19).VV.values, transform=ccrs.SouthPolarStereo(),
#             regrid_shape=30, length=5, linewidth=0.5, alpha=0.8)
ax[0].quiver(ds_mpy.X.values, ds_mpy.Y.values, ds_wind.UU.isel(ATMLAY=19).values,
            ds_wind.isel(ATMLAY=19).VV.values, transform=ccrs.SouthPolarStereo(),
            regrid_shape=30, scale=260)
cont_1 = (ds.QQ.sel(ATMLAY=1, method='nearest')
          .plot.pcolormesh('lon', 'lat', ax=ax[1], transform=ccrs.PlateCarree(),
          robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Specific Humidity (g/kg)'}))
cont_2 = (ds.LQS.isel(ATMLAY=slice(13, 19)).sum(dim='ATMLAY')
          .plot.pcolormesh('lon', 'lat', ax=ax[2], transform=ccrs.PlateCarree(),
          robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Snow mixing ratio (g/kg)'}))

cont_3 = ds.CC.plot.pcolormesh('lon', 'lat', ax=ax[3], transform=ccrs.PlateCarree(
), robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Cloud Cover (%)'})
cont_4 = ds.IWP.plot.pcolormesh('lon', 'lat', ax=ax[4], transform=ccrs.PlateCarree(
), robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Ice Water Path ($kg/m^{2}$)'})

cont_5 = ds.CWP.plot.pcolormesh('lon', 'lat', ax=ax[5], transform=ccrs.PlateCarree(
), robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Liq. Water Path ($kg/m^{2}$)'})


for i in range(6):
    ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=18)
# fig.canvas.draw()
fig.tight_layout()

fig.savefig('/home/sh16450/Documents/repos/Antarctica_clouds/Fig3/Fig3.png', format='PNG', dpi=300)

# ===========================================================
# ========= SECOND FIGURE ===================================
# ===========================================================
names = ['ThetaE', 'RH', 'TT']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(
    15, 7), subplot_kw={'projection': proj})
ax = axs.ravel().tolist()

for i in range(3):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    ax[i].add_feature(cartopy.feature.LAND)
    ax[i].add_feature(cartopy.feature.OCEAN)

    ax[i].gridlines()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax[i].set_boundary(circle, transform=ax[i].transAxes)


# cmap = 'YlGnBu_r'
# ds_wind.UV.isel(ATMLAY=19).plot.pcolormesh('lon', 'lat', ax=ax[0], transform=ccrs.PlateCarree(),
#                                                robust=True, cbar_kwargs={'shrink': 0.7})
# ax[0].quiver(ds_mpy.X.values, ds_mpy.Y.values, ds_wind.UU.isel(ATMLAY=19).values,
#             ds_wind.isel(ATMLAY=19).VV.values, transform=ccrs.SouthPolarStereo(),
#             regrid_shape=30, scale=260)
cont_1 = (ds.thetae
          .plot.pcolormesh('lon', 'lat', ax=ax[0], transform=ccrs.PlateCarree(),
          robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ ThetaE ($\circ$C)'}))
cont_2 = (ds.RH.sel(ATMLAY=1, method='nearest')
          .plot.pcolormesh('lon', 'lat', ax=ax[1], transform=ccrs.PlateCarree(),
          robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Rel. Humidity (%)'}))
cont_3 = (ds.TT.sel(ATMLAY=1, method='nearest')
          .plot.pcolormesh('lon', 'lat', ax=ax[2], transform=ccrs.PlateCarree(),
          robust=True, cbar_kwargs={'shrink': 0.7, 'label': r'$\Delta$ Temperature ($\circ$C)'}))


for i in range(3):
    ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
    ax[i].set_title(names[i], fontsize=18)
# fig.canvas.draw()
fig.tight_layout()

fig.savefig('/home/sh16450/Documents/repos/Antarctica_clouds/Fig3/Fig4.png', format='PNG', dpi=300)




# cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                     orientation='horizontal', fraction=0.13, pad=0.01, shrink=0.8)
# cbar.set_label(
#     'Average DJF cloud cover 2002-2015 (%)', fontsize=18)
# fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2_new.pdf',
#             format='PDF')
# fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2_new.png',
#             format='PNG', dpi=500)
