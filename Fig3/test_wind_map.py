import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import xarray as xr

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'


ds_atm = xr.open_dataset(file_str + 'MAR35_DS_Oct2009_UVSMT.nc')
MAR_grid = xr.open_dataset(
    file_str + 'MARcst-AN35km-176x148.cdf')

ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})


ds_wind = xr.Dataset(ds_atm[['UU', 'VV', 'UV']].sel(TIME='2009-10-14').isel(TIME=0),
                coords={'lat': (('Y', 'X'), ds_grid.LAT), 'lon': (('Y', 'X'), ds_grid.LON)})
ds_mpy = ds_wind.metpy.parse_cf()
# Set extent of the map for plotting
lat_min = -90
lat_max = -60
lon_min = -180  # 70
lon_max = 180

proj = ccrs.SouthPolarStereo()
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(
    16, 16), subplot_kw={'projection': proj})
# ax = axs.ravel().tolist()
axs.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
# Add wind speed filled contours
ds_wind.UV.isel(ATMLAY=19).plot.pcolormesh('lon', 'lat', ax=axs, transform=ccrs.PlateCarree(),
                                               robust=True, cbar_kwargs={'shrink': 0.7})
# regrid shape controls the density of the wind barbs
axs.barbs(ds_mpy.X.values, ds_mpy.Y.values, ds_wind.UU.isel(ATMLAY=19).values, ds_wind.isel(ATMLAY=19).VV.values,
          transform=ccrs.SouthPolarStereo(),regrid_shape=35, length=5, linewidth=1, alpha=0.8)
# regrid_shape=35, length=5, linewidth=1, alpha=0.8
# add the coastline
axs.add_feature(cartopy.feature.COASTLINE.with_scale(
    '50m'), zorder=1, edgecolor='black')
