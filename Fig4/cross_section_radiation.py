import xarray as xr
import xesmf as xe
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

# Open the no driftig snow file
ds_nobs = xr.open_dataset(
    file_str + 'MAR35_nDS_Oct2009.nc')

# Open the drifting snow file
ds_bs = xr.open_dataset(
    file_str + 'MAR35_DS_Oct2009.nc')

# Open the height file of sigma levels
ds_zz = xr.open_dataset(file_str + 'MAR35_nDS_Oct2009_zz.nc')
layer_agl = ds_zz.ZZ - ds_zz.SH
# Read in the grid
MAR_grid = xr.open_dataset(
    file_str + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})

# Create a new grid for interpolation of MAR data via xesmf
# 0.25 deg resolution rectangual lat/lon grid
lon = np.arange(0, 360, 0.25)
lat = np.arange(-90, -55, 0.25)
# fake variable of zeros
ds_var = np.zeros([shape(lat)[0], shape(lon)[0]])
# New array of the new grid on which to interpolate on
ds_grid_new = xr.Dataset({'variable': (['lat', 'lon'], ds_var)},
                         coords={'lat': (['lat'], lat),
                                 'lon': (['lon'], lon)})

# ================================================================
# =========== ANALYSIS ===========================================
# ================================================================
# Difference between ds and nobs
diff = ds_bs - ds_nobs
diff['LAT'] = ds_grid.LAT
diff['LON'] = ds_grid.LON
# Cross section lat lons
start = (-90, 0)
end = (-65, 0)

# This also works to plot a cross section but on MAR grid
# diff.SWNC3D.sel(X=0,Y=slice(-2400,2400),TIME='2009-10-14').plot()
# Create regridder
ds_in = diff.LWN3D.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON, 'x': ds_bs.X,
                              'y': ds_bs.Y})
# Create the regridder
regridder = xe.Regridder(ds_in, ds_grid_new, 'bilinear', reuse_weights=True)
# Regrid the data to the 0.25x0.25 grid
ds_LWN3D = regridder(ds_in)
ds_SWN3D = regridder(diff.SWN3D.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                           'x': ds_bs.X, 'y': ds_bs.Y}))
ds_SWNC3D = regridder(diff.SWNC3D.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                           'x': ds_bs.X, 'y': ds_bs.Y}))
ds_LWNC3D = regridder(diff.LWNC3D.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                           'x': ds_bs.X, 'y': ds_bs.Y}))
ds_height = regridder(layer_agl.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                              'x': ds_bs.X, 'y': ds_bs.Y}))
# Create the cross section by using xarray interpolation routine
# Interpolate along all lats and 0 longitude (could be any lat lon line)
ds_LWN3D = ds_LWN3D.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
ds_SWN3D = ds_SWN3D.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
net = ds_SWN3D + ds_LWN3D
ds_SWNC3D = ds_SWNC3D.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
ds_LWNC3D = ds_LWNC3D.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
net_clear = ds_SWNC3D + ds_LWNC3D
ds_h = ds_height.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
# mean height of sigma layer (m agl)
mean_sigma = ds_h.sel(TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
# =============================================================================
# Plot the cross section
# =============================================================================
fig, axs = plt.subplots(
    nrows=2, ncols=3, figsize=(16, 14), sharex=True, sharey=True)
axs = axs.ravel().tolist()

# Plot LWnet diff using contourf
ds_LWN3D = ds_LWN3D.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_LWN3D.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[0], cbar_kwargs={'label': r'$\Delta$ LWnet $(Wm^{2})$'})

# Plot SWnet diff using contourf
ds_SWN3D = ds_SWN3D.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_SWN3D.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[1], cbar_kwargs={'label': r'$\Delta$ SWnet $(Wm^{2})$'})
# Plot Net diff using contourf
net = net.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = net.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[2], cbar_kwargs={'label': r'$\Delta$ Net Rad. $(Wm^{2})$'})

# Plot LWnet clear diff using contourf
ds_LWNC3D = ds_LWNC3D.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_LWNC3D.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[3], cbar_kwargs={'label': r'$\Delta$ LWnet clear $(Wm^{2})$'})

# Plot LWnet clear diff using contourf
ds_SWNC3D = ds_SWNC3D.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_SWNC3D.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[4], cbar_kwargs={'label': r'$\Delta$ SWnet clear $(Wm^{2})$'})

# Plot Net diff using contourf
net_clear = net_clear.assign_coords({'height': (('ATMLAY'), mean_sigma.values)})
contour = net_clear.isel(ATMLAY=slice(10,-1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
          robust=True, ax=axs[5], cbar_kwargs={'label': r'$\Delta$ Clear Net Rad. $(Wm^{2})$'})

fs = 11
for ax in axs:
    ax.set_ylabel('Height agl (m)', fontsize=fs)
    ax.set_xlabel('Latitude', fontsize=fs)
    ax.set_title('')

fig.tight_layout()

sns.despine()
fig.savefig('/home/sh16450/Documents/repos/Antarctica_clouds/Fig4/Fig6.png', format='PNG', dpi=300)
