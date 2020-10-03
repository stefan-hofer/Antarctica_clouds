import xarray as xr
import xesmf as xe
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
# file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'

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
lon = np.arange(-180, 180, 0.25)
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
ds_in = diff.TT.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON, 'x': ds_bs.X,
                               'y': ds_bs.Y})
# Create the regridder
regridder = xe.Regridder(
    ds_in, ds_grid_new, 'bilinear', reuse_weights=True)
# Regrid the data to the 0.25x0.25 grid
ds_TT = regridder(ds_in)
ds_LQS = regridder((diff.CC3D*100).assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                                  'x': ds_bs.X, 'y': ds_bs.Y}))
ds_height = regridder(layer_agl.assign_coords({'lat': ds_bs.LAT, 'lon': ds_bs.LON,
                                               'x': ds_bs.X, 'y': ds_bs.Y}))
# Create the cross section by using xarray interpolation routine
# Interpolate along all lats and 0 longitude (could be any lat lon line)
ds_TT = ds_TT.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_LQS = ds_LQS.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_h = ds_height.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
# mean height of sigma layer (m agl)
mean_sigma = ds_h.sel(
    TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
# =============================================================================
# Plot the cross section
# =============================================================================
fig, axs = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 7), sharex=True, sharey=True)
# ax = axs.ravel().tolist()

# Plot TT using contourf
ds_TT = ds_TT.assign_coords(
    {'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_TT.isel(ATMLAY=slice(9, -1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
                                                                                          robust=True, ax=axs[0], cbar_kwargs={'label': r'$\Delta$ Temperature $(\circ C)$'})


# Plot cloud fraction using contour, with some custom labeling
ds_LQS = ds_LQS.assign_coords(
    {'height': (('ATMLAY'), mean_sigma.values)})
lqs_contour = ds_LQS.isel(ATMLAY=slice(9, -1)).sel(TIME='2009-10-14')[0, :, :].plot.pcolormesh('lat', 'height',
                                                                                               robust=True, ax=axs[1], cbar_kwargs={'shrink': 1, 'label': r'$\Delta$ Cloud Cover (%)'})


fs = 11
for ax in axs:
    ax.set_ylabel('Height agl (m)', fontsize=fs)
    ax.set_xlabel('Latitude', fontsize=fs)
    ax.set_title('')

fig.tight_layout()

sns.despine()
fig.savefig('/home/sh16450/Documents/repos/Antarctica_clouds/Fig4/Fig5.png',
            format='PNG', dpi=300)
