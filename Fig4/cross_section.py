import xarray as xr
import xesmf as xe
from datetime import datetime
import seaborn as sns

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units
from metpy.interpolate import cross_section

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

# Create a new grid for interpolation of MAR data via xesmf
# 0.25 deg resolution
lon = np.arange(0, 360, 0.25)
lat = np.arange(-90, -55, 0.25)
# fake variable of zeros
ds_var = np.zeros([shape(lat)[0], shape(lon)[0]])
# New array
ds_grid_new = xr.Dataset({'variable': (['lat', 'lon'], ds_var)},
                         coords={'lat': (['lat'], lat),
                                 'lon': (['lon'], lon)})



# ================================================================
# =========== ANALYSIS ===========================================
# ================================================================
# Difference between ds and nods
diff = ds_bs - ds_nobs
diff['LAT'] = ds_grid.LAT
diff['LON'] = ds_grid.LON
# Create metpy accesory
data = diff.assign_coords({'lat':ds_bs.LAT, 'lon': ds_bs.LON, 'x': ds_bs.X, 'y': ds_bs.Y}).metpy.parse_cf()
# Cross section along 69.3 latitude and between 1 and 25 longitude
# Andenes = 16deg longitude
start = (-90, 0)
end = (-65, 0)

# This also works to plot a cross section
# diff.SWNC3D.sel(X=0,Y=slice(-2400,2400),TIME='2009-10-14').plot()
# Create regridder
ds_in = diff.TT.assign_coords({'lat':ds_bs.LAT, 'lon': ds_bs.LON, 'x': ds_bs.X, 'y': ds_bs.Y})
# Create the regridder
regridder = xe.Regridder(ds_in, ds_grid_new, 'bilinear', reuse_weights=True)
# Regrid the data
ds_TT = regridder(ds_in)
ds_LQS = regridder(diff.CC3D.assign_coords({'lat':ds_bs.LAT, 'lon': ds_bs.LON,
                                          'x': ds_bs.X, 'y': ds_bs.Y}))
# Create the cross section
ds_TT = ds_TT.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])
ds_LQS = ds_LQS.interp(lat=np.arange(start[0], end[0], 0.25), lon=start[1])


# =============================================================================
# Plot the cross section
# =============================================================================
fig, axs = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 7), sharex=True, sharey=True)
# ax = axs.ravel().tolist()

# Plot RH using contourf
contour = (ds_TT.isel(ATMLAY=slice(10,-1))
           .sel(TIME='2009-10-14')[0, :, :].plot(robust=True, ax=axs[0],
           cbar_kwargs={'label': r'$\Delta$ Temperature $(\circ C)$'}, yincrease=False))

# Plot cloud fraction using contour, with some custom labeling
lqs_contour = (ds_LQS.isel(ATMLAY=slice(10,-1))
               .sel(TIME='2009-10-14')[0, :, :].plot(robust=True, ax=axs[1], yincrease=False,
               cbar_kwargs={'shrink': 1, 'label': r'$\Delta$ Cloud Cover (%)'}))

fs = 11
for ax in axs:
    ax.set_ylabel('Sigma level', fontsize=fs)
    ax.set_xlabel('Latitude', fontsize=fs)
    ax.set_title('')

fig.tight_layout()

sns.despine()
fig.savefig('/home/sh16450/Documents/repos/Antarctica_clouds/Fig4/Fig5.png', format='PNG', dpi=300)
