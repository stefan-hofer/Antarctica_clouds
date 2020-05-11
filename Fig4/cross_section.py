import xarray as xr
import metpy as mp
from datetime import datetime


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

# ================================================================
# =========== ANALYSIS ===========================================
# ================================================================
# Difference between ds and nods
diff = ds_bs - ds_nobs
diff['LAT'] = ds_grid.LAT
diff['LON'] = ds_grid.LON
# Create metpy accesory
data = diff.assign_coords({'lat':ds_bs.LAT, 'lon': ds_bs.LON, 'x': ds_bs.X, 'y': ds_bs.Y}).metpy.parse_cf().squeeze()
# Cross section along 69.3 latitude and between 1 and 25 longitude
# Andenes = 16deg longitude
start = (-90, 0)
end = (-55, 0)

cross = cross_section(data, start, end).set_coords(('lat', 'lon'))


# Cross section along 69.3 latitude and between 1 and 25 longitude
# Andenes = 16deg longitude
start = (69.3, 1)
end = (69.3, 25)
