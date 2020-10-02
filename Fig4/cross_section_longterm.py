import xarray as xr
import xesmf as xe
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
# file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'
# file_str = '/home/sh16450/Dropbox/Academic/Data/Blowing_snow/'

file_str = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly/'
file_str_nobs = '/projects/NS9600K/shofer/blowing_snow/MAR/3D_monthly_nDR/'
file_str_zz = '/projects/NS9600K/shofer/blowing_snow/MAR/case_study_BS_2009/'

year_s = '2000-01-01'
year_e = '2019-12-31'
# Open the no driftig snow file
ds_bs_CC = (xr.open_dataset(
    file_str + 'mon-CC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_TT = (xr.open_dataset(
    file_str + 'mon-TT-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

# Open the drifting snow file
ds_nobs_CC = (xr.open_dataset(
    file_str_nobs + 'mon-CC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_TT = (xr.open_dataset(
    file_str_nobs + 'mon-TT-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

# Open the height file of sigma levels
ds_zz = xr.open_dataset(file_str_zz + 'MAR35_nDS_Oct2009_zz.nc')
layer_agl = (ds_zz.ZZ - ds_zz.SH).rename({'X': 'x', 'Y': 'y'})
masl = (ds_zz.ZZ.rename({'X': 'x', 'Y': 'y'}))
# Read in the grid
MAR_grid = xr.open_dataset(
    file_str_zz + 'MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], ds_nobs_CC.X),
                             'y': (['y'], ds_nobs_CC.Y)})

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
diff = (ds_bs_TT - ds_nobs_TT).rename({'X': 'x', 'Y': 'y'})
diff_CC = (ds_bs_CC - ds_nobs_CC).rename({'X': 'x', 'Y': 'y'})


diff['LAT'] = ds_grid.LAT
diff['LON'] = ds_grid.LON

diff_CC['LAT'] = ds_grid.LAT
diff_CC['LON'] = ds_grid.LON

# Cross section lat lons
start = (-90, 0)
end = (-65, 0)

# This also works to plot a cross section but on MAR grid
# diff.SWNC3D.sel(X=0,Y=slice(-2400,2400),TIME='2009-10-14').plot()
# Create regridder
ds_in = diff.TT.assign_coords({'lat': diff.LAT, 'lon': diff.LON, 'x': diff.x,
                               'y': diff.y})
# Create the regridder
regridder = xe.Regridder(
    ds_in, ds_grid_new, 'bilinear')
# Regrid the data to the 0.25x0.25 grid
ds_TT = regridder(ds_in)
ds_LQS = regridder((diff_CC.CC3D*100).assign_coords({'lat': diff.LAT, 'lon': diff.LON,
                                                     'x': diff.x, 'y': diff.y}))
ds_height = regridder(layer_agl.assign_coords({'lat': diff.LAT, 'lon': diff.LON,
                                               'x': diff.x, 'y': diff.y}))

ds_height_masl = regridder(masl.assign_coords({'lat': diff.LAT, 'lon': diff.LON,
                                               'x': diff.x, 'y': diff.y}))
# Create the cross section by using xarray interpolation routine
# Interpolate along all lats and 0 longitude (could be any lat lon line)
ds_TT = ds_TT.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_LQS = ds_LQS.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_h = ds_height.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_h_masl = ds_height_masl.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
# mean height of sigma layer (m agl)
# mean_sigma = ds_h.sel(
#     TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
# mean_masl = ds_h_masl.sel(
#     TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]

mean_sigma = ds_h.sel(
    TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
mean_masl = ds_h_masl.sel(TIME='2009-10-14')[0, :, :]


# =============================================================================
# Plot the cross section
# =============================================================================
fig, axs = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 7), sharex=True, sharey=True)
# ax = axs.ravel().tolist()

# Plot TT using contourf
ds_TT = ds_TT.assign_coords(
    {'height': (('ATMLAY'), mean_sigma.values)})
contour = ds_TT.isel(ATMLAY=slice(9, -1)).plot.pcolormesh('lat', 'height', robust=True,
                                                          ax=axs[0], cbar_kwargs={'label': r'$\Delta$ Temperature $(\circ C)$'})


# Plot cloud fraction using contour, with some custom labeling
ds_LQS = ds_LQS.assign_coords(
    {'height': (('ATMLAY'), mean_sigma.values)})
lqs_contour = ds_LQS.isel(ATMLAY=slice(9, -1)).plot.pcolormesh('lat', 'height',
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

# ==================================
# === SECOND OPTION PLOT ===========
# ==================================
# RESCALE THE height axis
ds_TT = ds_TT.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl.values)})
ds_LQS = ds_LQS.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl.values)})


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


source_scale = (4000, 11000)  # Scale values between 100 and 600
destination_scale = (4000, 6000)  # to a scale between 100 and 150

# create the scaled height array
height_scaled = scale(ds_LQS.height, source_scale, destination_scale)
# Replace only values where height is greater than 4000
mean_masl = mean_masl.where(mean_masl < 4000).fillna(height_scaled)


# ax.plot(data_scaled)


# ==================================================================
fig, axs = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 4), sharex=True, sharey=True)
# Set the y-ticks to a custom scale
for ax in axs:
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 4333,
                   4666, 5000, 5333, 5666, 6000, 6333])
    ax.set_ylim(0, 5000)
    # Set the labels to the actual values
    ax.set_yticklabels(["0", "1000", "2000", "3000", "4000",
                        "5000", "6000", "7000", "8000", "9000", "10000", "11000"])
# ax = axs.ravel().tolist()
# TESTING

contour = ds_TT.isel(ATMLAY=slice(0, -1)).plot.pcolormesh('lat', 'height', robust=True,
                                                          ax=axs[0], cbar_kwargs={'label': r'$\Delta$ Temperature $(\circ C)$'})


# Plot cloud fraction using contour, with some custom labeling

lqs_contour = ds_LQS.isel(ATMLAY=slice(0, -1)).plot.pcolormesh('lat', 'height',
                                                               robust=True, ax=axs[1], cbar_kwargs={'shrink': 1, 'label': r'$\Delta$ Cloud Cover (%)'})


for ax in axs:
    fs = 11
    ax.set_ylabel('Height amsl (m)', fontsize=fs)
    ax.set_xlabel('Latitude', fontsize=fs)
    ax.set_title('')

fig.tight_layout()
sns.despine()
