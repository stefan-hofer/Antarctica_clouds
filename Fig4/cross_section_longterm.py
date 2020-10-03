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
ds_bs_LWNC = (xr.open_dataset(
    file_str + 'mon-LWNC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_LWN = (xr.open_dataset(
    file_str + 'mon-LWN3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWNC = (xr.open_dataset(
    file_str + 'mon-SWNC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWN = (xr.open_dataset(
    file_str + 'mon-SWN3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_SWN = (xr.open_dataset(
    file_str + 'mon-SWN3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_bs_COD = (xr.open_dataset(
    file_str + 'mon-COD3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')


# Calculate the CRE for the blowing snow simulations
CRE_bs = (ds_bs_SWN.SWN3D + ds_bs_LWN.LWN3D) - \
    (ds_bs_SWNC.SWNC3D + ds_bs_LWNC.LWNC3D)
# Open the drifting snow file
ds_nobs_CC = (xr.open_dataset(
    file_str_nobs + 'mon-CC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_TT = (xr.open_dataset(
    file_str_nobs + 'mon-TT-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWNC = (xr.open_dataset(
    file_str_nobs + 'mon-LWNC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_LWN = (xr.open_dataset(
    file_str_nobs + 'mon-LWN3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWNC = (xr.open_dataset(
    file_str_nobs + 'mon-SWNC3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_SWN = (xr.open_dataset(
    file_str_nobs + 'mon-SWN3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')
ds_nobs_COD = (xr.open_dataset(
    file_str_nobs + 'mon-COD3D-MAR_ERA5-1979-2019.nc')).sel(TIME=slice(year_s, year_e)).mean(dim='TIME')

# CRE_diff.plot(x='X', y='Y', col='ATMLAY', col_wrap=5)

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
# Calculate the CRE for the nobs case
CRE_nobs = (ds_nobs_SWN.SWN3D + ds_nobs_LWN.LWN3D) - \
    (ds_nobs_SWNC.SWNC3D + ds_nobs_LWNC.LWNC3D)
# Difference in CRE between bs and nobs simulation
CRE_diff = CRE_bs - CRE_nobs
# DIFF COD3D
diff_COD = (ds_bs_COD.COD3D -
            ds_nobs_COD.COD3D).rename({'X': 'x', 'Y': 'y'})
# Difference between ds and nobs
diff = (ds_bs_TT - ds_nobs_TT).rename({'X': 'x', 'Y': 'y'})
diff_CC = (ds_bs_CC - ds_nobs_CC).rename({'X': 'x', 'Y': 'y'})
diff_CRE = CRE_diff.rename({'X': 'x', 'Y': 'y'})


diff['LAT'] = ds_grid.LAT
diff['LON'] = ds_grid.LON

diff_CC['LAT'] = ds_grid.LAT
diff_CC['LON'] = ds_grid.LON

diff_CRE['LAT'] = ds_grid.LAT
diff_CRE['LON'] = ds_grid.LON

diff_COD['LAT'] = ds_grid.LAT
diff_COD['LON'] = ds_grid.LON
# Cross section lat lons
start = (-90, 140)
end = (-65, 140)

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
ds_CRE = regridder(diff_CRE.assign_coords({'lat': diff.LAT, 'lon': diff.LON,
                                           'x': diff.x, 'y': diff.y}))
ds_COD = regridder(diff_COD.assign_coords({'lat': diff.LAT, 'lon': diff.LON,
                                           'x': diff.x, 'y': diff.y}))
# Create the cross section by using xarray interpolation routine
# Interpolate along all lats and 0 longitude (could be any lat lon line)


def merge_over_south_pole(ds, lon, lat_start=-90, lat_end=-65):
    new_lats = np.arange(lat_end, lat_start, -0.25)
    ds_test = ds.interp(lat=new_lats, lon=lon-180)
    # fake that it is the actual longitude
    ds_test['lon'] = lon
    ds_test.attrs['actual_lon'] = lon - 180

    ds_orig = ds.interp(lat=np.arange(
        lat_start, lat_end, 0.25), lon=lon)
    ds_orig['lat'] = -180 - ds_orig['lat']
    # Merge the cross section over the South Pole
    merged = xr.merge([ds_test, ds_orig])
    # Where LATS are lower or equal to -90 is where the original lon is
    return merged


merged_TT = merge_over_south_pole(ds_TT, 140)
merged_CC = merge_over_south_pole(ds_LQS, 140)
ds_CRE.name = 'CRE'
merged_CRE = merge_over_south_pole(ds_CRE, 140)

ds_COD.name = "COD"
merged_COD = merge_over_south_pole(ds_COD, 140)

ds_height.name = 'height'
merged_h = merge_over_south_pole(ds_height, 140)
merged_masl = merge_over_south_pole(ds_height_masl, 140)

ds_TT = ds_TT.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_LQS = ds_LQS.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_h = ds_height.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_h_masl = ds_height_masl.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_cre = ds_CRE.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
ds_cod = ds_COD.interp(lat=np.arange(
    start[0], end[0], 0.25), lon=start[1])
# mean height of sigma layer (m agl)
# mean_sigma = ds_h.sel(
#     TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
# mean_masl = ds_h_masl.sel(
#     TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]

mean_sigma = ds_h.sel(
    TIME='2009-10-14').where(ds_h.lat < -70).mean(dim='lat')[0, :]
mean_masl = ds_h_masl.sel(TIME='2009-10-14')[0, :, :]
mean_masl_merged = merged_masl.ZZ.sel(TIME='2009-10-14')[0, :, :]


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
ds_TTm = merged_TT.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl_merged.values)})
ds_LQSm = merged_CC.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl_merged.values)})
ds_CREm = merged_CRE.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl_merged.values)})
# ds_CODm = merged_COD.assign_coords(
    {'height': ((['ATMLAY', 'lat']), mean_masl_merged.values)})


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


source_scale=(4000, 11000)  # Scale values between 100 and 600
destination_scale=(4000, 6000)  # to a scale between 100 and 150

# create the scaled height array
# height_scaled = scale(ds_LQS.height, source_scale, destination_scale)
height_scaled=scale(
    ds_LQSm.height, source_scale, destination_scale)
# Replace only values where height is greater than 4000
# mean_masl = mean_masl.where(mean_masl < 4000).fillna(height_scaled)
mean_masl=mean_masl_merged.where(
    mean_masl_merged < 4000).fillna(height_scaled)

# ax.plot(data_scaled)


# ==================================================================
fig, axs=plt.subplots(
    nrows = 3, ncols = 1, figsize = (5, 10), sharex = True, sharey = True)
# Set the y-ticks to a custom scale
for ax in axs.flatten():
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 4333,
                   4666, 5000, 5333, 5666, 6000, 6333])
    ax.set_ylim(0, 5000)
    # Set the labels to the actual values
    ax.set_yticklabels(["0", "1000", "2000", "3000", "4000",
                        "5000", "6000", "7000", "8000", "9000", "10000", "11000"])
    ax.set_xticks([-110, -100, -90, -80, -70])
    ax.set_xticklabels(["-70", "-80", "-90", "-80", "-70"])
# ax = axs.ravel().tolist()
# TESTING

contour=ds_TTm.TT.isel(ATMLAY = slice(0, -1)).plot.pcolormesh('lat', 'height', robust = True,
                                                              ax = axs[0], cbar_kwargs = {'label': r'$\Delta$ Temperature $(\circ C)$'})


# Plot cloud fraction using contour, with some custom labeling

lqs_contour=ds_LQSm.CC3D.isel(ATMLAY = slice(0, -1)).plot.pcolormesh('lat', 'height',
                                                                     robust = True, ax = axs[1], cbar_kwargs = {'shrink': 1, 'label': r'$\Delta$ Cloud Cover (%)'})
cre_contour=ds_CREm.CRE.isel(ATMLAY = slice(0, -1)).plot.pcolormesh('lat', 'height',
                                                                    robust=True, ax=axs[2], cbar_kwargs={'shrink': 1, 'label': r'$\Delta$ CRE ($Wm^{-2}$)'})
# cod_contour = ds_CODm.COD.isel(ATMLAY=slice(0, -1)).plot.pcolormesh('lat', 'height',
#                                                                robust=True, ax=axs[1][1], cbar_kwargs={'shrink': 1, 'label': r'$\Delta$ COD (unitless)'})

for ax in axs.flatten():
    fs = 11
    ax.set_ylabel('Height amsl (m)', fontsize=fs)
    ax.set_xlabel('Latitude', fontsize=fs)
    ax.set_title('')

fig.tight_layout()
sns.despine()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/cross_section_lt.png',
            format='PNG', dpi=300)
