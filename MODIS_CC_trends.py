import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import xesmf as xe

# !ice shelves 19) 20) 21) 22) 23) 24) 25) 26)
# !Peninsula 15)  16)  17)  25) 26)
# !East AIS 1)  2)  9) 14) 18) 24)
# !West Ais A 3) 4)  5) 10) 12) 23)
# !West Ais B 6) 7) 8) 11) 13) 21) 22)
# !Ross 19)
# !Ronne-Filchnner  20)
# ===========================
# Load the Data
# ==========================
sns.set_context('paper')

# AVHRR


def preprocess(ds):
    data = ds.sel(lat=slice(-90, -40))
    return data


# Load all the MAR data
MAR = xr.open_dataset(
    '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/wBS/mon-CC-MAR_ERA5-1981-2018.nc')
MAR_noBS = xr.open_dataset(
    '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/noBS/mon-CC-MAR_ERA5-1980-2018.nc')
MAR_grid = xr.open_dataset(
    '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/wBS/MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})


# MAR_grid.drop(['X', 'Y'])

# Add LAT LON to MAR data
MAR['lat'] = ds_grid.LAT
MAR['lon'] = ds_grid.LON
MAR['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)
MAR = MAR.drop_vars(['TIME_bnds'])
MAR = MAR.rename({'TIME': 'time'})

MAR_noBS = MAR_noBS.rename({'TIME': 'time'})
MAR_noBS = MAR_noBS.drop_vars(['TIME_bnds'])
MAR_noBS['lat'] = ds_grid.LAT
MAR_noBS['lon'] = ds_grid.LON


# Load all the CLARA-A2 data
data = xr.open_mfdataset('/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ORD30944/*.nc',
                         preprocess=preprocess)  # read all the data
cloud_data = xr.DataArray(data['cfc'])

# MODIS
data_M = xr.open_dataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/statistics/Aqua_Cloud_Fraction_Mean_Mean.nc')
data_M = data_M.rename({'Begin_Date': 'time'})
cloud_data_M = xr.DataArray(
    data_M['Aqua_Cloud_Fraction_Mean_Mean'].sel(lat=slice(-40, -90)))

data_ERA = xr.open_mfdataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ERA5/SINGLELEVS/ERA5_*.nc', combine='by_coords')
data_ERA = data_ERA.rename({'latitude': 'lat', 'longitude': 'lon'})
cloud_data_ERA = data_ERA.tcc

# ==============================================================================
# Regridding operations
# ==============================================================================
# This creates the output grid, atm I think can be done with any variable
# as long as lat lon grid is present
ds_out = cloud_data_ERA
# Can be any MAR input grid as long as lat lon is present (rename!)
# REGRID MAR
ds_in = MAR
regridder_MAR = xe.Regridder(ds_in, ds_out, 'bilinear')
regridder_MAR_lin = xe.Regridder(ds_in, ds_out, 'nearest_s2d')

# REGRID AVHRR
ds_in_AVHRR = cloud_data
regridder_AVHRR = xe.Regridder(ds_in_AVHRR, ds_out, 'bilinear')

# REGRID MODIS
ds_in_MODIS = cloud_data_M
regridder_MODIS = xe.Regridder(ds_in_MODIS, ds_out, 'bilinear')

# Can try if it also works for the whole dataset by:
# ==============================================================================
# Define functions
# ==============================================================================


def xarray_trend(xarr, dim='time'):
    # getting shapes

    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]

    # creating x and y variables for linear regressioncon
    x = np.arange(0, shape(xarr[dim])[0], 1)
    x = x.reshape(len(x), 1)
    # x = xarr[dim].to_pandas().index.to_julian_date().values[:, None]
    y = xarr.to_masked_array().reshape(n, -1)

    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean
    ya = y - ym  # anomaly
    xa = x - xm  # anomaly

    # variance and covariances
    xss = (xa ** 2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya ** 2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss)**0.5
    t = r * (df / ((1 - r) * (1 + r)))**0.5
    p = stats.distributions.t.sf(abs(t), df)

    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5

    # preparing outputs
    out = xarr[:2].mean(dim)
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name = '_slope'
    xarr_slope.attrs['units'] = 'units / ' + dim
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name = '_Pvalue'
    xarr_p.attrs['info'] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name='slope')
    xarr_out['pval'] = xarr_p

    return xarr_out


if __name__ == '__main__':
    # JJA trends between 2002 and 2015 (MODIS period)
    # CLARA
    AVHRR_CC = cloud_data.loc['2002-07-01':'2015-11-01']
    AVHRR_JJA = AVHRR_CC.where(AVHRR_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_AVHRR = xarray_trend(AVHRR_JJA, dim='year')
    trend_AVHRR_regrid = regridder_AVHRR(trend_AVHRR.slope)

    # MODIS
    MODIS_CC = (cloud_data_M.loc['2002-07-01':'2015-11-01'])*100
    MODIS_JJA = MODIS_CC.where(MODIS_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_MODIS = xarray_trend(MODIS_JJA, dim='year')
    trend_MODIS_regrid = regridder_MODIS(trend_MODIS.slope)

    # MAR with blowing snow
    MAR_CC = (MAR.CC.loc['2002-07-01':'2015-11-1'])*100
    MAR_JJA = MAR_CC.where(MAR_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_MAR = xarray_trend(MAR_JJA, dim='year')
    trend_MAR.coords['lon'] = MAR.lon
    trend_MAR.coords['lat'] = MAR.lat
    trend_MAR_regrid = regridder_MAR(trend_MAR.slope)

    # MAR without Blowing snow
    MAR_CC_noBS = (MAR_noBS.CC.loc['2002-07-01':'2015-11-1'])*100
    MAR_JJA_noBS = MAR_CC_noBS.where(MAR_CC_noBS['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_MAR_noBS = xarray_trend(MAR_JJA_noBS, dim='year')
    trend_MAR_noBS.coords['lon'] = MAR_noBS.lon
    trend_MAR_noBS.coords['lat'] = MAR_noBS.lat
    trend_MAR_regrid_noBS = regridder_MAR(trend_MAR_noBS.slope)

    diff = (trend_MAR_regrid - trend_MAR_regrid_noBS)

    # ERA5
    ERA_CC = (cloud_data_ERA.loc['2002-07-01':'2015-11-01'])*100
    ERA_JJA = ERA_CC.where(ERA_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_ERA = xarray_trend(ERA_JJA, dim='year')
    # ==============================================
    # Compare trends over specific sub-areas of Antarctica
    # !ice shelves 19) 20) 21) 22) 23) 24) 25) 26)
    # !Peninsula 15)  16)  17)  25) 26)
    # !East AIS 1)  2)  9) 14) 18) 24)
    # !West Ais A 3) 4)  5) 10) 12) 23)
    # !West Ais B 6) 7) 8) 11) 13) 21) 22)
    # !Ross 19)
    # !Ronne-Filchnner  20)
    mask_ice_shelves = mask_MAR_regrid.isin(
        [19, 20, 21, 22, 23, 24, 25, 26])

    def cc_seasonal_mask(ds, season='DJF', mask=[19, 26], regriddes_mask=mask_MAR_regrid):
        ds_seasonal = ds.where(
            ds['time.season'] == season).groupby('time.year').mean(dim='time')

        ds_masked = ds_seasonal.where(
            mask_MAR_regrid.isin(mask))

        return ds_masked

    ERA_shelves_CC = cc_seasonal_mask(
        cloud_data_ERA, 'DJF', [19, 20, 21, 22, 23, 24, 25, 26])
    ERA_shelves_msl = cc_seasonal_mask(
        data_ERA.msl, 'DJF', [19, 20, 21, 22, 23, 24, 25, 26])
    ERA_ROSS_CC = cc_seasonal_mask(cloud_data_ERA, 'DJF', [19])
    ERA_ROSS_msl = cc_seasonal_mask(data_ERA.msl/100, 'DJF', [19])
    # ===============================================
    # Compare trends between 2002 and 2015
    names = ['MAR', 'MAR_noBS',
             'MAR - MAR_noBS', 'AVHRR', 'ERA5', 'MODIS']
    proj = ccrs.SouthPolarStereo()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(
        12, 12), subplot_kw={'projection': proj})
    ax = axs.ravel().tolist()

    for i in range(6):
        # Limit the map to -60 degrees latitude and below.
        ax[i].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

        ax[i].add_feature(cartopy.feature.LAND)
        ax[i].add_feature(cartopy.feature.OCEAN)

        ax[i].gridlines()

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax[i].set_boundary(circle, transform=ax[i].transAxes)

    cont = ax[0].pcolormesh(trend_MAR_regrid['lon'], trend_MAR_regrid['lat'],
                            trend_MAR_regrid*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[1].pcolormesh(trend_MAR_regrid_noBS['lon'], trend_MAR_regrid_noBS['lat'],
                     trend_MAR_regrid_noBS*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[2].pcolormesh(trend_MAR_regrid['lon'], trend_MAR_regrid['lat'],
                     diff*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[3].pcolormesh(trend_AVHRR_regrid['lon'], trend_AVHRR_regrid['lat'],
                     trend_AVHRR_regrid*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[5].pcolormesh(trend_MODIS_regrid['lon'], trend_MODIS_regrid['lat'],
                     trend_MODIS_regrid*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[4].pcolormesh(trend_ERA['lon'], trend_ERA['lat'],
                     trend_ERA.slope*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    for i in range(6):
        ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
            '50m'), zorder=1, edgecolor='black')
        ax[i].set_title(names[i], fontsize=16)
    # fig.canvas.draw()
    fig.tight_layout()
    cbar = fig.colorbar(cont, ax=ax, ticks=[-15, -10, -5, 0, 5, 10, 15],
                        orientation='horizontal', fraction=0.13, pad=0.01, shrink=0.8)
    cbar.set_label(
        '2002-07:2015-11 DJF Cloud cover trends * 13 yrs.', fontsize=15)
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_CC_DJF_2002-2015_2x2_new.pdf',
                format='PDF')
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_CC_DJF_2002-2015_2x2_new.png',
                format='PNG', dpi=500)


# =========================================================================

# POTENTIAL OTHER PLOTTING IDEAS
trend_AVHRR_regrid.groupby('lon').mean(dim='lat').plot()
trend_AVHRR_regrid.groupby('lon').mean(dim='lat').plot()
