import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
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
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/mon-CC-MAR_ERA5-1981-2018.nc')
MAR_grid = xr.open_dataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/MARcst-AN35km-176x148.cdf')
# Add LAT LON to MAR data
MAR['LAT'] = MAR_grid.LAT
MAR['LON'] = MAR_grid.LON


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
    # JJA trends between 1995-2016
    # CLARA
    AVHRR_CC = cloud_data.loc['2002-07-01':'2015-11-01']
    AVHRR_JJA = AVHRR_CC.where(AVHRR_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_AVHRR = xarray_trend(AVHRR_JJA, dim='year')
#    trend_A = regridder_AVHRR(trend_AVHRR.slope)

    MODIS_CC = (cloud_data_M.loc['2002-07-01':'2015-11-01'])*100
    MODIS_JJA = MODIS_CC.where(MODIS_CC['time.season'] == 'DJF').groupby(
        'time.year').mean(dim='time')
    trend_MODIS = xarray_trend(MODIS_JJA, dim='year')

    MAR_CC = (MAR.CC.loc['2002-07-01':'2015-11-1'])*100
    MAR_JJA = MAR_CC.where(MAR_CC['TIME.season'] == 'DJF').groupby(
        'TIME.year').mean(dim='TIME')
    trend_MAR = xarray_trend(MAR_JJA, dim='year')
    trend_MAR.coords['LON'] = MAR.LON
    trend_MAR.coords['LAT'] = MAR.LAT
# ===============================================
    # NICE PLOT OF RACMO DATA
    proj = ccrs.SouthPolarStereo()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(
        16, 24), subplot_kw={'projection': proj})
    ax = axs.ravel().tolist()

    for i in range(3):
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

    cont = ax[0].pcolormesh(MAR['LON'], MAR['LAT'],
                            trend_MAR.slope*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[1].pcolormesh(trend_AVHRR['lon'], trend_AVHRR['lat'],
                     trend_AVHRR.slope*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')
    ax[2].pcolormesh(trend_MODIS['lon'], trend_MODIS['lat'],
                     trend_MODIS.slope*13, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')

    for i in range(3):
        ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
            '50m'), zorder=1, edgecolor='black')
    fig.canvas.draw()
    fig.tight_layout()
    cbar = fig.colorbar(cont, ax=ax, ticks=[-15, -10, -5, 0, 5, 10, 15],
                        orientation='horizontal', fraction=0.13, pad=0.01, shrink=0.8)
    cbar.set_label(
        '2002-07:2015-11 DJF Cloud cover trends * 13 yrs.', fontsize=15)
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_CC_DJF_2002-2015.pdf',
                format='PDF')
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_CC_DJF_2002-2015.pdf',
                format='PNG', dpi=500)


# =========================================================================


fig, axs = plt.subplots(nrows=1, ncols=2)
trend_AVHRR.slope.plot(ax=axs[0])
(trend_MODIS.slope * 100).plot(ax=axs[1])
