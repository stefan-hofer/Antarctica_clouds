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

# Open TD and TT (2m) data from ERA5
ds = xr.open_mfdataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ERA5/SINGLELEVS/TT_TD/ERA5_*_TT_TD_singlelevs_mthly.nc')
ds = ds.drop_vars('expver')

ds_new = xr.Dataset({'TT': (['time', 'lat', 'lon'], ds.t2m.values[:, :, :, 0]),
                     'TD': (['time', 'lat', 'lon'], ds.d2m.values[:, :, :, 0])},
                    coords={'time': (['time'], ds.time),
                            'lat': (['lat'], ds.latitude.values),
                            'lon': (['lon'], ds.longitude.values),
                            })
data_ERA = xr.open_mfdataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ERA5/SINGLELEVS/ERA5_*_singlelevs_mthly.nc', combine='by_coords')
data_ERA = data_ERA.rename({'latitude': 'lat', 'longitude': 'lon'})


def summer_mean(ds, start='2002-07-01', end='2015-11-01', season='DJF'):
    ds_new = ds.loc[start:end]
    if season == 'annual':
        ds_JJA = ds_new.groupby(
            'time.year').mean(dim='time')
    else:
        ds_JJA = ds_new.where(ds_new['time.season'] == season).groupby(
            'time.year').mean(dim='time')
    ds_JJA_climatology = ds_JJA.mean(dim='year')

    return ds_JJA_climatology, ds_JJA


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
    # DJF
    climatology_TT, JJA_TT = summer_mean(
        ds_new.TT, start='1990-06-01', end='2019-11-01')
    climatology_TT_ann, ann_TT = summer_mean(
        ds_new.TT, start='1990-06-01', end='2019-11-01', season='annual')
    trend_TT = xarray_trend(JJA_TT, dim='year')
    trend_TT_ann = xarray_trend(ann_TT, dim='year')

    climatology_CC, JJA_CC = summer_mean(
        data_ERA.tcc*100, start='1990-06-01', end='2019-11-01')
    trend_CC = xarray_trend(JJA_CC, dim='year')

    # Plotting
    names = [
        'ERA5 DJF Temp. Trend (1990-2019)', 'ERA5 annual Temp. Trend (1990-2019)']
    # Compare trends between 2002 and 2015
    proj = ccrs.SouthPolarStereo()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(
        12, 6), subplot_kw={'projection': proj})
    ax = axs.ravel().tolist()

    for i in range(2):
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

    cmap = 'YlGnBu_r'
    cont = ax[0].pcolormesh(trend_TT['lon'], trend_TT['lat'],
                            trend_TT.slope*30, transform=ccrs.PlateCarree(), vmin=-3, vmax=3, cmap='RdBu_r')
    cont2 = ax[1].pcolormesh(trend_TT_ann['lon'], trend_TT_ann['lat'],
                             trend_TT_ann.slope*30, transform=ccrs.PlateCarree(), vmin=-3, vmax=3, cmap='RdBu_r')
    # cont2 = ax[1].pcolormesh(trend_CC['lon'], trend_CC['lat'],
    #                          trend_CC.slope*30, transform=ccrs.PlateCarree(), vmin=-15, vmax=15, cmap='RdBu_r')

    for i in range(2):
        ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
            '50m'), zorder=1, edgecolor='black')
        ax[i].set_title(names[i], fontsize=16)
    # fig.canvas.draw()
    fig.tight_layout()
    fig.colorbar(cont, ax=ax[0], ticks=list(
        np.arange(-3, 3.5, 1)), shrink=0.8)
    fig.colorbar(cont2, ax=ax[1], ticks=list(
        np.arange(-3, 3.5, 1)), shrink=0.8)
    # fig.colorbar(cont2, ax=ax[1], ticks=list(
    #     np.arange(-15, 15.5, 3)), shrink=0.8)
    # cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #                    orientation = 'horizontal', fraction = 0.13, pad = 0.01, shrink = 0.8)
    # cbar.set_label(
    #    'Average DJF cloud cover 2002-2015 (%)', fontsize=18)
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_TT_DJF_1990-2019_1x2.pdf',
                format='PDF')
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_TT_DJF_1990-2019_1x2.png',
                format='PNG', dpi=500)
