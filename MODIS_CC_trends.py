import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# ===========================
# Load the Data
# ==========================
sns.set_context('paper')

# AVHRR
def preprocess(ds):
    data = ds.sel(lat=slice(-90,-40))
    return data
# Load all the CLARA-A2 data
data = xr.open_mfdataset('/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ORD30944/*.nc',
                         preprocess=preprocess) # read all the data
cloud_data = xr.DataArray(data['cfc'])

# MODIS
data_M = xr.open_dataset('/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/statistics/Aqua_Cloud_Fraction_Mean_Mean.nc')
data_M = data_M.rename({'Begin_Date':'time'})
cloud_data_M = xr.DataArray(data_M['Aqua_Cloud_Fraction_Mean_Mean'].sel(lat=slice(-40,-90)))

# ==============================================================================
# Define functions
# ==============================================================================
def xarray_trend(xarr, dim='time'):
    # getting shapes

    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]

    # creating x and y variables for linear regressioncon
    x = np.arange(0,shape(xarr[dim])[0], 1)
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
    AVHRR_JJA = AVHRR_CC.where(AVHRR_CC['time.season'] == 'DJF').groupby('time.year').mean(dim='time')
    trend_AVHRR = xarray_trend(AVHRR_JJA, dim='year')
#    trend_A = regridder_AVHRR(trend_AVHRR.slope)

    MODIS_CC = cloud_data_M.loc['2002-07-01':'2015-11-01']
    MODIS_JJA = MODIS_CC.where(MODIS_CC['time.season'] == 'DJF').groupby('time.year').mean(dim='time')
    trend_MODIS = xarray_trend(MODIS_JJA, dim='year')

    fig, axs = plt.subplots(nrows=1, ncols=2)
    trend_AVHRR.slope.plot(ax=axs[0])
    (trend_MODIS.slope * 100).plot(ax=axs[1])
