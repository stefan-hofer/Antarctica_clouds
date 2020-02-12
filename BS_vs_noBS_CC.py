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
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/mon-CC-MAR_ERA5-1981-2018.nc')
MAR_noBS = xr.open_dataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/MAR_noBS/mon-CC-MAR_ERA5-1980-2018.nc')
MAR_grid = xr.open_dataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/MARcst-AN35km-176x148.cdf')
ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})

# MAR_grid.drop(['X', 'Y'])

# Add LAT LON to MAR data
for data in [MAR, MAR_noBS]:
    data['lat'] = ds_grid.LAT
    data['lon'] = ds_grid.LON
    data['RIGNOT'] = ds_grid.RIGNOT.where(ds_grid.RIGNOT > 0)

MAR = MAR.drop_vars(['TIME_bnds'])
MAR = MAR.rename({'TIME': 'time'})

MAR_noBS = MAR_noBS.drop_vars(['TIME_bnds'])
MAR_noBS = MAR_noBS.rename({'TIME': 'time'})


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

# =============================
# Create functions
# ==============================


def summer_mean(ds, start='2002-07-01', end='2015-11-01', season='DJF'):
    if start:
        ds = ds.loc[start:end]

    ds_JJA = ds.where(ds['time.season' == 'DJF']).groupby(
        'time.year').mean(dim='time')
    ds_JJA_climatology = ds_JJA.mean(dim='year')

    return ds_JJA_climatology


if __name__ == '__main__':

    # JJA trends between 2002 and 2015 (MODIS period)
    # CLARA

    AVHRR_summer_mean = summer_mean(data['cfc'])
    clim_AVHRR_regrid = regridder_AVHRR(AVHRR_summer_mean)

    # MODIS
    MODIS_summer_mean = summer_mean(
        data_M['Aqua_Cloud_Fraction_Mean_Mean'].sel(lat=slice(-40, -90))*100)
    clim_MODIS_regrid = regridder_MODIS(MODIS_summer_mean)

    # MAR
    MAR_summer_mean = summer_mean((MAR.CC)*100)
    clim_MAR_regrid = regridder_MAR(MAR_summer_mean)

    MAR_noBS_summer_mean = summer_mean((MAR_noBS.CC)*100)
    clim_MAR_noBS_regrid = regridder_MAR(MAR_noBS_summer_mean)
    diff = clim_MAR_regrid - clim_MAR_noBS_regrid
    # Plot e.g. (mask_MAR_regrid == 20).plot()
    # This is the RIGNOT mask on the ERA5 grid
    mask_MAR_regrid = xr.ufuncs.rint(regridder_MAR_lin(MAR.RIGNOT))

    # ERA5
    ERA_summer_mean = summer_mean((data_ERA.tcc)*100)

    # Plotting
    names = ['MAR', 'MAR_No_Blowing_Snow',
             'MAR-MAR_noBS', 'AVHRR', 'ERA5', 'MODIS']
    # Compare trends between 2002 and 2015
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

    cmap = 'YlGnBu_r'
    cont = ax[0].pcolormesh(clim_MAR_regrid['lon'], clim_MAR_regrid['lat'],
                            clim_MAR_regrid, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    cont = ax[1].pcolormesh(clim_MAR_noBS_regrid['lon'], clim_MAR_noBS_regrid['lat'],
                            clim_MAR_noBS_regrid, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    cont = ax[2].pcolormesh(clim_MAR_noBS_regrid['lon'], clim_MAR_noBS_regrid['lat'],
                            diff, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    ax[3].pcolormesh(clim_AVHRR_regrid['lon'], clim_AVHRR_regrid['lat'],
                     clim_AVHRR_regrid, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    ax[5].pcolormesh(clim_MODIS_regrid['lon'], clim_MODIS_regrid['lat'],
                     clim_MODIS_regrid, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    ax[4].pcolormesh(ERA_summer_mean['lon'], ERA_summer_mean['lat'],
                     ERA_summer_mean, transform=ccrs.PlateCarree(), vmin=0, vmax=100, cmap=cmap)
    for i in range(6):
        ax[i].add_feature(cartopy.feature.COASTLINE.with_scale(
            '50m'), zorder=1, edgecolor='black')
        ax[i].set_title(names[i], fontsize=16)
    # fig.canvas.draw()
    fig.tight_layout()
    cbar = fig.colorbar(cont, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        orientation='horizontal', fraction=0.13, pad=0.01, shrink=0.8)
    cbar.set_label(
        'Average DJF cloud cover 2002-2015 (%)', fontsize=18)
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2.pdf',
                format='PDF')
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2.png',
                format='PNG', dpi=500)
