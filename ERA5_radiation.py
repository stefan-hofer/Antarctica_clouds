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

ds = xr.open_mfdataset(
    '/uio/kant/geo-metos-u1/shofer/data/MAR_ANT_35/data/ERA5/SINGLELEVS/ERA5_*.nc', combine='by_coords')


if __name__ == '__main__':
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
    fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Trend_CC_DJF_2002-2015.png',
                format='PNG', dpi=500)
