import xarray as xr
import xesmf as xe
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import pandas as pd
import glob
import datetime as dt
import matplotlib.path as mpath

import cartopy.feature as feat


plt.close('all')
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(7, 10))
spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0], projection=proj)


ax = [ax1]
names = ['CC', 'COD', 'LWP', 'IWP']
for i in range(1):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax[i].add_feature(feat.LAND, facecolor='#363737')
    # ax[i].add_feature(feat.OCEAN)

    ax[i].gridlines()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax[i].set_boundary(circle, transform=ax[i].transAxes)


# Geodetic always draws shortes distance between two points
plt.plot([140, -40], [-60, -60],
         transform=ccrs.Geodetic(), lw=8, color='#03719c')

fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/cross_section_inset.png',
            format='PNG', dpi=300)
