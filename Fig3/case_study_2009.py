import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature

# file_str = '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/'
file_str = '/home/shofer/Dropbox/Academic/Data/Blowing_snow/'

# Open all the files for Oct 2009
ds_nobs = xr.open_dataset(
    file_str + 'MAR35_nDS_Oct2009.nc')
ds_bs = xr.open_dataset(
    file_str + 'MAR35_DS_Oct2009.nc')
MAR_grid = xr.open_dataset(
    file_str + 'MARcst-AN35km-176x148.cdf')

ds_grid = xr.Dataset({'RIGNOT': (['y', 'x'], MAR_grid.RIGNOT.values),
                      'SH': (['y', 'x'], MAR_grid.SH.values),
                      'LAT': (['y', 'x'], MAR_grid.LAT.values),
                      'LON': (['y', 'x'], MAR_grid.LON.values)},
                     coords={'x': (['x'], MAR_grid.x),
                             'y': (['y'], MAR_grid.y)})
# Difference for 14th of October
diff = ds_bs.sel(TIME='2009-10-14').where((ds_bs.LAT > -90) & (ds_bs.LAT < -65) & (ds_bs.LON < 180) & (ds_bs.LON > 120)) - \
    ds_nobs.sel(TIME='2009-10-14').where((ds_bs.LAT > -90) &
                                         (ds_bs.LAT < -65) & (ds_bs.LON < 180) & (ds_bs.LON > 120))

# Plotting
names = ['MAR_BS', 'MAR_No_Blowing_Snow', 'Difference BS - noBS']
# Compare trends between 2002 and 2015
proj = ccrs.SouthPolarStereo()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(
    12, 12), subplot_kw={'projection': proj})
ax = axs.ravel().tolist()

for i in range(3):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([120, 180, -90, -65], ccrs.PlateCarree())

    ax[i].add_feature(cartopy.feature.LAND)
    ax[i].add_feature(cartopy.feature.OCEAN)

    ax[i].gridlines()

    # # Compute a circle in axes coordinates, which we can use as a boundary
    # # for the map. We can pan/zoom as much as we like - the boundary will be
    # # permanently circular.
    # theta = np.linspace(0, 2*np.pi, 100)
    # center, radius = [0.5, 0.5], 0.5
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    #
    # ax[i].set_boundary(circle, transform=ax[i].transAxes)

# cmap = 'YlGnBu_r'
cont_1 = ax[0].pcolormesh(ds_bs['LON'].values, ds_bs['LAT'].values,
                          diff.CC.isel(TIME=0), transform=ccrs.PlateCarree())

cont_2 = ax[1].pcolormesh(ds_bs['LON'], ds_bs['LAT'],
                          diff.IWP.isel(TIME=0), transform=ccrs.PlateCarree())
cont_3 = ax[2].pcolormesh(ds_bs['LON'], ds_bs['LAT'],
                          diff.CWP.isel(TIME=0), transform=ccrs.PlateCarree())

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
fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2_new.pdf',
            format='PDF')
fig.savefig('/uio/kant/geo-metos-u1/shofer/repos/Antarctica_clouds/Plots/Diff_Climatology_CC_DJF_2002-2015_2x2_new.png',
            format='PNG', dpi=500)
