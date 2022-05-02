import h5py
import numpy as np
import xarray as xr
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib.path as mpath

f = h5py.File(
    'CAL_LID_L2_BlowingSnow_Antarctica-Standard-V1-00.2015-02.hdf5', 'r')


list(f.keys())
dset = f["Snow_Fields"]
dset["Blowing_Snow_Layer_Optical_Depth"].shape  # yields (7445905, 1)

dset_geo = f["Geolocation_Fields"]
anc = f["Ancillary_Fields"]
meta = f['Metadata']

lats = np.arange(-89.5, 90.5, 1)
lons = np.arange(-180, 180, 1)


# extract the blowing snow frequency
blow_freq = (dset["Snow_Blowing_Frequence_Grid"]
             [0, :, :] / dset['Observation_Grid']) * 100

ds = xr.Dataset({"bs_freq": (['lat', 'lon'], blow_freq)},
                coords={'lat': (['lat'], lats),
                        'lon': (['lon'], lons),
                        'time': pd.date_range("2015-02-01", periods=1)})

# PLOT THE MAP
plt.close('all')
proj = ccrs.SouthPolarStereo()
fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={
    'projection': proj}, figsize=(7, 10))
# fig = plt.figure(figsize=(8, 8))
# spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
# ax1 = fig.add_subplot(spec2[0, 0], projection=proj)

ax = [axs]
for i in range(1):
    # Limit the map to -60 degrees latitude and below.
    ax[i].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax[i].add_feature(feat.LAND)
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

ds['bs_freq'].plot.pcolormesh(
    'lon', 'lat', transform=ccrs.PlateCarree(), ax=ax[0], cmap=cm.bilbao)
axs.add_feature(feat.COASTLINE)

# cb = fig.colorbar(cont, ax=ax[0], ticks=list(
#     np.arange(-30, 35, 10)), shrink=0.8, orientation='horizontal')
# cb.set_label(r'$\Delta$ Snow Content (g/kg)', fontsize=16)

fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map.png',
            format='PNG', dpi=300)
fig.savefig('/projects/NS9600K/shofer/blowing_snow/station_map.pdf',
            format='PDF')

Snow_Blow_Frequency_Grid
Units: count
Format: Float_32
Fill Value:
    Valid Range: N / A
    Definition: 360x180x9 grid
    Contains the monthly blowing snow layer occurrence in each 1x1 degree grid
    Blow_Snow_Frequency_Grid(0) = occurrence of blowing snow layers
    Blow_Snow_Frequency_Grid(1) = occurrence of blowing snow layers where the backscatter ratio of the top altitude bin within layer to first bin above ground < 0.5
    Blow_Snow_Frequency_Grid(2) = occurrence of blowing snow layers where the ratio of average backscatter strength within blowing snow layer to the blowing snow threshold > 1.5 and layer depth > 50 m
    Blow_Snow_Frequency_Grid(3) = occurrence of blowing snow layers where the 10 m wind speed > 6 m / s and layer depth > 50 m
    Blow_Snow_Frequency_Grid(4) = occurrence of blowing snow layers where the surface elevation(as determined from DEM) > 0.0 and 10 m wind speed > 7 m / s
    Blow_Snow_Frequency_Grid(5) = occurrence of blowing snow layers where the layer depth > 30m and < 350 m
    Blow_Snow_Frequency_Grid(6) = occurrence of blowing snow layers where the average layer 532 depolarization > 0.25 and layer average color ratio > 0.8
    Blow_Snow_Frequency_Grid(7) = occurrence of blowing snow layers where one of the following conditions is true(as stated in (1), (4), & (6)) the backscatter ratio of the top altitude bin within layer to first bin above ground < 0.5 OR the surface elevation(as determined from DEM) > 0.0 and 10 m wind speed > 7 m / s OR the average layer 532 depolarization > 0.25 and layer average color ratio > 0.8
    Blow_Snow_Frequency_Grid(8) = occurrence of the Blowing Snow Confidence Flag > 4


# lat = dset_geo["Latitude"][:,0]
# lon = dset_geo["Longitude"][:,0]
