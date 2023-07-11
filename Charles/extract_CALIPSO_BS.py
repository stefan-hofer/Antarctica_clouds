import h5py
import numpy as np
import xarray as xr
from cmcrameri import cm
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.path as mpath
import glob
import os

# directory = '/projects/NS9600K/shofer/blowing_snow/sat_data/CALIOP_blowing_snow/'
directory = '/home/sh16450/caliop_blowing_snow/'
files = sorted(glob.glob(
    directory + 'CAL_LID_L2_BlowingSnow_Antarctica-Standard-V1-00*.hdf5'))

for file in files:
    f = h5py.File(file, 'r')
    dset = f["Snow_Fields"]

    # extract the blowing snow frequency
    blow_freq = (dset["Snow_Blowing_Frequence_Grid"]
                 [0, :, :] / dset['Observation_Grid']) * 100
    lats = np.arange(-89.5, 90.5, 1)
    lons = np.arange(-180, 180, 1)
    time = file.split("/")[-1].split(".")[-2] + '-01'
    print('The time is: {}'.format(time))

    ds = xr.Dataset({"bs_freq": (['lat', 'lon'], blow_freq)},
                    coords={'lat': (['lat'], lats),
                            'lon': (['lon'], lons),
                            'time': pd.date_range(time, periods=1)})

    save_str = file.split(".")[0] + "." + time + ".nc"
    if os.path.exists(save_str):
        print('The file {} already exists!'.format(save_str))
    else:
        print('Saving the file in {}!'.format(save_str))
        ds.to_netcdf(save_str)


list(f.keys())

dset["Blowing_Snow_Layer_Optical_Depth"].shape  # yields (7445905, 1)

dset_geo = f["Geolocation_Fields"]
anc = f["Ancillary_Fields"]
meta = f['Metadata']
