import seaborn as sns
import xarray as xr

ds_nobs = xr.open_dataset(
    '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/MAR35_nDS_Oct2009.nc')
ds_bs = xr.open_dataset(
    '/uio/lagringshotell/geofag/projects/miphclac/shofer/MAR/case_study_BS_2009/MAR35_DS_Oct2009.nc')
