import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units


def thetae_mar(ds):
    '''Converts dataset, containing surface pressure, temperature
    and relative humidity, to equivalent potential
    temperature. Provide only one (surface) layer.
    =============================================
    Usage:
    thetae = thetae_mar(ds_rh)
    '''
    ds = ds.sel(ATMLAY=1, method='nearest')
    # Use correct CF unit for conversion to MetPy
    ds['TT'].attrs['units'] = 'degC'
    # Convert to MetPy dataset
    ds_mpy = ds.metpy.parse_cf()
    # Compute dewpoint from RH and TT
    ds_td = mpcalc.dewpoint_from_relative_humidity(ds_mpy.TT, ds_mpy.RH)
    # Compute ThetaE from TT, TD and SP
    thetae = mpcalc.equivalent_potential_temperature(ds_mpy.SP,
                                                     ds_mpy.TT, ds_td)

    return thetae
