import cdsapi

c = cdsapi.Client()
for year in range(1979, 2021):
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'variable': [
                '2m_dewpoint_temperature', '2m_temperature',
                # '10m_u_component_of_wind', '10m_v_component_of_wind', 'high_cloud_cover',
                # 'low_cloud_cover', 'mean_sea_level_pressure', 'mean_surface_downward_long_wave_radiation_flux',
                # 'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux',
                # 'medium_cloud_cover', 'total_cloud_cover', 'total_column_cloud_ice_water',
                # 'total_column_cloud_liquid_water',
            ],
            'product_type': 'monthly_averaged_reanalysis',
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
            # North, West, South, East. Default: global
            'area': [-45, -180, -90, 180],
        },
        'ERA5_' + str(year) + '_TT_TD_singlelevs_mthly.nc')
