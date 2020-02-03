import cdsapi

c = cdsapi.Client()

for year in range(1979, 2020):

    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'format': 'netcdf',
            'variable': [
                'geopotential', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'vertical_velocity',
            ],
            'product_type': 'monthly_averaged_reanalysis',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
            'pressure_level': [
                '300', '500', '850',
                '925',
            ],
            'year': [
                str(year),
            ],
        },
        'ERA5_' + str(year) + '_plevs_mthly.nc')
