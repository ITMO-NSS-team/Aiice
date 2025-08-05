import datetime
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import netCDF4 as nc
import pandas as pd

def init_seas_mask():
    im = Image.open('mask_data/arctic_seas_6931.tif')
    imarray = np.array(im)
    #np.save('mask_data/seas_mask_IHO.npy', imarray)
    imarray[imarray == 999] = None
    plt.imshow(imarray)
    plt.show()

'''def init_osisaf():
    path = 'D:/Aiice'

    dates = []
    matrices = []

    for file in os.listdir(path):
        date = datetime.datetime.strptime(file.split('_')[-1], '%Y%m%d1200.nc')
        dates.append(date)
        ds = nc.Dataset(f'{path}/{file}')
        matrix = np.array(ds.variables['ice_conc'])[0]
        matrix[matrix < 0] = 0
        #plt.imshow(matrix)
        #plt.show()
        matrices.append(matrix)

    return dates, matrices'''


def climatological(target_year):
    path = 'D:/Aiice/osisaf'
    start = 1979
    #start = target_year-5
    end = target_year
    years = np.arange(start, end)

    '''years_pack = []
    for year in years:
        dates = pd.date_range(f'{year}0101', f'{year}1231', freq='1D')
        days_pack = []
        for date in dates:
            print(date)
            file_name = date.strftime(f'osisaf_%Y%m%d.npy')
            matrix = np.load(f'{path}/{file_name}')
            days_pack.append(matrix)
        years_pack.append(days_pack)

    for i in range(len(years_pack)):
       years_pack[i] = years_pack[i][:365]

    years_pack = np.array(years_pack)
    climatology = np.mean(years_pack, axis=0)
    np.save('climatology.npy', climatology)'''

    climatology = np.load('climatology5.npy')

    forecast_dates = pd.date_range(f'{end}0101', f'{end}1231', freq='1D')
    landmask = np.load('land_mask.npy')
    land = deepcopy(landmask).astype(float)
    land[land != 1] = None
    plt.rcParams['figure.figsize'] = (6, 5)

    for d in range(len(forecast_dates)):
        print(forecast_dates[d].strftime('Climatology5 for %Y%m%d'))
        plt.imshow(climatology[d], cmap='Blues_r')
        plt.title(forecast_dates[d].strftime('Climatology5 for %Y%m%d'))
        c = plt.colorbar()
        c.set_label('Ice concentration, %')
        plt.imshow(land, cmap='Pastel2_r')
        plt.contour(landmask, [0], colors='black', linewidths=0.8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'D:/Aiice/climatology5/2012/{forecast_dates[d].strftime("clim5_%Y%m%d")}.png')
        plt.close()


def init_osisaf(start, end):
    period = pd.date_range(start, end, freq='1D')
    files_names = [f.strftime('osisaf_%Y%m%d.npy') for f in period]
    path = 'D:/Aiice/osisaf'

    landmask = np.load('land_mask.npy')
    land = deepcopy(landmask).astype(float)
    land[land != 1] = None
    plt.rcParams['figure.figsize'] = (6, 5)

    for d in range(len(files_names)):
        matrix = np.load(f'{path}/{files_names[d]}')
        print(period[d].strftime('%Y%m%d'))
        plt.imshow(matrix, cmap='Blues_r')
        plt.title(period[d].strftime('OSI-SAF for %Y%m%d'))
        c = plt.colorbar()
        c.set_label('Ice concentration, %')
        plt.imshow(land, cmap='Pastel2_r')
        plt.contour(landmask, [0], colors='black', linewidths=0.8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'D:/Aiice/osisaf_vis/2012/{period[d].strftime("osisaf_%Y%m%d")}.png')
        plt.close()


climatological(2012)
#init_osisaf('20120101', '20121230')

'''c = np.load('climatology.npy')
c5 = np.load('climatology5.npy')

plt.imshow(c[150], cmap='Blues_r')
plt.show()
plt.imshow(c5[150], cmap='Blues_r')
plt.show()'''

'''icenet_df = nc.Dataset('D:/ice_sources/icenet/icenet_sip_forecasts_tempscaled.nc')
vars = icenet_df.variables
time = np.array(vars['time'])

timesteps = [datetime.datetime(2012, 1, 1)+ datetime.timedelta(days=int(d)) for d in time]

plt.imshow(icenet_df.variables['__xarray_dataarray_variable__'][0, :, :, 0])
plt.show()

for t in range(6, len(timesteps)):'''

