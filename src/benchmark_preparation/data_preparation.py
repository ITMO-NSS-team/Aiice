import datetime
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def save_osisaf():
    nc_path = 'D:/Aiice'
    save_path = 'D:/Aiice/osisaf'
    for file in os.listdir(nc_path):
        date = datetime.datetime.strptime(file.split('_')[-1], '%Y%m%d1200.nc')
        new_file_name = f'osisaf_{date.strftime("%Y%m%d")}.npy'
        ds = nc.Dataset(f'{nc_path}/{file}')
        matrix = np.array(ds.variables['ice_conc'])[0]
        matrix[matrix < 0] = 0
        matrix = matrix.astype(int)
        #plt.imshow(matrix)
        #plt.colorbar()
        #plt.show()
        np.save(f'{save_path}/{new_file_name}', matrix)

def correct_missings_files_with_interpolation(matrices_path, times, format):
    for i, time in enumerate(times):
        print(time)
        files_list = os.listdir(matrices_path)
        exts = False
        for file_name in files_list:
            if time.strftime(format) in file_name:
                exts = True
                break
        if exts == False:
            print(f'{time} missed')
            prev_matrix = np.load(f'{matrices_path}/{times[i - 1].strftime(format)}')
            miss_num = 1
            next_matrix = None
            while next_matrix is None:
                try:
                    next_matrix = np.load(f'{matrices_path}/{times[i+miss_num].strftime(format)}')
                except Exception as e:
                    miss_num = miss_num+1
            print(f'Interpolate between {times[i-1]} and {times[i + miss_num]} for {miss_num} steps')
            missing_matrix = np.linspace(prev_matrix, next_matrix, miss_num+2).astype(int)
            for m in range(missing_matrix.shape[0]-2):
                print(f'Save {times[i + m]}')
                np.save(f'{matrices_path}/{times[i + m].strftime(format)}', missing_matrix[m+1])

def correct_osisaf_gaps():
    matrices_path = 'D:/Aiice/osisaf'
    start_date = os.listdir(f'{matrices_path}')[0][-12:-4]
    end_date = os.listdir(f'{matrices_path}')[-1][-12:-4]
    dateRange = pd.date_range(start_date, end_date)
    correct_missings_files_with_interpolation(f'{matrices_path}', dateRange, 'osisaf_%Y%m%d.npy')


def land_mask_save():
    path = 'D:/Aiice/oceanmask_nh_ease_as_osisaf.nc'
    ds = nc.Dataset(path)
    mask = np.array(ds.variables['oceanmask'])[0]
    mask[mask < 0] = 2
    mask[mask <= 1] = 0
    mask[mask != 0] = 1
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    np.save('land_mask.npy', mask)

land_mask_save()