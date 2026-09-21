import ftplib
import os
from pathlib import Path
import pandas as pd


def name_format(time:str, type:str):
    if type == 'archive':
        return f'ice_conc_nh_ease2-250_cdr-v3p0_{time}1200.nc'
    if type == 'deprec':
        return f'ice_conc_nh_ease2-250_icdr-v3p0_{time}1200.nc'
    if type == 'current':
        return f'ice_conc_nh_ease2-250_icdr-v3p0-amsr_{time}1200.nc'


def download(folder_to_save, file_name, ftp):
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    file = file_name
    file_to_save = file_name

    try:
        if not os.path.exists(f'{folder_to_save}/{file_to_save}'):
            with open(Path(folder_to_save, file_to_save), 'wb') as newfile:
                ftp.retrbinary('RETR %s' % file, newfile.write)
                print(f'{file} downloaded')
        else:
            print(f'{file} already downloaded')
    except Exception as e:
        print(e)
        os.remove(Path(folder_to_save, file_to_save))
        return False
    return True


def download_pack(folder_to_save, start_day, end_day):
    type = 'current' # archive deprec
    dates = pd.date_range(start_day, end_day, freq='1D')
    if type == 'archive':
        dir = '/reprocessed/ice/conc/v3p0'
    if type == 'deprec':
        dir = '/reprocessed/ice/conc-cont-reproc/v3p0'
    if type == 'current':
        dir = '/reprocessed/ice/conc-cont-reproc-amsr/v3p0'


    for time in dates:
        month = time.strftime('%m')
        year = time.strftime('%Y')
        time = time.strftime('%Y%m%d')
        file = name_format(time, type)
        print(file)
        try:
            if not os.path.exists(Path(folder_to_save, file)):
                ftp = ftplib.FTP('osisaf.met.no')
                ftp.login()
                ftp.cwd(f'{dir}/{year}/{month}')
                print(f'{dir}/{year}/{month}')
                is_downloaded = download(folder_to_save, file, ftp)
                ftp.quit()
        except Exception as e:
            print(e)
            pass


if __name__ == '__main__':
    # OSI SAF splits the record into three products, set `type` in download_pack:
    #   'archive' - OSI-450-a, 1979 to 2020
    #   'deprec'  - OSI-430-a, 2021 to October 2025
    #   'current' - OSI-438 (AMSR2), from October 2025 onwards
    folder_to_save = 'path/to/raw/netcdf'
    download_pack(folder_to_save, '20250701', '20260501')
