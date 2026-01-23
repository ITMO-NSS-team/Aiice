import ftplib
import os
from pathlib import Path
import pandas as pd


def name_format(time:str, archive=True):
    if archive:
        return f'ice_conc_nh_ease2-250_cdr-v3p0_{time}1200.nc'
    else:
        return None

def download(folder_to_save, file_name, ftp,  silent=False):
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
        if not silent:
            print(e)
        os.remove(Path(folder_to_save, file_to_save))
        return False
    return True


def download_pack(folder_to_save, start_day, end_day):
    archive = True
    dates = pd.date_range(start_day, end_day, freq='1D')
    if archive:
        dir = '/reprocessed/ice/conc/v3p0'
    else:
        dir = 'reprocessed/ice/conc-cont-reproc/v3p0'


    for time in dates:
        month = time.strftime('%m')
        year = time.strftime('%Y')
        time = time.strftime('%Y%m%d')
        file = name_format(time, archive)
        try:
            if not os.path.exists(Path(folder_to_save, file)):
                ftp = ftplib.FTP('osisaf.met.no')
                ftp.login()
                ftp.cwd(f'{dir}/{year}/{month}')
                print(time)
                is_downloaded = download(folder_to_save, time, ftp, silent=True)
                ftp.quit()
        except Exception as e:
            print(e)
            pass


folder_to_save = 'D:/Aiice'
download_pack(folder_to_save, '19790101', '20201231')