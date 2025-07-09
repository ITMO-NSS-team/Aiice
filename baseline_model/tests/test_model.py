from model.model_unet import CustomUnet
from trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch

class MakeTimeSeriesPerWeek:
    """
    Processes daily ice concentration data into weekly averaged time series for model input
    
    Args:
        path (str): Path to directory containing OSISAF daily ice concentration files
        
    Attributes:
        x_train, y_train (dict): Training features and targets
        x_val, y_val (dict): Validation features and targets  
        x_test, y_test (dict): Test features and targets
        train, val, test (list): Raw weekly data splits
        
    Methods:
        _get_day: Extracts day from filename
        _get_year: Extracts year from filename
        _get_month: Extracts month from filename
        _normalize: Normalizes array by max value
        _base_normalize: Normalizes array by fixed base value (100)
        _form_data: Creates weekly average from 7 daily images
        _collect_average_info: Organizes data into weekly chunks
        _prepare_data: Splits weekly chunks into features/targets
        load: Main method to process and prepare all data
    """
    def __init__(self, path:str):
        self.path = Path(path)
        self.x_train = dict()
        self.y_train = dict()
        self.x_val = dict()
        self.y_val = dict()
        self.x_test = dict()
        self.y_test = dict()
        self.train = list()
        self.val = list()
        self.test = list()
    
    def _get_day(self, filename:str):
        """Extracts day from OSISAF filename (format: ice_YYYYMMDD.npy)"""
        return int(filename.split('_')[1].split('.')[0][-2:])

    def _get_year(self, filename:str):
        """Extracts year from OSISAF filename"""
        return int(filename.split('_')[1].split('.')[0][:4])
    
    def _get_month(self, filename:str):
        """Extracts month from OSISAF filename"""
        return int(filename.split('_')[1].split('.')[0][4:6])
    
    def _normalize(self, ar:np.array):
        """Normalizes array by its maximum value"""
        return ar / ar.max().item()
    
    def _base_normalize(self, arr:np.array):
        """Normalizes ice concentration values by base value 100"""
        return arr / 100
    
    def _form_data(self, inp_path:list):
        """
        Creates weekly average from 7 daily images
        
        Args:
            inp_path (list): List of 7 daily file paths
            
        Returns:
            np.array: Weekly averaged ice concentration
        """
        sum_im = None
        for im in inp_path:
            current = np.load(im)
            current = self._base_normalize(current)
            if sum_im is None:
                sum_im = current
            else:
                sum_im += current
        return sum_im / 7
    
    def _collect_average_info(self):
        """Organizes daily data into weekly chunks of 72 timesteps"""
        feat_and_target = list()
        current = list()
        week = list()
        for file in tqdm(sorted(os.listdir(self.path))):
            day = self._get_day(file)
            month = self._get_month(file)
            if day > 28 and month == 2:
                continue
            else:
                week.append(self.path / Path(file))
                if len(week) == 7:
                    val = torch.tensor(self._form_data(week))
                    current.append(val)
                    week = list()
                    if len(current) == 72:
                        feat_and_target.append(current)
                        current = list()
        self.train = feat_and_target[:-2]
        self.val = feat_and_target[-2:-1]
        self.test = [feat_and_target[-1]]

    def _prepare_data(self, inp:list, feat:dict, target:dict):
        """
        Splits weekly chunks into features (first 48 timesteps) and targets (last 24 timesteps)
        
        Args:
            inp (list): List of weekly data chunks
            feat (dict): Dictionary to store features
            target (dict): Dictionary to store targets
        """
        for val in tqdm(inp):
            feat[len(feat.keys())] = torch.stack(val[:48])
            target[len(target.keys())] = torch.stack(val[48:])

    def load(self):
        """Main method to process and prepare all data splits"""
        self._collect_average_info()
        self._prepare_data(self.train, self.x_train, self.y_train)
        self._prepare_data(self.val, self.x_val, self.y_val)
        self._prepare_data(self.test, self.x_test, self.y_test)


class IceDataset(Dataset):
    """
    PyTorch Dataset class for ice concentration data
    
    Args:
        x (dict): Dictionary of input features
        y (dict): Dictionary of target values
        
    Methods:
        __len__: Returns number of samples
        __getitem__: Returns (features, target) tuple for given index
    """
    def __init__(self, x:dict, y:dict) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x.keys())
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

temp_data = MakeTimeSeriesPerWeek(r'D:\Arctic_project\dataset\osisaf')
temp_data.load()
dataset_train = IceDataset(temp_data.x_train, temp_data.y_train)
dataset_val = IceDataset(temp_data.x_val, temp_data.y_val)
dataset_test = IceDataset(temp_data.x_test, temp_data.y_test)

dataloader_train = DataLoader(dataset_train, batch_size=8)
dataloader_val = DataLoader(dataset_val, batch_size=8)
dataloader_test = DataLoader(dataset_test, batch_size=8)

model = CustomUnet(num_layers=5, channels_in=48, channels_out=24, image_size=432)
trainer = Trainer(model=model)
model = trainer.train(dataloader_train, num_epoch=1)
trainer.evaluate(dataloader_val)
trainer.test(dataloader_test)
