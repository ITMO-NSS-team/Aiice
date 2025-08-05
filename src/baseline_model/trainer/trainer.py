import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
from baseline_model.visualize.visualize import Visualizer
from baseline_model.logger.logger import Loggger
import logging

logging.basicConfig(level=logging.INFO) 

class Trainer:
    """
    A comprehensive trainer class for PyTorch models, encapsulating training,
    evaluation, and testing workflows.

    This class provides functionalities to train a given PyTorch model,
    evaluate its performance on a validation set using common image metrics
    (PSNR, SSIM, MAE), and test its final performance. It handles device
    management (CPU/GPU), default optimizer and loss function, and progress
    visualization.
    """
    def __init__(self, model:nn.Module) -> None:
        """
        Initializes the Trainer with a specified PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = Loggger()
        self.optimizers = {
            'Adam': optim.Adam,
            'AdamW': optim.AdamW
        }
        self.visualizer = Visualizer(self.logger)
        self.ssim = StructuralSimilarityIndexMeasure()
    
    def train(self, dataloader_train:DataLoader, num_epoch:int = 100, lr:float = 1e-3, optimizer: str = None, loss_function: nn.Module = None, make_plot:bool=False)->None:
        """
        Trains the model using the provided training DataLoader.

        Args:
            dataloader_train (DataLoader): The DataLoader containing the training data.
            num_epoch (int, optional): The number of training epochs. Default is 100.
            lr (float, optional): The learning rate for the optimizer. Default is 1e-3.
            optimizer (Optimizer, optional): The optimizer to use for training. If None,
                                             AdamW with the specified learning rate is used.
            loss_function (nn.Module, optional): The loss function to use. If None,
                                                 L1Loss (MAE) is used.

        Returns:
            nn.Module: The trained model.
        """
        if optimizer is None:
            optimizer = self.optimizers['Adam'](self.model.parameters(), lr = lr)
        else:
            optimizer = self.optimizers[optimizer](self.model.parameters(), lr=lr)
        
        if loss_function is None:
            loss_function = F.l1_loss
        
        self.model = self.model.to(self.device)
        self.model = self.model.train()

        for epoch in tqdm(range(num_epoch)):    
            losses = list()
            for data in tqdm(dataloader_train):
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.to(torch.float32), y.to(torch.float32)
                preds = self.model(x)
                loss = loss_function(preds, y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.train_loss = np.array(losses).mean().item()
        
        # logging.info(f'Object custom logger inside {self.logger.train_loss}')
        
        if make_plot:
            self.visualizer.visualize_loss()
        
        return self.model


    def evaluate(self, dataloader_val:DataLoader, make_plot:bool = False):
        """
        Evaluates the model on the provided validation DataLoader.

        Calculates and plots PSNR, SSIM, and MAE metrics for the validation set.

        Args:
            dataloader_val (DataLoader): The DataLoader containing the validation data.
        """
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        predictions = list()
        ground_truth = list()

        with torch.no_grad():
            for data in dataloader_val:
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.to(torch.float32), y.to(torch.float32)
                preds = self.model(x)
                predictions.append(preds)
                ground_truth.append(y)
        
        self._calc_metrics(predictions, ground_truth)
       
        logging.info(f'Eval logs ssim {self.logger.ssim_eval}')
        logging.info(f'Eval logs psnr {self.logger.psnr_eval}')
        logging.info(f'Eval logs MAE {self.logger.eval_loss}')
        if make_plot:
            self.visualizer.visualuze_boxplot()

    def test(self, dataloader_test:DataLoader, make_plot:bool = False)->None:
        """
        Tests the model on the provided test DataLoader and prints the average metrics.

        Calculates and prints the mean PSNR, SSIM, and MAE over the entire test set.

        Args:
            dataloader_test (DataLoader): The DataLoader containing the test data.
        """
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        predictions = list()
        ground_truth = list()
        
        with torch.no_grad():
            for data in dataloader_test:
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.to(torch.float32), y.to(torch.float32)
                preds = self.model(x)
                predictions.append(preds)
                ground_truth.append(y)
        
        self._calc_metrics(preds=predictions, ground_truth=ground_truth, stage='test')

        
        logging.info(f'Test logs ssim {self.logger.ssim_test}')
        logging.info(f'Test logs psnr {self.logger.psnr_test}')
        logging.info(f'Test logs MAE {self.logger.mae_test}')

        
        print(f'SSIM metric is {np.array(self.logger.ssim_test).mean().item()}')
        print('-'*10)
        print(f'PSNR metric is {np.array(self.logger.psnr_test).mean().item()}')
        print('-' * 10)
        print(f'MAE metric is {np.array(self.logger.mae_test).mean().item()}')

        if make_plot:
            self.visualizer.visualuze_boxplot(stage='test')


    def _calc_psnr(self, pred:torch.Tensor, label:torch.Tensor, scaled:bool = True, max_val: int = None)->float:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) between predicted and label tensors.

        Args:
            pred (torch.Tensor): The predicted tensor.
            label (torch.Tensor): The ground truth label tensor.
            scaled (bool, optional): If True, assumes pixel values are scaled to [0, 1]
                                     and uses a max_val of 1. Otherwise, assumes
                                     a max_val of 100. Defaults to True.

        Returns:
            float: The calculated PSNR value.
        """
        if scaled:
            max_val = 1
        else:
            max_val = 100
        mse = F.mse_loss(pred, label)
        return 20*torch.log10(max_val / torch.sqrt(mse)).item()
    

    def _calc_metrics(self, preds:list, ground_truth:list, stage:str = 'eval'):
        for index in range(len(preds)):
            current_preds = torch.squeeze(preds[index])
            current_label = torch.squeeze(ground_truth[index])
            
            for index_im in range(len(current_label)):
                if stage == 'eval':
                    self.logger.psnr_eval = self._calc_psnr(current_preds[index_im], current_label[index_im])
                    self.logger.ssim_eval = self.ssim(current_preds[index_im].unsqueeze(0).unsqueeze(0), 
                                                      current_label[index_im].unsqueeze(0).unsqueeze(0)).item()
                    self.logger.eval_loss = F.l1_loss(current_preds[index_im], current_label[index_im]).item()
                else:
                    self.logger.psnr_test = self._calc_psnr(current_preds[index_im], current_label[index_im])
                    self.logger.ssim_test = self.ssim(current_preds[index_im].unsqueeze(0).unsqueeze(0), 
                                                      current_label[index_im].unsqueeze(0).unsqueeze(0)).item()
                    self.logger.mae_test = F.l1_loss(current_preds[index_im], current_label[index_im]).item()
            
