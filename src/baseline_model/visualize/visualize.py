import matplotlib.pyplot as plt
from baseline_model.logger.logger import Loggger
import seaborn as sns

class Visualizer:
    def __init__(self, logger:Loggger ) -> None:
        self.logger = logger

    def visualize_loss(self):
        plt.plot(range(len(self.logger.train_loss)), self.logger.train_loss,  marker='o', markersize=10, markerfacecolor='blue', markeredgecolor='black', markeredgewidth=1.5)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
    
    def visualuze_boxplot(self, stage = 'eval'):
        if 'eval':
            psnr = self.logger.psnr_eval
            ssim = self.logger.ssim_eval
            mae = self.logger.eval_loss
        else:
            psnr = self.logger.psnr_test
            ssim = self.logger.ssim_test
            mae = self.logger.mae_test

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.boxplot(y=psnr, color='blue', ax=ax[0])
        ax[0].set_title('PSNR metric')
        ax[0].set_ylabel('PSNR') 

        sns.boxplot(y=ssim, color='blue', ax=ax[1])
        ax[1].set_title('SSIM metric')
        ax[1].set_ylabel('SSIM') 

        sns.boxplot(y=mae, color='blue', ax=ax[2])
        ax[2].set_title('MAE metric')
        ax[2].set_ylabel('MAE') 

        plt.tight_layout() 
        plt.show()
    