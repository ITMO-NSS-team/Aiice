from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Loggger:
    """
    A dataclass for logging various metrics during model training, evaluation, and testing.

    This class provides a structured way to store lists of loss values and performance
    metrics such as PSNR, SSIM, and MAE for different stages of model development.
    It uses properties for controlled access and appending new metric values.

    Attributes:
        _train_loss (List[float]): Stores training loss values. Each new value is appended.
        _recalculated_train_loss (bool): Flag indicating if training loss has been recalculated.
        _eval_loss (List[float]): Stores evaluation loss values. Each new value is appended.
        _recalculated_eval_loss (bool): Flag indicating if evaluation loss has been recalculated.
        _psnr_eval (List[float]): Stores PSNR metric values from evaluation.
        _ssim_eval (List[float]): Stores SSIM metric values from evaluation.
        _psnr_test (List[float]): Stores PSNR metric values from testing.
        _ssim_test (List[float]): Stores SSIM metric values from testing.
        _mae_test (List[float]): Stores MAE metric values from testing.
    """
    _train_loss: List[float] = field(
        default_factory=list, 
        metadata={'description' : 'logging all losses inside values from dict, key is num of epoch'}
        )
    
    _recalculated_train_loss: bool  = field(
        default=False
    )
    
    _eval_loss: List[float] = field(
        default_factory=list,
        metadata={'description' : 'collecting all values from evaluating model'}
    ) 
    _recalculated_eval_loss: bool = field(
        default=False
    )

    _psnr_eval: List[float] = field(
        default_factory=list,
        metadata={'description': 'collecting psnr metric for eval'}
    )

    _ssim_eval: List[float] = field(
        default_factory=list,
        metadata={'description': 'collecting ssim metric for eval'}
    )
    _psnr_test: List[float] = field(
        default_factory=list,
        metadata={'desacription': 'collecting psnr metric for test'}
    )

    _ssim_test: List[float] = field(
        default_factory=list,
        metadata={'description': 'collecting ssim metric for test'}
    )

    _mae_test: List[float] = field(
        default_factory=list,
        metadata={'description':'MAE metric for test data'}
    )
    
    @property
    def train_loss(self):
        """
        Gets the list of training loss values.
        """
        return self._train_loss

    @train_loss.setter
    def train_loss(self, val:float):
       """
        Appends a new training loss value to the list.

        Args:
            val (float): The training loss value to add.
        """
       self._train_loss.append(val)

    @property
    def eval_loss(self):
        """
        Gets the list of evaluation loss values.
        """
        return self._eval_loss
    
    @eval_loss.setter
    def eval_loss(self, val:float):
        """
        Appends a new evaluation loss value to the list.

        Args:
            val (float): The evaluation loss value to add.
        """
        self._eval_loss.append(val)

    
    @property
    def ssim_eval(self):
        """
        Gets the list of SSIM metric values from evaluation.
        """
        return self._ssim_eval
    
    @property
    def psnr_eval(self):
        """
        Gets the list of PSNR metric values from evaluation.
        """
        return self._psnr_eval
    
    @property
    def ssim_test(self):
        """
        Gets the list of SSIM metric values from testing.
        """
        return self._ssim_test
    
    @property
    def psnr_test(self):
        """
        Gets the list of PSNR metric values from testing.
        """
        return self._psnr_test
    
    @property
    def mae_test(self):
        """
        Gets the list of MAE metric values from testing.
        """
        return self._mae_test
    
    @ssim_eval.setter
    def ssim_eval(self, val:float):
        """
        Appends a new SSIM metric value for evaluation.

        Args:
            val (float): The SSIM value to add.
        """
        self._ssim_eval.append(val)

    @psnr_eval.setter
    def psnr_eval(self, val:float):
        """
        Appends a new PSNR metric value for evaluation.

        Args:
            val (float): The PSNR value to add.
        """
        self._psnr_eval.append(val)

    @ssim_test.setter
    def ssim_test(self, val:float):
        """
        Appends a new SSIM metric value for testing.

        Args:
            val (float): The SSIM value to add.
        """
        self._ssim_test.append(val)

    @psnr_test.setter
    def psnr_test(self, val:float):
        """
        Appends a new PSNR metric value for testing.

        Args:
            val (float): The PSNR value to add.
        """
        self._psnr_test.append(val)

    @mae_test.setter 
    def mae_test(self, val:float):
        """
        Appends a new MAE metric value for testing.

        Args:
            val (float): The MAE value to add.
        """
        self._mae_test.append(val)


