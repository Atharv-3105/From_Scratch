import random 
import numpy as np 
import torch

def set_seed(seed: int) -> None:
    """ 
        A defined seed for all the random number generator which will
        be used throughout this project
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #Lock cuDNN to determinisitic alogrithms 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  