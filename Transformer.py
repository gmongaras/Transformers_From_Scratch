import torch
from torch import nn
from torch import optim
import numpy as np




class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        
        # The word embedding layer
        