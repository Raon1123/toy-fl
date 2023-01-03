import numpy as np

import torch
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class GPR(torch.nn.Module):
    def __init__(self) -> None:
        super(GPR, self).__init__()

    def predict_loss(self):
        pass

    def select_clients(self):
        pass

    def update_loss(self):
        pass

    def update_discount(self):
        pass

    def Reset_Discount(self):
        pass

    