import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from black_box import FeedForwardNN


class RND(nn.Module):
    """
    An Intrinsic Curiosity Module that consists of two neural 
    networks: a randomly initialized one, and another one, 
    which is trained to predict the output of the first network.
    The error of prediction is used as a dense intrinsic reward 
    for an RL agent to augment the sparse extrinsic reward.
    """
    def __init__(self, in_dim, out_dim):
        self.target = FeedForwardNN(in_dim, out_dim)
        self.predictor = FeedForwardNN(in_dim, out_dim)
    
    def get_reward():
        pass