import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from black_box import FeedForwardNN
from welford import WelfordVarianceEstimator


class RND:
    """
    An Intrinsic Curiosity Module that consists of two neural 
    networks: a randomly initialized one, and another one, 
    which is trained to predict the output of the first network.
    The error of prediction is used as a dense intrinsic reward 
    for an RL agent to augment the sparse extrinsic reward.
    """
    def __init__(self, in_shape):
        self.lr = 3e-3
        self.target = FeedForwardNN(in_shape,  (32,), (64,128,64))
        self.predictor = FeedForwardNN(in_shape, (32,), (64,64))
        self.predictor_optim = Adam(self.predictor.parameters(), lr=self.lr)
        self.welford = WelfordVarianceEstimator((0,0))

    def get_reward(self, obs):
        targ = self.target(obs)
        pred = self.predictor(obs)
        loss = nn.MSELoss()(targ, pred)
        self.welford.step(loss.detach())
        self.predictor_optim.zero_grad()
        loss.backward()
        self.predictor_optim.step()
        return loss.detach()/self.welford.get_variance()**0.5