from torch import nn
from torch.optim import Adam
from black_box import FeedForwardNN
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from welford import WelfordVarianceEstimator

class RND:
    """
        An Intrinsic Curiosity Module that consists of two neural 
        networks: a randomly initialized one, and another one, 
        which is trained to predict the output of the first network.
        The error of prediction is used as a dense intrinsic reward 
        for an RL agent to augment the sparse extrinsic reward.
    """
    def __init__(self, in_shape, init_obs):
        '''
            Parameters:
                in_shape - The shape of an observation
                init_obs - A bunch of observations gathered by running a random agent in the environment to initialize the variance estimator
        '''
        self.lr = 1e-4
        self.target = FeedForwardNN(in_shape,  (32,), (32,32,32))
        self.predictor = FeedForwardNN(in_shape, (32,), (32,32))
        self.predictor_optim = Adam(self.predictor.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.predictor_optim, 0.999)
        self.obs_w = WelfordVarianceEstimator(init_obs)
        init_loss = []
        for obs in init_obs:
            targ = self.target(obs)
            pred = self.predictor(obs)
            loss = nn.MSELoss()(targ, pred)
            init_loss.append(loss.detach().numpy())
        self.loss_w = WelfordVarianceEstimator(init_loss)
        
    def get_reward(self, obs):     
        # Normalize the observation and collect the stats
        self.obs_w.step(obs)
        obs = (obs-self.obs_w.get_mean())/(self.obs_w.get_variance()**0.5 + 1e-10)
        
        # Get the loss
        targ = self.target(obs)
        pred = self.predictor(obs)
        loss = nn.MSELoss()(targ, pred)
        
        # Learn
        self.predictor_optim.zero_grad()
        loss.backward()
        self.predictor_optim.step()
        
        # Collect the stats        
        self.loss_w.step(loss.detach().numpy())

        return loss.detach().numpy()

    def anneal_lr(self):
        self.scheduler.step()
    
    def reset_rew_std(self, rew):
        self.loss_w = WelfordVarianceEstimator(rew)
        
    def get_rew_std(self):
        return self.loss_w.get_variance()**0.5