"""
    This file contains several neural network architectures 
    for us to define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    """
        A standard in_dim-dim_1-...dim_n-out_dim 
        Fully Connected Feed Forward Neural Network.
    """
    def __init__(self, in_shape, out_shape, hidden_shape = (64,64), out_activation=None):
        """
            Initialize the network and set up the layers.
            Parameters:
                in_shape - input dimensions as a tuple of ints
                out_shape - output dimensions as a tuple of ints
                hidden_shape - shapes of hidden layers as a tuple of ints
                out_activation - activation of the output layer as a function(output)
            Return:
                None
        """

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # Flatten the ingoing and outgoing tensors if they are multidimensional
        in_dim = np.prod(in_shape)
        out_dim = np.prod(out_shape)
        
        self.in_layer = nn.Linear(in_dim, hidden_shape[0])
        self.hidden_layers = []
        for dim in range(len(hidden_shape)-1):
            self.hidden_layers.append(nn.Linear(hidden_shape[dim], hidden_shape[dim+1]))
        self.out_layer = nn.Linear(hidden_shape[-1], out_dim)
        self.out_activation = out_activation

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.
            Parameters:
                obs - observation to pass as input
            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        if len(self.in_shape) != 1:
            obs = torch.flatten(obs)
        activation = F.relu(self.in_layer(obs))
        for layer in self.hidden_layers:
            activation = F.relu(layer(activation))
        output = self.out_layer(activation)
        if self.out_activation:
            output = self.out_activation(output)
        if len(self.out_shape) != 1:
            output = torch.reshape(output, self.out_shape)

        return output
    
def black_box(box_type, **inner_workings):
    """
        A bit of an experiment in fitting different interfaces together.
        It basically predefines everything apart from the shapes of the input and output.
        Parameters:
            box_type - the class that's going to be the brain inside the black box
            inner_workings - all of the hidden hyperparameters
        Returns a class that has two methods:
            __init__(self, in_shape, out_shape), 
            and forward(self, obs) that eats a tensor with 
            in_shape and spits out a tensor with out_shape
    """
    class BlackBox(nn.Module):
        def __init__(self, in_shape, out_shape):
            super().__init__()
            self.brain = box_type(in_shape, out_shape, **inner_workings)
        
        def forward(self, obs):
            return self.brain.forward(obs)
    
    return BlackBox