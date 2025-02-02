"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch
import torch.nn
import wandb

from arguments import get_args
from ppo import PPO
import black_box
from eval_policy import eval_policy

import os

def train(env, hyperparameters, actor_model, critic_model):
    """
        Trains the model.
        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training
        Return:
            None
    """
    
    print("Training", flush=True)

    wandb_run = wandb.init(project='ngu',entity='bigfoot', config = hyperparameters)
    wandb.config.update({'env':env})

    # Extract environment information
    obs_shape = env.observation_space.shape

    if type(env.action_space) == gym.spaces.Box:
        act_type = 'box'
        act_shape = env.action_space.shape
    
    if type(env.action_space) == gym.spaces.Discrete:
        act_type = 'discrete'
        # For discrete spaces the action is a softmax vector of probabilities to take each action
        act_shape = (env.action_space.n,)
    
    # Create a model for PPO.
    if act_type == 'box':
        actor = black_box.FeedForwardNN(in_shape=obs_shape, out_shape=act_shape, hidden_shape=(64,64), out_activation=None)
    if act_type == 'discrete':
        actor = black_box.FeedForwardNN(in_shape=obs_shape, out_shape=act_shape, hidden_shape=(64,64), out_activation=torch.nn.Softmax())
    critic = black_box.FeedForwardNN(in_shape=obs_shape, out_shape=(1,), hidden_shape=(64,64), out_activation=None)
    model = PPO(actor, critic, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print("Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print("Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print("Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print("Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=1_000_000)
    wandb_run.finish()

    
# !!!!!!!!!!!!!Works like shit or not at all right now!!!!!!!!!!!!!!!!
def test(env, actor_model, ep):
    """
        Tests the model.
        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in
        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Build our policy the same way we build our actor model in PPO
    policy = black_box.FeedForwardNN(obs_shape, act_shape)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, ep=ep, render=True)

def main(args):
    """
        The main function to run.
        Parameters:
            args - the arguments parsed from command line
        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
                'timesteps_per_batch': 2000, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.998, 
                'n_updates_per_iteration': 10,
                'lr': 5e-3, 
                'clip': 0.17,
                'lambda_return' : 0.99,
                'annealing_rate' : 0.998,
                'std_set_iteration' : 8,
                'exploration_factor' : 0.5,
                'render': True,
                'render_every_i': 10,
              }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym
    env = gym.make('MountainCar-v0', render_mode = 'rgb_array')

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model, ep=args.ep)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    main(args)