# Never Give Up

This repo is copied from the PPO for Beginners implementation by Eric Yu. I intend to build NGU on top of it, and make it learn PICO-8 (I'll start with Celeste, then we'll see).

# TODO:

Implement discrete action space support

Make a gym environment from PICO-8

Build the whole of NGU lol

# Known issues

The only way I found of interfacing with PICO-8 from code was by simulating keypresses. This is neither convenient nor safe: you have to yield your keyboard to the program and not touch anything while it's running. The solution I came up with is to enclose the learning process in a virtual machine. Inefficient, I know, but it works, so *shrug*.
