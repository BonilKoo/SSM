import argparse
import os
import random
import sklearn
import numpy as np

# set up seed value
def seed_everything(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	sklearn.utils.check_random_state(seed)

def float_range(arg):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")
    
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"{arg} is not in the range [0.0, 1.0]")
    
    return value

def print_changed_args(args_trained, args_input):
    if args_trained.n_rw != args_input.rw:
        print(f'The value of argument "rw changes from "{args_input.rw}" to "{args_trained.n_rw}" according to the trained model.')
    if args_trained.n_alpha != args_input.alpha:
        print(f'The value of argument "rw changes from "{args_input.alpha}" to "{args_trained.n_alpha}" according to the trained model.')
    if args_trained.n_iteration != args_input.iterations:
        print(f'The value of argument "rw changes from "{args_input.iterations}" to "{args_trained.n_iteration}" according to the trained model.')
    if args_trained.n_walkers != args_input.nWalker:
        print(f'The value of argument "rw changes from "{args_input.n_walkers}" to "{args_trained.nWalker}" according to the trained model.')