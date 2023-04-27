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