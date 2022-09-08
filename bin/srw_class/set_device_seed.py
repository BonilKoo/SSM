#!/usr/bin/env python 
# coding:utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, random, sklearn
import numpy as np

# set up seed value
def seed_everything(seed=3078):
	import sklearn
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	sklearn.utils.check_random_state(seed)
