from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

logging.basicConfig(filename="log.txt", level=logging.INFO)
logging.debug("Debug logging test...")

# Set random seed for reproducibility
seed = 999
logging.info(f"Random Seed: {seed}")
random.seed(seed)
torch.manual_seed(seed)