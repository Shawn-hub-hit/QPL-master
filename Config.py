import numpy as np
import torch
import os
import pathlib
from datetime import datetime
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import csv
from collections import Counter
from transformers import BertTokenizer,BertModel

from collections import Iterable
from torch.utils import data
import math
import pickle
from math import log
from scipy import spatial
from pathlib import Path
import argparse
import logging
from scipy import spatial

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

print(pathlib.Path.cwd())
DATA_DIR = pathlib.Path('../datasets')
DATA_OUT = pathlib.Path('./')
dataname = 'Beijing_data'
datapath = DATA_DIR / dataname 
# data_records = 'Meituan'

# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(0)
torch.manual_seed(0)

class Constant():
    def __init__(self, cur_time=None):
        if cur_time:
            self.cur_time = cur_time
        else:
            self.cur_time = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')      #set up dir for save
        self.set_logger()
        # self.setup_dir()

    def setup_dir(self):
        tensorboard_logdir = DATA_OUT / 'logs_u' / self.cur_time
        self.logger.info(f"setup dir in {self.cur_time}")
        if not os.path.exists(tensorboard_logdir):
            os.makedirs(tensorboard_logdir)
        self.tensorboard_logger=SummaryWriter(tensorboard_logdir)

    def set_logger(self):
        '''
        Write logs to file and console
        '''
        self.savedir = DATA_OUT / 'checkpoints_u' / self.cur_time
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        log_file = os.path.join(self.savedir, 'logs.txt')
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger("run log")
        
