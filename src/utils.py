import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = '/home/ubuntu/hw4/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['test_file'] = 'test_category_hw.txt'
Config['checkpoint_path'] = ''
Config['train_compatibility']='pairwise_compatibility_train.txt'
Config['valid_compatibility']='pairwise_compatibility_valid.txt'

Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 10
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5
