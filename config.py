import argparse
import os

from utils.train_utils import add_flags_from_config

config_args = {
    'model_config': {
        'model': ('MKGNN', 'which model to use: [MKGNN] in "main.py" and [GWNN, GCN, GAT]  in "compare.py"'),
        'clip-num':(0.0, 'Laplace matrix clip'),
        'cuda':(0, 'which device to use, -1 for using CPU'),
        'order':(3, 'order of cheb approx, only for GWN'),
        'dp':(0.8, 'dropout rate'),
        'n-hid':(64, 'list of hidden dimensions'),
        'use-bias': (True, 'use bias or not'),
        'alpha':(0.1, 'only for GAT'),
        'n-heads':(4, 'only for GAT'),
    },
    'data_config': {
        'top_k': (20, 'top k job'),
        'datadir':('data', 'directory of the dataset'),
        'val-prop': (0.2, 'proportion of validation samples'),
        'batch-size':(8, 'batch size'),
        'k_job': (100, 'job class'),
        'k_person': (500, 'person_class'),
        'out_file':('/individual/hanxiao/codesh/out_prob.pkl', 'inference file')
    },
    'training_config':{
        'seed': (101, 'seed'),
        'lr': (0.001, 'learning rate'),
        'n-epoch': (5000, 'number of total epochs'),
        'weight_decay': (0.001, 'parameter for optimizer'),
        'gamma': (0.9, 'parameter for LR'),
        'stepsize': (1000, 'parameter for LR'),
        'beta_s': (0.4, 'parameter for optimizer'),
        'beta_e': (0.9999, 'parameter for optimizer'),
    }
}


parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)