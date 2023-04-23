'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-04-23 23:01:54
FilePath: /chengdongzhou/action/CoS/configs.py
'''
import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: 0). if seed=-1, seed is not fixed.')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--times', type=int, default=1, help='num of different seed')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--device', type=int, default=2, choices=[0,1,2])
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--window_width', type=int, default=0, help='window width')
parser.add_argument('--normalize', action='store_true', default=False, help='normalize signal based on mean/std of training samples')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--not_save_res','--nsr', default=False, action='store_true',
                          help='The default value is saved.')
parser.add_argument('--mode', type=str, default='ce', help='mode')
parser.add_argument('--trial', type=str, default='default', help='trial id')
parser.add_argument('--chhander', action='store_true', default=False)
parser.add_argument('--opt_type', type=str, default='sgd')
parser.add_argument('--not_avg_exp', action='store_false', default=True)

# optimization
parser.add_argument('--learning_rate','--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default = 0.0005 , help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lr_scheduler', type=str, default='S', help='lr_scheduler')

# MultiStepLR or ExponentialLR
parser.add_argument('--gamma',type=float,default=0.1,help='gamma')##
parser.add_argument('--milestones',type=list,default=[40,80,120,160],help='optimizer milestones')

# CosineAnnealingLR
parser.add_argument('--n_epochs',type=float,default=40,help='n_epochs for CosineAnnealingLR')

# dataset and model
parser.add_argument('--dataset', type=str, default='ucihar')
parser.add_argument('--model', type=str, default='CNN')
parser.add_argument('--dataset_path', type=str, default='/data1/experiment/chengdongzhou/action/CoS/dataset')
parser.add_argument('--train_portion', type=float, default=1.0, help='use portion of trainset')
parser.add_argument('--model_path', type=str, default='save', help='path to save model')
parser.add_argument('--load_model', type=str, default='', help='load the pretrained model')

# coefficients
parser.add_argument('--lam', type=float, default=0.0, help='An alternate measure coefficient, which defaults to 0')

# for CoS method
parser.add_argument('--semi_radio', type=float, default=0.0)
parser.add_argument('--proj_dim', type=int, default=0)
parser.add_argument('--proj_type', type=str, default='simple')
parser.add_argument('--supervision', action='store_true', default=False)
parser.add_argument('--vis_feature', action='store_true', default=False)
parser.add_argument('--linear_evaluation', action='store_true', default=False)
parser.add_argument('--data_aug', type=str, default=None,\
    choices=['na','shuffle','jit_scal','perm_jit','resample','noise','scale','negate','t_flip','rotation','perm'])
parser.add_argument('--data_aug2', type=str, default=None,\
    choices=['na','shuffle','jit_scal','perm_jit','resample','noise','scale','negate','t_flip','rotation','perm'])
parser.add_argument('--temperature', type=float, default=0.12)


args = parser.parse_args()
if args.pretrain:
    args.lambda_cls = 0.0
    args.lambda_ssl = 1.0
        
def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

#Display settings
#print(dict_to_markdown(vars(args), max_str_len=120))