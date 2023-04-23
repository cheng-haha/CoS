'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-04-23 23:22:43
FilePath: /chengdongzhou/action/CoS/main.py
'''

import os
import torch
from configs import args, dict_to_markdown
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
import numpy as np
from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger
from utils.train import MetaTrain
import warnings
warnings.filterwarnings("ignore")
import pyfiglet
print('----------------------------------------------------------')
result = pyfiglet.figlet_format(text="CoS FRAME", font="slant")
print(result)
print('----------------------------------------------------------')
use_cuda = torch.cuda.is_available()
np.random.seed(0)
if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))
 
    train_set = HARDataset(dataset=args.dataset, split='train')
    valid_set = HARDataset(dataset=args.dataset, split='valid')
    args.save_folder = os.path.join(args.model_path, args.dataset, args.model , args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # logging    
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger  = initialize_logger(log_dir)
    for i in range(args.times):
        args.time   = i
        # train
        logger.info(f'\n-------run time {i}--------\n')
        train = MetaTrain(args)
        print(f'==>The training method is {train.__name__}')
        train_times, Loss_test , EvaAcc_test, EvaF1_test = train(train_set,valid_set,logger)
        
