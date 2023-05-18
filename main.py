'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-05-18 22:12:10
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
from utils.setup import GetModel, set_seed
from utils.train import MetaTrain,evaluate
import warnings
warnings.filterwarnings("ignore")
import pyfiglet
print('----------------------------------------------------------')
result = pyfiglet.figlet_format(text="CoS FRAME", font="slant")
print(result)
print('----------------------------------------------------------')
use_cuda = torch.cuda.is_available()
if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))
 
    train_set = HARDataset(dataset=args.dataset, split='train')
    valid_set = HARDataset(dataset=args.dataset, split='valid')
    test_set  = HARDataset(dataset=args.dataset, split='test')
    args.save_folder = os.path.join(args.model_path, args.dataset, args.model , args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # logging    
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger  = initialize_logger(log_dir)
    acc_list,f1_list=[],[]
    for i in range(args.times):
        args.time   = i
        # train
        set_seed(i+1)
        logger.info(f'\n-------run time {i}--------\n')
        train = MetaTrain(args)
        print(f'==>The training method is {train.__name__}')
        train_times, Loss_test , EvaAcc_test, EvaF1_test, state = train(train_set,valid_set,logger)
        model = GetModel()
        total_loss, acc_test, f1_test , state = evaluate(model,logger=logger,epoch = state['best_epoch'],eval_loader=test_set)
        acc_list.append(acc_test)
        f1_list.append(f1_test)
    logger.info(f'Mean acc:{np.mean(acc_list)}, Mean f1:{np.mean(acc_list)}')
        
