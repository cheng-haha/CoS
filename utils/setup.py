'''
Description: 
Date: 2023-04-18 20:56:41
LastEditTime: 2023-04-30 15:31:29
FilePath: /chengdongzhou/action/CoS/utils/setup.py
'''
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR,CosineAnnealingLR,StepLR
import torch
from configs import args
import numpy as np
import torch.optim as optim
import models
import random

def LrSchedulerSet(optimizer,args):
    '''
    default: StepLR , gamma is 0.1 , step_size is 40
    '''
    if args.lr_scheduler == 'C':
        print('==>LrScheduler Set CosineAnnealingLR')
        return  CosineAnnealingLR( optimizer, T_max=args.n_epochs )
    elif args.lr_scheduler == 'S':
        print(f'==>LrScheduler Set StepLR , decay epoch is {args.n_epochs} , gamma is {args.gamma}')
        return  StepLR( optimizer, step_size=args.n_epochs, gamma=args.gamma )
    elif args.lr_scheduler == 'E':
        print('==>LrScheduler Set ExponentialLR')
        return  ExponentialLR( optimizer, gamma=args.gamma )
    elif args.lr_scheduler == 'M':
        print(f'==>LrScheduler Set MultiStepLR , scale is {args.milestones} , gamma is {args.gamma} ')
        return  MultiStepLR( optimizer , args.milestones , gamma=args.gamma )
    else:
        raise('No Set Lr_Scheduler!')


def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def GetModel():
    try:
        model       = getattr(models, args.model)(args.dataset).cuda()
    except Exception as e:
        raise NotImplementedError("{} is not implemented".format(args.model))
    return model

def GetMOS(opt_type='sgd'):
    '''
    M: Model
    O: Optimier
    S: Scheduler
    '''
    # print(args.model)
    try:
        model       = getattr(models, args.model)(args.dataset).cuda()
    except Exception as e:
        raise NotImplementedError("{} is not implemented".format(args.model))
    if opt_type     == 'sgd':
        optimizer   = optim.SGD(      params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum )
    else:
        raise NotImplementedError("{} is not implemented".format(args.opt_type))
    scheduler = LrSchedulerSet( optimizer , args )
    return model, optimizer, scheduler


