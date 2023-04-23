'''
Description: 
Date: 2023-04-18 20:56:41
LastEditTime: 2023-04-23 20:37:09
FilePath: /chengdongzhou/action/CoS/utils/logger.py
'''

import logging
import numpy as np
from configs import args


                
def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger      = logging.getLogger()
    fhandler    = logging.FileHandler(filename=file_dir, mode='a')
    chhander    = logging.StreamHandler() if args.chhander else None
    formatter   = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt ='%m-%d %H:%M')
    fhandler.setFormatter(formatter)
    chhander.setFormatter(formatter) if args.chhander else None
    logger.addHandler(fhandler)
    logger.addHandler(chhander) if args.chhander else None
    logger.setLevel(logging.INFO)
    return logger


def record_result(result, epoch, acc, f1, c_mat , record_flag = 0):
    """ Record evaluation results."""
    if record_flag == 0:
        result.write('Best validation epoch | accuracy: {:.4f}, F1: {:.4f} (at epoch {})\n'.format(acc, f1, epoch))
    elif record_flag == -1:
        result.write('\n\nTest (Best) | accuracy: {:.4f}, F1: {:.4f}\n'.format(acc, f1, epoch))
        result.write(np.array2string(c_mat))
        result.flush()
        result.close
    elif record_flag == -2:
        result.write('\n\nTest (Final) | accuracy: {:.4f}, F1: {:.4f}\n'.format(acc, f1, epoch))
    elif record_flag == -3:
        result.write('\n\nFinal validation epoch | accuracy: {:.4f}, F1: {:.4f}\n'.format(acc, f1, epoch))
    