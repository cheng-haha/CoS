'''
Description: 
Date: 2023-04-18 20:56:41
LastEditTime: 2023-04-23 16:54:17
FilePath: /chengdongzhou/action/CoS/utils/metric.py
'''
import numpy as np
import time
from common import maxp_list
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def get_data_size(data_name):
    if data_name == 'ucihar':
        size = (1,1,128,9)
    elif data_name == 'pamap2':
        size = ( 1, 1 , 171 , 36 )
    elif data_name =='unimib':
        size = ( 1,1, 151, 3 )
    elif data_name == 'wisdm':
        size = ( 1,1, 200, 3 ) 
    else:
        raise Exception( 'please input correct data name')
    return size

def get_classes(data_name):
    if data_name == 'ucihar':
        classes = 6
    elif data_name == 'pamap2':
        classes = 12
    elif data_name =='unimib':
        classes = 17
    elif data_name == 'wisdm':
        classes = 6
    else:
        raise Exception( 'please input correct data name')
    return classes

def GetFeatureMapSize(data_name,idex_layer):
    size = get_data_size(data_name)[2:]
    maxpooling_size = maxp_list[data_name]
    h,w = size
    if idex_layer > 0:
        for i in range(idex_layer):
            h //= maxpooling_size[0][0]
            w //= maxpooling_size[0][1]
        return ( h , w )
    else:
        raise  ValueError(f'check your idex_layer')
