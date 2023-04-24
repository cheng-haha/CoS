'''
Description: 
Date: 2023-04-23 13:20:57
LastEditTime: 2023-04-24 07:53:02
FilePath: /chengdongzhou/action/CoS/dataset/HAR_dataset.py
'''
import numpy as np
import os
from torch.utils.data import Dataset
from collections import Counter
import torch
from configs import args

ROOTPATH = args.dataset_path


class HARDataset(Dataset):
    def __init__(self, dataset  =   'ucihar', split    =   'train', use_portion   =    1.0):
        self._select_dataset(dataset)
        data_path   = os.path.join(self.ROOT_PATH , "x_{}.npy".format(split))
        label_path  = os.path.join(self.ROOT_PATH , "y_{}.npy".format(split))
        self.data   = np.load(data_path)
        self.label  = np.load(label_path)
        self.use_portion = use_portion
        self.idxs   = np.arange( len(self.data) )
        self.sample_idxs = None
        if use_portion < 1:
            sample_idxs = np.random.choice(self.idxs , int(use_portion * len(self.data)) , replace=False)
            self.data   = self.data[sample_idxs]
            self.label  = self.label[sample_idxs]
            print(Counter(self.label))
            self.sample_idxs = sample_idxs
        self.data       = np.expand_dims( self.data , axis=1 ) 
        print(f'==>{split} data shape is',self.data.shape)
        _, channel_dim,  self.window_width, self.feat_dim = self.data.shape
        samples         = self.data.transpose(3,0,1,2).reshape(self.feat_dim, -1)

        if split == 'train':
            self.mean   = np.mean(samples, axis=1)
            self.std    = np.std(samples, axis=1)
        
    def normalize(self, mean, std):
        self.data = self.data - mean.reshape(1, -1, 1)
        self.data = self.data / std.reshape(1, -1, 1)

    def get_sampled_idxs(self):
        if self.sample_idxs is not None:
            return self.sample_idxs
        # else:
        #     raise('Do not get the sampled idxs')

    def unlabel_processing(self,sample_idxs):
        if self.sample_idxs is not None:
            unlabel_idxs = np.delete(self.idxs,sample_idxs,None)
            self.data    = self.data[unlabel_idxs]
            self.label   = self.label[unlabel_idxs]
        
        print(f'==>unlabel data shape is',self.data.shape)
        
    def _select_dataset(self, dataset):
        if dataset == 'ucihar':
            self.ROOT_PATH = ROOTPATH + "/ucihar"
            self.sampling_rate  = 50
            self.n_actions = 6
            self.window_width   = 128
        else:
            raise NotImplementedError("This dataset is not supported for now")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.Tensor(self.data[i]), self.label[i]



if __name__ == "__main__":
    ds      = 'ucihar'
    train   = HARDataset(dataset=ds, split="train")
    val     = HARDataset(dataset=ds, split="valid")
    test    = HARDataset(dataset=ds, split="test")

    print("# train : {}".format(len(train)))
    n_train = dict(Counter(train.label))
    print(sorted(n_train.items()))
    print("# val : {}".format(len(val)))
    n_val = dict(Counter(val.label))
    print(sorted(n_val.items()))
    print("# test : {}".format(len(test)))
    n_test = dict(Counter(test.label))
    print(sorted(n_test.items()))
    pass