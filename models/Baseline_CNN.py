'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-04-23 19:53:16
FilePath: /chengdongzhou/action/CoS/models/Baseline_CNN.py
'''
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from configs import args
from utils.metric import GetFeatureMapSize
from common import channel_list,conv_list,maxp_list


class CNN(nn.Module):
    def __init__(self, data_name , sub_number = 3  ):
        super(CNN, self).__init__()
        channel     = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]

        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=max_params )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number)
        self.classifier  = nn.Linear(   channel[2]* h * w , channel[-1]   )

    def forward(self, x):
        B,_,_,_ = x.size()
        x   = self.layer1(x)
        x   = self.layer2(x)        
        x   = self.layer3(x)
        x   = x.view(B,-1)
        x   = self.classifier( x )
        res = {}
        res['output'] = x 
        return res
    
    def maker_layers(self,inp,oup,conv_params=None,pooling_params=None):
        assert isinstance(conv_params,list),print('the format of kernel params is error')
        assert isinstance(pooling_params,list) or pooling_params == None ,print('the format of pooling params is error')
        return nn.Sequential(
            nn.Conv2d( inp, oup, *(conv_params) ),
            nn.BatchNorm2d( oup ),
            nn.ReLU( True ),
            nn.MaxPool2d( *(pooling_params) ) if pooling_params else nn.Identity()
        )

