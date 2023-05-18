'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-05-01 10:48:12
FilePath: /chengdongzhou/action/CoS/models/complex/CoS_CNN.py
'''
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import  channel_list,conv_list,maxp_list
from utils.metric import GetFeatureMapSize 
from configs import args


class SimpleProjector(nn.Module):
    def __init__(self,  out_dim ,  sensor_dim ,  proj_dim ,*args ,**kwargs ):
        super(SimpleProjector, self).__init__()
        self.proj_dim   = proj_dim
        self.prejector  = nn.Sequential( 
                                        nn.AdaptiveAvgPool2d( ( ( 1, sensor_dim ) ) ),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear( out_dim*sensor_dim , proj_dim ),
                                        )

    def forward(self, x):
        x = self.prejector(x)
        return x

class CoS_CNN(nn.Module):
    def __init__(self, data_name , sub_number = 3, embedding_dim = args.proj_dim ):
        super(CoS_CNN, self).__init__()
        self.sub_number = sub_number
        channel     = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        for i in range(self.sub_number):
            setattr(self,f'layer{i+1}', getattr(self,'maker_layers')( channel[i-1] if i-1 >= 0 else 1  , channel[i] ,conv_params=conv_params, pooling_params=max_params ) ) 
            setattr(self,f'auxiliary{i+1}', SimpleProjector( channel[i] , GetFeatureMapSize( data_name,i+1 )[1] , embedding_dim ) )       
        h,w = GetFeatureMapSize(data_name,sub_number)
        self.classifier  = nn.Linear(   channel[2]* h * w , channel[-1]   )

    def forward(self, x):
        feature_list    = []
        feat_list       = None
        rep             = None
        B,_,_,_ = x.size()
        for i in range(self.sub_number):
            x = getattr(self,f'layer{i+1}')(x)
            feature_list.append(x)
        rep = x.view(B,-1)
        out = self.classifier(rep)
        if self.training:
            feat_list       = [ getattr(self,f'auxiliary{i+1}')(feat) for i,feat in enumerate(feature_list) ]
            feat_list.reverse()
            for index in range(len(feat_list)):
                feat_list[index] = F.normalize(feat_list[index], dim=1)
        res = {}
        res['output'] = out 
        res['feat_list'] = feat_list
        res['representation'] = rep
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
