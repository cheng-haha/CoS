'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-04-23 19:53:02
FilePath: /chengdongzhou/action/CoS/models/CoS_CNN.py
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
        self.proj_dim = proj_dim
        self.prejector = nn.Sequential( 
                                        nn.AdaptiveAvgPool2d( ( ( 1, sensor_dim ) ) ),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear( out_dim*sensor_dim , proj_dim ),
                                        )

    def forward(self, x):
        x = self.prejector(x)
        return x


class CoS_CNN(nn.Module):
    def __init__(self, data_name , embedding_dim = args.proj_dim ):
        super(CoS_CNN, self).__init__()
        channel     = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]

        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=max_params )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
            
        self.auxiliary1 = SimpleProjector( channel[0] , GetFeatureMapSize( data_name,1)[1] , 
                                        embedding_dim ) 
            
        self.auxiliary2 = SimpleProjector( channel[1] , GetFeatureMapSize( data_name,2)[1] , 
                                        embedding_dim ) 

        self.auxiliary3 = SimpleProjector( channel[2] , GetFeatureMapSize(data_name,3)[1] , 
                                        embedding_dim  ) 
        
        self.classifier  = nn.Linear( channel[-2] , channel[-1]   )

    def forward(self, x):
        feature_list    = []
        feat_list       = None
        rep             = None
        B,_,_,_ = x.size()
        x = self.layer1(x)
        feature_list.append(x)
        x = self.layer2(x)
        feature_list.append(x)
        x = self.layer3(x)
        feature_list.append(x)
        rep = x.view(B,-1)
        out = self.classifier(rep)
        if self.training:
            out1_feature = self.auxiliary1(feature_list[0])
            out2_feature = self.auxiliary2(feature_list[1])
            out3_feature = self.auxiliary3(feature_list[2])
            feat_list = [out3_feature , out2_feature , out1_feature ]
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
