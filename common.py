'''
Description: 
Date: 2023-04-18 20:53:33
LastEditTime: 2023-04-24 12:44:41
FilePath: /chengdongzhou/action/CoS/common.py
'''

channel_list    =  {
                    'ucihar':   [ 64, 128, 256, 6  ],
                    'pamap2':   [ 64, 128, 256, 12 ],
                    'wisdm' :   [ 64, 128, 256, 6  ],
                    'unimib':   [ 64, 128, 256, 17 ],
                        }

conv_list       =   {
                    'ucihar':   [ (5,1), (1,1), (2,0) ],
                    'pamap2':   [ (5,1), (1,1), (2,0) ],
                    'wisdm' :   [ (3,1), (1,1), (1,0) ],
                    'unimib':   [ (3,1), (1,1), (1,0) ],
                        }

maxp_list       =   {
                    'ucihar':   [ (4,1), (4,1) ],
                    'pamap2':   [ (5,1), (5,1) ],
                    'wisdm' :   [ (3,1), (3,1) ],
                    'unimib':   [ (4,1), (4,1) ],
                        }

