###
 # @Description: 
 # @Date: 2023-04-23 20:43:19
 # @LastEditTime: 2023-04-23 21:23:09
 # @FilePath: /chengdongzhou/action/CoS/scripts/run.sh
### 
python main.py --dataset ucihar --model CoS_CNN\
                                --device 0\
                                --batch_size 128\
                                --learning_rate 0.01\
                                --mode CoSBase\
                                --lam 5.0\
                                --supervision \
                                --epochs 200 \
                                --proj_dim 128\
                                --data_aug negate\
                                --chhander

python main.py --dataset ucihar --model CNN\
                        --device 0\
                        --batch_size 128\
                        --learning_rate 0.01\
                        --epochs 200\
                        --mode CoSBase\
                        --chhander