###
 # @Description: 
 # @Date: 2023-04-23 20:43:19
 # @LastEditTime: 2023-05-18 22:16:18
 # @FilePath: /chengdongzhou/action/CoS/scripts/run.sh
### 

# Baseline
python main.py --dataset ucihar --model CNN\
                        --device 1\
                        --batch_size 128\
                        --learning_rate 0.01\
                        --epochs 200\
                        --mode CoSBase\
                        --times 1\
                        --chhander
CoS                
python main.py --dataset ucihar --model CoS_CNN\
                                --device 1\
                                --batch_size 128\
                                --learning_rate 0.01\
                                --mode CoSBase\
                                --times 1\
                                --lam 5.0\
                                --supervision \
                                --epochs 200 \
                                --proj_dim 128\
                                --data_aug negate\
                                --chhander
