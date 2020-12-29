data_dir=/home/t2_u1/data
dataset=vidvrd
phase='test'

python baseline.py --data_dir ${data_dir} --dataset ${dataset} --preprocess --phase ${phase}