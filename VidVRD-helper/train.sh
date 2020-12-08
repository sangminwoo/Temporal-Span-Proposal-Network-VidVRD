data_dir=/home/t2_u1/data
dataset=vidvrd

python baseline.py --data_dir $data_dir --dataset $dataset --detect
# python baseline.py --data_dir $data_dir --train
# python baseline.py --data_dir $data_dir --load_feature --train