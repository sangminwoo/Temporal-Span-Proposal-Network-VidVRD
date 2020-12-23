data_dir=/home/t2_u1/data
dataset=vidvrd

# python baseline.py --data_dir $data_dir --dataset $dataset --detect
python baseline.py --data_dir $data_dir --dataset $dataset --train
# python baseline.py --data_dir $data_dir --dataset $dataset --load_feature --train