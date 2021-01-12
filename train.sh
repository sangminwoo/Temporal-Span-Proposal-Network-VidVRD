config=configs/baseline.yaml
data_dir=/home/t2_u1/data
dataset=vidor

python base.py --config ${config} --data_dir $data_dir --dataset $dataset --train