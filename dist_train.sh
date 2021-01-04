data_dir=/home/t2_u1/data
dataset=vidvrd
nodes=1
ngpus_per_node=4
local_rank=0

python baseline.py --data_dir $data_dir --dataset $dataset --train \
--nodes ${nodes} --ngpus_per_node ${ngpus_per_node} --local_rank ${local_rank}