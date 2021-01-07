data_dir=/home/t2_u1/data
dataset=vidvrd
split=train # test
task=relation # object, action, relation
prediction=./vidvrd-baseline-output/models/baseline_relation_prediction.json

python evaluate.py --data_dir ${data_dir} --dataset ${dataset} \
--split ${split} --task ${task} --prediction ${prediction}