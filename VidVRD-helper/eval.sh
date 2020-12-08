data_dir=/home/t2_u1/data
dataset=vidvrd
split=test
task=relation
prediction=./vidvrd-baseline-output/models/baseline_relation_prediction.json

python evaluate.py $data_dir $dataset $split $task $prediction