# Relation Classification

## Dependencies

* python 3.6.7
* pytorch 1.0.0

## Test Using Pretrained Model

First, generate relation classsification results:
```
python test.py --model_path logs/pretrained_model/relation_cls.path
```
the results are saved in `logs/results.json`.

Then, generate final prediction results:
```
python gen_final_res.py
```
the results path is `results/result_final.json.

You can use the tools in `VidVRD-helper` to evaluate the results.
```
cd ../VidVRD-helper
./eval.sh ../relation_classifiction/results/result_final.json
```

## Train Relation Classification Model
```
python train.py
```
