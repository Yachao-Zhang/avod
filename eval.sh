python3 scripts/preprocessing/gen_mini_batches.py
python3 avod/experiments/run_evaluation.py --pipeline_config=avod/data/outputs/$1/$1.config --device='0' --data_split='val'

