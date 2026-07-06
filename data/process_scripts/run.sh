python preprocess_physionet_2012.py 
python preprocess_physionet_2019.py
# python preprocess_mimic_iii.py

python generate_splits.py --val_ratio 0.2  --dataset mimic_iii
python generate_splits.py --val_ratio 0.2  --dataset physionet_2012
python generate_splits.py --val_ratio 0.2  --dataset physionet_2019