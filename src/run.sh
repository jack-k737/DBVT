python main.py --device cuda:0 --model_type dbvt --dataset physionet_2012 --fold 1 &
python main.py --device cuda:1 --model_type dbvt --dataset physionet_2012 --fold 2 &
python main.py --device cuda:2 --model_type dbvt --dataset physionet_2012 --fold 3 &
python main.py --device cuda:3 --model_type dbvt --dataset physionet_2012 --fold 4 &
python main.py --device cuda:0 --model_type dbvt --dataset physionet_2012 --fold 5 &



python main.py --device cuda:1 --model_type dbvt --dataset physionet_2019 --fold 1 &
python main.py --device cuda:2 --model_type dbvt --dataset physionet_2019 --fold 2 &
python main.py --device cuda:3 --model_type dbvt --dataset physionet_2019 --fold 3 &

wait
python main.py --device cuda:0 --model_type dbvt --dataset physionet_2019 --fold 4 &
python main.py --device cuda:1 --model_type dbvt --dataset physionet_2019 --fold 5 &



python main.py --device cuda:2 --model_type dbvt --dataset mimic_iii --fold 1 &
python main.py --device cuda:3 --model_type dbvt --dataset mimic_iii --fold 2 &
python main.py --device cuda:0 --model_type dbvt --dataset mimic_iii --fold 3 &
python main.py --device cuda:2 --model_type dbvt --dataset mimic_iii --fold 4 &
python main.py --device cuda:3 --model_type dbvt --dataset mimic_iii --fold 5 &

wait    
python ../scripts/aggregate_test_metrics.py