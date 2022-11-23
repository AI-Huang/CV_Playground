python ./scripts/rmnist.py --no-data-augmentation --epochs 500 --batch-size 10 --seed 31
python ./scripts/rmnist.py --no-data-augmentation --epochs 500 --batch-size 8 --seed 42
python ./scripts/rmnist.py --no-data-augmentation --epochs 500 --batch-size 4 --seed 42
python ./scripts/rmnist.py --no-data-augmentation --model-name LeNet5_Dropout --epochs 500 --batch-size 10 --seed 42
python ./scripts/rmnist.py --n_samples 1024 --model-name LeNet5_Dropout --epochs 50 --batch-size 32 --seed 42
