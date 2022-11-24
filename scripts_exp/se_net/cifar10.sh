date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
# --n=3
python train_resnet_cifar10.py --n=18 --version=1 --se_net --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done
