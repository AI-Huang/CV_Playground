# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python train_resnet_cifar10.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=subtract_mean_pad_crop --validation_split=0.1 --batch_size=128 --epochs=200  --learning_rate=0.1 --optimizer_name=SGD --weight_decay=0.0001 --momentum=0.9 --lr_schedule=cifar10_scheduler --date_time=${date_time} --run=${i};
# done

# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python train_resnet_cifar10.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=subtract_mean_pad_crop --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
# done

# keras_augmentation
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python train_resnet_cifar10.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done

# keras_augmentation
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python train_resnet_cifar10.py --n=5 --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done

# keras_augmentation
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python train_resnet_cifar10.py --n=7 --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done

# keras_augmentation
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python train_resnet_cifar10.py --n=9 --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done

# keras_augmentation
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python train_resnet_cifar10.py --n=18 --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done
