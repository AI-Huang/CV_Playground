export PYTHONPATH="${PYTHONPATH}:./"

# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=subtract_mean_pad_crop --validation_split=0.1 --batch_size=128 --epochs=200  --learning_rate=0.1 --optimizer_name=SGD --weight_decay=0.0001 --momentum=0.9 --lr_schedule=cifar10_scheduler --date_time=${date_time} --run=${i};
# done

# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=subtract_mean_pad_crop --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
# done

# # keras_augmentation
# for n in "3 5 7 9 18"; do
# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python ./scripts/train_resnet_cifar10_tf.py --n=${n} --version=1 --dataset=cifar10 --data_augmentation=keras_augmentation --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
# done
# done

# std_norm_pad_crop
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
# --batch_size=32
# --optimizer_name=Adam
# --learning_rate=0.001
# --lr_schedule=keras_lr_scheduler
# discuss on data_augmentation
# --data_augmentation=pad_crop, std_norm_pad_crop
python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=pad_crop --validation_split=0.0 --batch_size=32 --epochs=200 --optimizer_name=Adam --learning_rate=0.001 --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done

# # std_norm_pad_crop, SENet
# date_time=$(date "+%Y%m%d-%H%M%S")
# for ((i=1;i<=5;i++)); do
# # --seed=42
# python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=std_norm_pad_crop --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
# done
