export PYTHONPATH="${PYTHONPATH}:./"

# std_norm_pad_crop
date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
# --batch_size=32
# --optimizer_name=Adam
# --learning_rate=0.001
# --lr_schedule=keras_lr_scheduler
# discuss on data_augmentation
# --data_augmentation=pad_crop, std_norm_pad_crop, random_translation
python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_preprocessing=std_norm --data_augmentation=random_translation --validation_split=0.0 --batch_size=32 --epochs=200 --optimizer_name=Adam --learning_rate=0.001 --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done
