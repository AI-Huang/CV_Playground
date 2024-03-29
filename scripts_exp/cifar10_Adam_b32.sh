date_time=$(date "+%Y%m%d-%H%M%S")
for ((i=1;i<=5;i++)); do
# --seed=42
python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --data_augmentation=subtract_mean_pad_crop --validation_split=0.0 --batch_size=32 --epochs=200  --learning_rate=0.001 --optimizer_name=Adam --lr_schedule=keras_lr_scheduler --date_time=${date_time} --run=${i};
done
