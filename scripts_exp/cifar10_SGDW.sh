for ((i=1;i<=5;i++));
do
# --seed=42
python ./scripts/train_resnet_cifar10_tf.py --n=3 --version=1 --dataset=cifar10 --validation_split=0.1 --batch_size=128 --epochs=200  --learning_rate=0.1 --optimizer_name=SGDW --weight_decay=0.0001 --momentum=0.9 --lr_schedule=cifar10_schedule;
done
