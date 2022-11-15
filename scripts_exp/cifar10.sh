for ((i=1;i<=5;i++));
do
# --seed=42
python train_keras.py --model_name=ResNet50 --dataset=cifar10 --batch_size=128 --epochs=200  --learning_rate=0.1 --optimizer_name=SGD --weight_decay=0.0001 --momentum=0.9 --lr_schedule=cifar10_schedule;
done
