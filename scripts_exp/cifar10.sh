for ((i=1;i<=5;i++));  
do   
python train_keras.py --dataset=cifar10 --seed=42 --batch_size=128 --epochs=200 --model=ResNet50 --learning_rate=0.1 --weight_decay=0.0001 --momentum=0.9 --lr_schedule=cifar10_schedule --optimizer_name=SGD;
done
