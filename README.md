# MNIST Playground

Playground for MNIST dataset.

使用不同框架，应用各种神经网络架构，在 MNIST 数据集上训练。

## Requirements and Dependencies

Model architectures are available in my `models` repository here: https://github.com/AI-Huang/models.

## Framework + architecture

| Framework | Architecture        |
| --------- | ------------------- |
| Keras     | LeCunLeNet5, LeNet5 |
| PyTorch   | LeNet5              |

```Python
python main_keras.py --model_name=LeNet5 # --epochs=100
```

```Python
python main_keras.py --model_name=ResNet50
```

### Keras LeNet5

```Python
python main_keras.py --model_name=LeNet5 --epochs=100
```

#### Feature extractors

## PyTorch CNN

教程[1]，PyTorch 官方的代码，使用简单二层卷积网络。

### PyTorch LeNet5

PyTorch, using LeNet5 architecture.

## MNIST Experiments

Use normalization:

```bash
python ./main_mnist.py --do-train --do-eval --epochs 500 --batch-size 32
```

Train without normalization:

```bash
python ./main_mnist.py --do-train --do-eval --epoch 500 --batch-size 32 --no-norm
```

## RMNIST Experiments

```bash
python ./main_rmnist.py --do-train --do-eval --epoch 500 --batch-size 8
```

## Backup of using TensorFlow 1.x

We mainly use TensorFlow 2.x. For tutorial for TensorFlow version 1.x, please ref to \[3\].

## References

[1] PyTorch example: MNIST [https://github.com/pytorch/examples/blob/master/mnist/main.py](https://github.com/pytorch/examples/blob/master/mnist/main.py)

[2]
