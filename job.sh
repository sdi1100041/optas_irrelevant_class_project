#!/bin/sh

python3 normal_main.py --task=CIFAR10 --lr=0.01 --epochs=100 --algorithm=Adam --model=ResNet18

python3 main.py --task=CIFAR100 --lr=0.01 --epochs=100 --algorithm=Adam --model=ResNet18 --regularization

python3 normal_main.py --task=CIFAR10 --lr=0.001 --epochs=100 --algorithm=Adam --model=ResNet18

python3 main.py --task=CIFAR100 --lr=0.001 --epochs=100 --algorithm=Adam --model=ResNet18 --regularization

python3 main.py --task=CIFAR100 --lr=0.001 --epochs=100 --model=ResNet18 --regularization

python3 normal_main.py --task=CIFAR10 --lr=0.001 --epochs=100 --model=ResNet18

python3 main.py --task=CIFAR100 --lr=0.01 --epochs=100 --model=ResNet18 --regularization

python3 normal_main.py --task=CIFAR10 --lr=0.01 --epochs=100 --model=ResNet18

python3 normal_main.py --task=CIFAR10 --lr=0.01 --epochs=100

python3 normal_main.py --task=CIFAR10 --lr=0.01 --epochs=100 --algorithm=Adam

python3 normal_main.py --task=CIFAR10 --lr=0.001 --epochs=100 --regularization

python3 normal_main.py --task=CIFAR10 --lr=0.001 --epochs=100

python3 normal_main.py --task=CIFAR10 --lr=0.01 --epochs=100 --algorithm=Adam

python3 main.py --task=CIFAR100 --lr=0.01 --epochs=100 --regularization --algorithm=Adam

python3 main.py --task=CIFAR100 --lr=0.001 --epochs=100 --regularization --algorithm=Adam

python3 normal_main.py --task=CIFAR10 --lr=0.001 --epochs=100 --algorithm=Adam
