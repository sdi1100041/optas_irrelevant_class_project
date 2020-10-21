import torch
from torchvision import datasets, transforms
import numpy as np

class MNISTTransform:
    def train_transform():
        return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307, ), (0.3081, ))
           ])

class EMNISTTransform:
    def train_transform():
        return transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
class CIFAR10Transform:
    def train_transform():
        return transforms.Compose([
               transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    def test_transform():
        return transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


def get_train_data(task: str):
    if task == "MNIST":
        dataset=  datasets.MNIST(root='../data', train=True, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset=  datasets.EMNIST(root='../data',split="balanced" ,train=True, download=True,transform=EMNISTTransform.train_transform())
    elif task == "CIFAR10":
        dataset= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
    else: #task == "CIFAR100"
        dataset1= datasets.CIFAR100(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
        dataset= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
        dataset1.targets= np.array(dataset1.targets)
        dataset.targets = np.array(dataset.targets)
        indices1 = (dataset1.targets != dataset1.class_to_idx['bicycle']) & (dataset1.targets != dataset1.class_to_idx['bus']) & (dataset1.targets != dataset1.class_to_idx['motorcycle']) & (dataset1.targets != dataset1.class_to_idx['pickup_truck']) & (dataset1.targets != dataset1.class_to_idx['train']) & (dataset1.targets != dataset1.class_to_idx['lawn_mower']) & (dataset1.targets != dataset1.class_to_idx['streetcar']) & (dataset1.targets != dataset1.class_to_idx['rocket']) & (dataset1.targets != dataset1.class_to_idx['tank']) & (dataset1.targets != dataset1.class_to_idx['tractor'])
        dataset1.targets=dataset1.targets[indices1]
        dataset1.data = dataset1.data[indices1]
        dataset1.targets[:] = 10
        dataset.targets = np.concatenate([dataset.targets, dataset1.targets])
        dataset.data = np.concatenate([dataset.data,dataset1.data])
        indices = np.arange(dataset.data.shape[0])
        np.random.shuffle(indices)
        dataset.targets = dataset.targets[indices]
        dataset.data = dataset.data[indices]
    return dataset

def get_validation_data(task: str):
    if task == "MNIST":
        dataset=  datasets.MNIST(root='../data', train=False, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset=  datasets.EMNIST(root='../data',split="balanced" ,train=False, download=True,transform=EMNISTTransform.train_transform())
    elif task == "CIFAR10":
        dataset= datasets.CIFAR10(root='../data', train=False, download=True, transform=CIFAR10Transform.test_transform())
    else: #task == "CIFAR100"
        dataset= datasets.CIFAR10(root='../data', train=False, download=True, transform=CIFAR10Transform.test_transform())
    return dataset
