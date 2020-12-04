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
               transforms.ToTensor()])

    def test_transform():
        return transforms.Compose([
               transforms.ToTensor()])

class AugmentedDataset(datasets.cifar.CIFAR100):
    def __init__(self,root, train, download, transform,cifar10_dataset):
        super().__init__(root=root, train=train,download=download,transform=transform)
        self.targets = np.array(self.targets)
        indices = (self.targets != self.class_to_idx['bicycle']) & (self.targets != self.class_to_idx['bus']) & (self.targets != self.class_to_idx['motorcycle']) & (self.targets != self.class_to_idx['pickup_truck']) & (self.targets != self.class_to_idx['train']) & (self.targets != self.class_to_idx['lawn_mower']) & (self.targets != self.class_to_idx['streetcar']) & (self.targets != self.class_to_idx['rocket']) & (self.targets != self.class_to_idx['tank']) & (self.targets != self.class_to_idx['tractor'])
        self.targets=self.targets[indices]
        self.data = self.data[indices]
        self.targets[:] = 10
        self.improper_dataset_size = len(self.data)
        self.improper_augmented_dataset_size=8*len(self.data)
        self.proper_dataset_size=len(cifar10_dataset.data)
        cifar10_dataset.targets=np.array(cifar10_dataset.targets)
        self.targets = np.concatenate([self.targets, cifar10_dataset.targets])
        self.data = np.concatenate([self.data,cifar10_dataset.data])

    def __len__(self):
        return self.improper_augmented_dataset_size + self.proper_dataset_size

    def __getitem__(self, idx):
        if (idx >= self.improper_augmented_dataset_size):
            idx = self.improper_dataset_size + idx - self.improper_augmented_dataset_size
            return transforms.ToTensor()(self.data[idx]), self.targets[idx]
        
        idx = idx % self.improper_dataset_size
        trnsf= idx / self.improper_dataset_size
        x = transforms.functional.to_pil_image(self.data[idx])
        if (trnsf%2):
            x=transforms.functional.vflip(x)
        if ((trnsf/2)%2):
            x=transforms.functional.hflip(x)
        if ((trnsf/4)%2):
            x=transforms.functional.rotate(x,90)
        
        return transforms.ToTensor()(x), self.targets[idx]
    

def get_train_data(task: str):
    if task == "MNIST":
        dataset=  datasets.MNIST(root='../data', train=True, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset=  datasets.EMNIST(root='../data',split="balanced" ,train=True, download=True,transform=EMNISTTransform.train_transform())
    elif task == "CIFAR10":
        dataset= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
    else: #task == "CIFAR100"
        dataset1= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
        dataset= AugmentedDataset(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform(), cifar10_dataset = dataset1)
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
