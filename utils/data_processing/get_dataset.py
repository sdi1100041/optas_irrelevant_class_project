import torch
from torchvision import datasets, transforms

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


def get_train_data(task: str, irr_class: int=-1):
    if task == "MNIST":
        dataset=  datasets.MNIST(root='../data', train=True, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset=  datasets.EMNIST(root='../data',split="balanced" ,train=True, download=True,transform=EMNISTTransform.train_transform())
    else:# task == "CIFAR10"
        dataset= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
    if irr_class != -1:
        dataset.targets= torch.tensor(dataset.targets)
        indices = dataset.targets != irr_class
        dataset.targets[dataset.targets > irr_class] -=1
        dataset.targets=dataset.targets[indices]
        dataset.data = dataset.data[indices]
    return dataset

def get_validation_data(task: str, irr_class: int=-1):
    if task == "MNIST":
        dataset=  datasets.MNIST(root='../data', train=False, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset=  datasets.EMNIST(root='../data',split="balanced" ,train=False, download=True,transform=EMNISTTransform.train_transform())
    else:# task == "CIFAR10"
        dataset= datasets.CIFAR10(root='../data', train=False, download=True, transform=CIFAR10Transform.test_transform())
    if irr_class != -1:
        dataset.targets= torch.tensor(dataset.targets)
        indices = dataset.targets != irr_class
        dataset.targets[dataset.targets > irr_class] -=1
        dataset.targets=dataset.targets[indices]
        dataset.data = dataset.data[indices]
    return dataset
