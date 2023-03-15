import torch
from torchvision import datasets, transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.CIFAR = datasets.CIFAR10('./data/CIFAR', train=False, download=True, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                   ]))

    def __getitem__(self, index):
        data, target = self.CIFAR[index]

        return data, target, index

    def __len__(self):
        return len(self.CIFAR)


