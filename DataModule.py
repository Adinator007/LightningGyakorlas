import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms, datasets


class CustomImageDataset(Dataset):
    def __init__(self, transforms):
        super(CustomImageDataset, self).__init__()
        self.train_set = datasets.CIFAR10(
            root="MNIST", download=True, train=True,
            transform=transforms
        )

    def __len__(self):
        return len(self.train_set) // 5 # vesszuk a dataset 1/5-odet

    def __getitem__(self, idx):
        return self.train_set.__getitem__(idx)



class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        train_set = CustomImageDataset(self.transform)

    def setup(self, stage):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)