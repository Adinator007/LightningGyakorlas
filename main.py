import os
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, layers), nn.ReLU(), nn.Linear(layers, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", test_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        # optimizer.param_groups[0]['capturable'] = True
        return optimizer

def main():
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets

    # Load data sets
    train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())

    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder()).cuda()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")

    # train model
    trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=10, default_root_dir=r"D:\Lightning\CheckPoints", enable_checkpointing=True, accelerator="cpu", devices=1
                         ) # True by default
    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set), val_dataloaders=DataLoader(valid_set) # ,
                # ckpt_path=r"D:\Lightning\CheckPoints\lightning_logs\version_0\checkpoints\epoch=0-step=48000.ckpt"
                ) # early stopping doable??

    from torch.utils.data import DataLoader

    # test the model
    trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))

if __name__ == '__main__':
    main()