import os
import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets
from pytorch_lightning import loggers as pl_loggers

from ImagenetTransferLearning import ImagenetTransferLearning


if __name__ == '__main__':
    # Load data sets

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser = ImagenetTransferLearning.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    train_set = CustomImageDataset()

    # train_set = Subset(train_set, int(len(train_set) * 0.2)) # joval kisebbre vesszuk a bemeneti adatok meretet a gyors iteralas miatt TODO kivenni
    # ezzel az a baj, hogy Subset-tel ter vissza. Utolag nem tudod szejjel szedni, mert nem Dataset object et kapsz vissza

    # use 20% of training data for validation
    train_set_size = len(train_set) // 2
    valid_set_size = len(train_set) - train_set_size
    # manualisan kisebbre veve, hogy gyorsabban iteraljak

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    # train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    os.environ['TORCH_HOME'] = r'D:\Lightning\PreTrained\resnet'
    model = ImagenetTransferLearning()
    # tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"D:\Lightning\logs\tensorboard")
    # ahhoz, hogy elinditsd anaconda-ban ki kell adni a kovetkezo parancsot:
    # tensorboard --logdir logs, a vegen a logs konyvtarnak az eleresi utvonalat kerdezi
    # ha belemesz a folder-be akkor mukodik a --logdir . kapcsolo is
    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"C:\Users\Adam\PycharmProjects\lightning\logs\tensorboard")

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath="my/path/",
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer.from_argparse_args(args, logger=tensorboard, callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)])
    # trainer = Trainer(args)
    trainer.fit(model, train_dataloaders=DataLoader(train_set, batch_size=10), val_dataloaders=DataLoader(valid_set, batch_size=10))
