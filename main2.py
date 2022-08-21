# ----------------
from argparse import ArgumentParser

from pytorch_lightning import Trainer, LightningModule


class LitModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser

def main():

    parser = ArgumentParser(add_help=False)

    # add PROGRAM level args
    parser.add_argument("--conda_env", type=str, default="some_name")
    parser.add_argument("--notification_email", type=str, default="will@email.com")

    parser.add_argument("--auto_scale_batch_size", type=str, default='power')
    # parser.add_argument("--max_epochs", type=int, default=10)
    # parser.add_argument("--default_root_dir", type=str, default=r"D:\Lightning\CheckPoints")

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)

    # init the model with Namespace directly
    model = LitModel(args)

    # or init the model with all the key-value pairs
    dict_args = vars(args)
    model = LitModel(**dict_args)

if __name__ == '__main__':
    main()