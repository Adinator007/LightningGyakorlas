import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
import torch

class ImagenetTransferLearning(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--height", type=int, default=256)
        parser.add_argument("--width", type=int, default=256)
        return parent_parser

    def __init__(self, width = 256, height = 256):
        super().__init__()

        self.glob = 0

        # init a pretrained resnet
        self.width = width
        self.height = height
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def training_epoch_end(self, outputs):
        # a loggerek az epoch ot implicit tudjak beepitve -> self.current_epoch, de mukodik a CheckPointModel {epoch} alias val is
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["logs"]["correct"] for x in outputs])
        total = sum([x["logs"]["total"] for x in outputs])

        '''
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("Train Loss", )
        '''

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", correct / total, self.current_epoch)

    def validation_step(self, x, *args, **kwargs): # ez miatt indult volna el?? not enough arguments for validation step
        loss, logs = self.forward(x)
        self.log('val_loss', loss)
        return {
            "val_loss": loss,
            "logs": logs
        }

    def on_train_batch_end(self, outputs, *args, **kwargs):
        self.logger.experiment.add_scalar("batch losses", outputs["loss"].item(), self.glob)
        # ez szepen megjelenik a tensorboard fooldalon alul
        self.glob += 1

    def training_step(self, x):
        # tensorboard.add_histogram(...)
        # tensorboard.add_figure(...)
        loss, logs = self.forward(x)
        return {
            "loss": loss,
            "logs": logs
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        # optimizer.param_groups[0]['capturable'] = True
        return optimizer

    def forward(self, x):
        import torch.nn.functional as F
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x[0]).flatten(1)
        result = self.classifier(representations)
        correct = result.argmax(1).eq(x[1]).sum().item()
        total = len(x[1])

        loss = F.cross_entropy(result, x[1])
        logs = {
            "correct": correct,
            "total": total
        }
        return loss, logs
        return output