import os
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import tensor
import sys
sys.path.append('/home/amachidon/work/agriadapt/segmentation')
sys.path.append('/home/amachidon/work/agriadapt')
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from pruning.EarlyStopper import EarlyStopper
from torch.utils.data import DataLoader
import segmentation.settings as settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from segmentation.models.slim_squeeze_unet import (
    SlimSqueezeUNet,
    SlimSqueezeUNetCofly,
)
from segmentation.models.slim_unet import SlimUNet


class Training:
    def __init__(
        self,
        device,
        architecture=settings.MODEL,
        epochs=settings.EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        learning_rate_scheduler=settings.LEARNING_RATE_SCHEDULER,
        batch_size=settings.BATCH_SIZE,
        regularisation_l2=settings.REGULARISATION_L2,
        image_resolution=settings.IMAGE_RESOLUTION,
        widths=settings.WIDTHS,
        dropout=settings.DROPOUT,
        verbose=1,
        wandb_group=None,
        dataset="geok",
        continue_model="",  # This is set to model name that we want to continue training with (fresh training if "")
        sample=0,
        tobacco_i=0
    ):
        self.architecture = architecture
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.regularisation_l2 = regularisation_l2
        self.image_resolution = image_resolution
        self.widths = widths
        self.dropout = dropout
        self.verbose = verbose
        self.wandb_group = wandb_group
        self.dataset = dataset
        self.continue_model = continue_model
        self.sample = sample
        self.tobacco_i=0

        self.best_fitting = [0, 0, 0, 0]

    def _report_settings(self):
        print("=======================================")
        print("Training with the following parameters:")
        print("Dataset: {}".format(self.dataset))
        print("Model architecture: {}".format(self.architecture))
        print("Epochs: {}".format(self.epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Learning rate scheduler: {}".format(self.learning_rate_scheduler))
        print("Batch size: {}".format(self.batch_size))
        print("L2 regularisation: {}".format(self.regularisation_l2))
        print("Image resolution: {}".format(self.image_resolution))
        print("Dropout: {}".format(self.dropout))
        print("Network widths: {}".format(self.widths))
        print("Loss function weights: {}".format(settings.LOSS_WEIGHTS))
        print(
            "Transfer learning model: {}".format(
                self.continue_model if self.continue_model else "None"
            )
        )
        print("=======================================")

    def _find_best_fitting(self, metrics):
        """
        Could you perhaps try training it by monitoring the validation scores for each
        width and then stopping the training at the epoch which maximises the difference
        between the widths when they are in the right order?

        Compare current metrics to best fitting and overwrite them if new best
        fitting were found given to a heuristic we have to come up with.

        Return True if best fitting was found, otherwise false.
        """
        # metrics = [
        #     metrics["iou/valid/25/weeds"],
        #     metrics["iou/valid/50/weeds"],
        #     metrics["iou/valid/75/weeds"],
        #     metrics["iou/valid/100/weeds"],
        # ]
        metrics = [
            metrics["valid/25/iou/weeds"],
            metrics["valid/50/iou/weeds"],
            metrics["valid/75/iou/weeds"],
            metrics["valid/100/iou/weeds"],
        ]

        # First check if the widths are in order.
        for i, m in enumerate(metrics):
            if i == 0:
                continue
            if metrics[i - 1] > m:
                # print("Metrics not in order, returning false.")
                return False

        # Then check if the differences between neighbours are higher than current best
        # if sum(
        #     [self.best_fitting[i] - self.best_fitting[i - 1] for i in range(1, 4)]
        # ) > sum([metrics[i] - metrics[i - 1] for i in range(1, 4)]):
        #     return False

        # print()
        # print("New best scores:")
        # print(f"Comparing metrics: {metrics}")
        # print(f"Current best:      {self.best_fitting}")
        # print()
        # self.best_fitting = metrics
        return True

    def _learning_rate_scheduler(self, optimizer):
        if self.learning_rate_scheduler == "no scheduler":
            return None
        elif self.learning_rate_scheduler == "linear":
            return LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=self.epochs,
            )
        elif self.learning_rate_scheduler == "exponential":
            return ExponentialLR(
                optimizer,
                0.99,
            )
    def test(self):
        ii = ImageImporter(
            self.dataset,
            validation=False,
            sample=False,
            smaller=self.image_resolution,
            tobacco_i=self.tobacco_i
        )
        _, test = ii.get_dataset()
        test_loader = DataLoader(test, batch_size=1, shuffle=False)

        # Prepare the model
        out_channels = len(settings.LOSS_WEIGHTS)
        if self.architecture == "slim":
            model = SlimUNet(out_channels)
        elif self.architecture == "squeeze":
            model = SlimSqueezeUNet(out_channels)
            if self.dataset == "cofly":
                model = SlimSqueezeUNetCofly(out_channels)
        else:
            raise ValueError("Unknown model architecture.")
        
        garage_home = Path(settings.PROJECT_DIR)/"segmentation/training/garage/tobacco/squeeze"
        garage_path = str(garage_home) + "/image_resolution{}/".format(
            str(self.image_resolution[0]).zfill(4)
        )
        #file_path = garage_path + "geok_squeeze_final.pt"
        file_path = garage_path + "model_checkpoint.pt"

        print(file_path)
        model.load_state_dict(torch.load(file_path))
        model.to(self.device)
        model.eval()

        metrics = Metricise(classes=["back", "weeds"], use_adaptive=False)
        metrics.evaluate(model, test_loader, "test", epoch=0)     
        print(metrics.report(wandb=False))


    def train(self):
        if self.verbose:
            print("Training process starting...")
            self._report_settings()
        # Prepare the data for training and validation
        ii = ImageImporter(
            self.dataset,
            validation=True,
            sample=self.sample,
            smaller=self.image_resolution,
            tobacco_i=self.tobacco_i
        )
        train, validation = ii.get_dataset()
        if self.verbose:
            print("Number of training instances: {}".format(len(train)))
            print("Number of validation instances: {}".format(len(validation)))

        garage_home = Path(settings.PROJECT_DIR)/"segmentation/training/garage/tobacco/squeeze"
        garage_path = str(garage_home) + "/image_resolution{}/".format(
            str(self.image_resolution[0]).zfill(4)
        )
        #os.mkdir(garage_path)

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False)

        # Prepare a weighted loss function
        loss_function = torch.nn.CrossEntropyLoss(
            weight=tensor(settings.LOSS_WEIGHTS).to(self.device)
        )

        # Prepare the model
        out_channels = len(settings.LOSS_WEIGHTS)
        if not self.continue_model:
            if self.architecture == "slim":
                model = SlimUNet(out_channels)
            elif self.architecture == "squeeze":
                model = SlimSqueezeUNet(out_channels)
                if self.dataset == "cofly":
                    model = SlimSqueezeUNetCofly(out_channels)
                # model = SlimPrunedSqueezeUNet(in_channels, dropout=self.dropout)
            else:
                raise ValueError("Unknown model architecture.")
        else:
            model = torch.load(
                Path(settings.PROJECT_DIR)
                / "segmentation/training/garage/"
                / self.continue_model
            )
        model.to(self.device)

        # Prepare the optimiser
        optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularisation_l2,
        )
        scheduler = self._learning_rate_scheduler(optimizer)
        early_stopper = EarlyStopper(patience=5, verbose=False, path=garage_path+'/checkpoint.pt')
        valid_losses = []
        losses = []
        valid_loss_list = []
        train_loss_list = []
        valid_loss = 0
        train_loss = 0
    
        for epoch in range(self.epochs):
            s = datetime.now()
            model.train()
            for X, y in train_loader:
                # Move to GPU
                X, y = X.to(self.device), y.to(self.device)
                # Reset optimiser
                optimizer.zero_grad()
                # For all set widths
                for width in sorted(self.widths, reverse=True):
                    # Set the current width
                    model.set_width(width)
                    # Forward pass
                    outputs = model.forward(X)
                    # Calculate loss function
                    loss = loss_function(outputs, y)
                    losses.append(loss.item())
                    
                    # Backward pass
                    loss.backward()
                # Update weights
                optimizer.step()
            
            train_loss = np.average(losses)
            train_loss_list.append(train_loss) 
                
            scheduler.step()
            model.eval()
            for X, y in valid_loader:
                X, y = X.to(self.device), y.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to the model
                for width in sorted(self.widths, reverse=True):
                    # Set the current width
                    model.set_width(width)
                    output = model.forward(X)
                    # calculate the loss
                    loss = loss_function(output, y)
                    # record validation loss
                    valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses)
            valid_loss_list.append(valid_loss)
            
            if self.verbose : #and epoch % 10 == 0:
                print(
                    "Epoch {} completed. Running time: {}. Validation loss: {}".format(
                        epoch + 1, datetime.now() - s, valid_loss
                    )
                )
            
            with torch.no_grad():
                metrics = Metricise(device=self.device)
                metrics.evaluate(
                    model,
                    valid_loader,
                    "valid",
                    epoch=0
                )
            res = metrics.report(None)

            if epoch > 5:
                if self._find_best_fitting(res):
                    torch.save(
                        model.state_dict(),
                        garage_path + "model_checkpoint.pt",
                    )
                    print("new best fit")

                early_stopper(valid_loss, model)
                if early_stopper.early_stop:      
                    print("Early stopping....")       
                    break
                    
        model.load_state_dict(torch.load(garage_path + "model_checkpoint.pt"))           
        torch.save(model.state_dict(), garage_path + "tobacco_squeeze_final.pt".format(epoch))


        x = np.arange(0, epoch+1)
        plt.plot(x, valid_loss_list, color='red', label='val_loss')
        plt.plot(x, train_loss_list, color='blue', label='train_loss')
        plt.legend()
        plt.savefig("training_losses_squeeze_snn_tobacco.pdf")


if __name__ == "__main__":
    # Train on GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    architecture = "squeeze"
    for image_resolution, batch_size in zip(
        [(512, 512)],
        [2**3],
    ):
        tr = Training(
            device,
            dataset="tobacco",
            image_resolution=image_resolution,
            architecture=architecture,
            batch_size=batch_size,
            regularisation_l2=0.01,
            tobacco_i=0
        )

        #tr.train()
        tr.test()
