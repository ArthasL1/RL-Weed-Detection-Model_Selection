import os
from datetime import datetime
from time import sleep

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import unique
from torch import argmax, tensor, cat
from torch.cuda import memory_summary, mem_get_info
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image, draw_segmentation_masks
from plotly import graph_objects as go

import segmentation.settings as settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNet
from segmentation.models.slim_unet import SlimUNet


class EvaluationHelper:
    def __init__(
        self,
        run,
        dataset,
        architecture="slim",
        device="cuda:0",
        visualise=False,
    ):
        self.run = run
        self.dataset = dataset
        self.architecture = architecture
        self.device = device
        self.visualise = visualise

        self.test_loader = None
        self.model = None
        self.results = None

    def _import_data(self):
        print("Importing data...", end="")
        ii = ImageImporter(self.dataset, only_test=True, smaller=(128, 128))
        _, test = ii.get_dataset()
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False)
        print("DONE")

    def _import_model(self):
        print("Importing model...", end="")
        runs_dir = f"../training/garage/runs/{self.run}/"
        model_dir = runs_dir + sorted(os.listdir(runs_dir))[-2]

        params = torch.load(model_dir)
        if self.architecture == "slim":
            self.model = SlimUNet(len(settings.LOSS_WEIGHTS))
        elif self.architecture == "squeeze":
            self.model = SlimSqueezeUNet(len(settings.LOSS_WEIGHTS))
        else:
            raise ValueError("Invalid model architecture.")
        self.model.load_state_dict(params)
        print("DONE")

    def evaluate(self):
        """
        Evaluate a given method with requested metrics.
        Optionally also visualise segmentation masks/overlays.
        """
        self._import_data()
        self._import_model()

        i = 0
        with torch.no_grad():
            metrics = Metricise(device=self.device)
            metrics.evaluate(self.model, self.test_loader, "test", -1)
            self.results = metrics.report(False)

            if self.visualise:
                for X, y in self.test_loader:
                    X = X.to(self.device)
                    for width in settings.WIDTHS:
                        self.model.set_width(width)
                        y_pred = self.model.forward(X)
                        self._save_images(X[0], y[0], y_pred[0], i, width)

                    i += 1

    def _save_images(self, X, y, y_pred, batch, width):
        # TODO: This is properly broken. Fix when you have time.
        if not os.path.isdir(f"plots/{self.run}/"):
            os.mkdir(f"plots/{self.run}/")
        org_image = torch.tensor(
            torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8
        )

        # To draw predictions
        mask = argmax(y.clone().detach(), dim=0)
        mask = torch.where(mask == 1, True, False)[None, :, :]
        image = draw_segmentation_masks(org_image, mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(
            f"plots/{self.run}/{str(batch).zfill(3)}_{str(int(width*100)).zfill(3)}_prediction.png"
        )

        # To draw groundtruth
        if width == 1:
            mask = y_pred[1].clone().detach()
            mask = torch.where(mask == 1, True, False)
            image = draw_segmentation_masks(org_image, mask, colors=["red"], alpha=0.5)
            plt.imshow(image.permute(1, 2, 0))
            plt.savefig(f"plots/{self.run}/{str(batch).zfill(3)}_000_groundtruth.png")

    @staticmethod
    def report_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("Total: {}".format(t / 1024 / 1024))
        print("Reserved: {}".format(r / 1024 / 1024))
        print("Allocated: {}".format(a / 1024 / 1024))
        print()


if __name__ == "__main__":
    eh = EvaluationHelper(run=470, dataset="geok", architecture="slim", visualise=False)
    eh.evaluate()
