import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from plotly import graph_objects as go

from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
import segmentation.settings as settings
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNet
from segmentation.models.slim_unet import SlimUNet


class ResultsGenerator:
    def __init__(
        self,
        architecture,
        image_size,
        model_id_list,
        dataset,
        reverse=False,
        verbose=1,
        garage_path=Path("../training/garage/runs"),
    ):
        self.architecture = architecture
        self.image_size = image_size
        self.model_id_list = model_id_list
        self.dataset = dataset
        self.reverse = reverse
        self.verbose = verbose
        self.garage_path = garage_path
        self.available_models = os.listdir(self.garage_path)

        self.test_loader = None
        self.results = []
        self.aggregated_results = None

    def load_data(self, tobacco_i=0):
        ii = ImageImporter(
            self.dataset,
            # Take only the testing data
            only_test=True,
            # Resolution of the images
            smaller=(self.image_size, self.image_size),
            # Testing field used with tobacco dataset.
            tobacco_i=tobacco_i,
        )
        _, testing = ii.get_dataset()
        self.test_loader = DataLoader(testing, batch_size=1, shuffle=False)

    def load_model(self, model_id):
        model_dir = [x for x in self.available_models if model_id in x]
        if len(model_dir) == 0:
            print(f"Model with id {model_id} missing from the garage.")
        if len(model_dir) > 1:
            raise Exception(
                f"Found more than one model with id {model_id}. "
                f"Please check the id list and the garage for inconsistencies."
            )
        models_path = self.garage_path / model_dir[0]
        params = torch.load(models_path / sorted(os.listdir(models_path))[-2])
        if self.architecture == "slim":
            model = SlimUNet(len(settings.LOSS_WEIGHTS))
        elif self.architecture == "squeeze":
            model = SlimSqueezeUNet(len(settings.LOSS_WEIGHTS))
        model.load_state_dict(params)
        return model

    def evaluate(self, model):
        metrics = Metricise()
        metrics.evaluate(model, self.test_loader, "test", 7)
        report = metrics.report(wandb=False)
        # Remove keys not related to weeds.
        report = {key: value for key, value in report.items() if "weeds" in key}
        self.results.append(report)

    def aggregate_results(self):
        results = {
            "architecture": [],
            "size": [],
            "reverse_training": [],
            "metric": [],
            "width": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": [],
        }
        for metric in self.results[0]:
            metric_parts = metric.split("/")
            results["architecture"].append(self.architecture)
            results["size"].append(self.image_size)
            results["reverse_training"].append(self.reverse)
            results["metric"].append(metric_parts[2])
            results["width"].append(metric_parts[1])
            values = [x[metric] for x in self.results]
            results["mean"].append(np.mean(values))
            results["std"].append(np.std(values))
            results["max"].append(np.max(values))
            results["min"].append(np.min(values))
        self.aggregated_results = results

    def report(self):
        for metric in self.results[0]:
            print(metric)
            values = [x[metric] for x in self.results]
            print(np.mean(values))
            print(np.std(values))
            print(np.max(values))
            print(np.min(values))

    def run(self, tobacco_i=0):
        if self.dataset == "geok":
            self.run_geok()
        elif self.dataset == "tobacco":
            self.run_tobacco(tobacco_i=tobacco_i)

    def run_geok(self):
        self.load_data()
        for model_id in self.model_id_list:
            model = self.load_model(model_id)
            model.eval()
            self.evaluate(model)
        self.aggregate_results()
        print()
        if self.verbose:
            self.report()
        # self.plots("iou")
        # self.plots("f1score")
        # self.plots("precision")
        # self.plots("recall")

    def run_tobacco(self, tobacco_i):
        for model_id in self.model_id_list:
            self.load_data(tobacco_i=tobacco_i)
            model = self.load_model(model_id)
            model.eval()
            self.evaluate(model)
        self.aggregate_results()
        if self.verbose:
            self.report()
        self.plots("iou")
        self.plots("f1score")
        self.plots("precision")
        self.plots("recall")

    def plots(self, metric):
        graphs = []
        for width in settings.WIDTHS:
            y = []
            for i in range(len(self.results)):
                y.append(self.results[i][f"test/{int(width * 100)}/{metric}/weeds"])
            graphs.append(
                go.Bar(x=[x for x in range(len(self.results))], y=y, name=width)
            )
        fig = go.Figure(data=graphs)
        fig.update_layout(
            {
                "title": f"{self.architecture} {self.image_size} {metric} {'descending' if self.reverse else 'ascending'}",
                "legend_title_text": "Widths",
                "xaxis_title": "Different networks",
                "yaxis_title": f"{metric}",
                "yaxis_range": [0.5, 1],
            }
        )
        fig.write_image(
            f"plots/{self.architecture}_{self.image_size}_{metric}_{self.reverse}.jpg"
        )
        # fig.show()


if __name__ == "__main__":
    architectures = [
        "slim",
        # "slim",
        # "slim",
        # "squeeze",
        # "squeeze",
        "squeeze",
    ]
    sizes = [
        512,
        # 256,
        # 128,
        512,
        # 256,
        # 128,
    ]
    # Geok dataset
    model_id_lists = [
        # Ascending training. We decided to train the other way around - the same as the authors of slimmable networks
        # do. Those are the ids that are not commented in this list.
        # [str(x) for x in [470, 462, 456, 452, 446, 440, 431, 425, 419, 413]],
        # [str(x) for x in [469, 461, 455, 451, 445, 439, 436, 430, 424, 418]],
        # [str(x) for x in [468, 466, 460, 454, 450, 444, 438, 435, 429, 423]],
        # [str(x) for x in [465, 459, 449, 443, 434, 428, 422, 416]],
        # [str(x) for x in [464, 458, 448, 442, 433, 427, 421, 415]],
        # [str(x) for x in [463, 457, 453, 447, 441, 432, 426, 420, 414]],
        # [str(x) for x in [526, 520, 516, 510, 504, 498, 492, 486, 480, 474]],
        # [str(x) for x in [525, 519, 515, 509, 503, 497, 491, 485, 479, 473]],
        # [str(x) for x in [524, 518, 514, 508, 502, 496, 490, 484, 478, 472]],
        # [str(x) for x in [523, 513, 507, 501, 495, 489, 483, 477]],
        # [str(x) for x in [522, 512, 506, 500, 494, 488, 482, 476]],
        # [str(x) for x in [521, 511, 505, 499, 493, 487, 481, 475]],
        [str(x) for x in [562]],
        [str(x) for x in [561]],
    ]

    # Tobacco dataset
    # architectures = ["slim", "squeeze"]
    # sizes = [512, 512]
    # model_id_lists = [
    #     [str(x) for x in [556]],
    #     [str(x) for x in [555]],
    # ]

    # DATASET = "tobacco"
    DATASET = "geok"

    first = True
    results = None
    for architecture, size, model_id_list in zip(architectures, sizes, model_id_lists):
        print(architecture, size)
        generator = ResultsGenerator(
            architecture=architecture,
            image_size=size,
            model_id_list=model_id_list,
            verbose=0,
            dataset=DATASET,
            reverse=True,
        )
        generator.run(tobacco_i=0)
        if first:
            results = generator.aggregated_results
            first = False
        else:
            for key in results:
                results[key].extend(generator.aggregated_results[key])
    results = DataFrame(results)
    results.to_csv(
        f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", index=False
    )
    print("Results successfully generated.")
