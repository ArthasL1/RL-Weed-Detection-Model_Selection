import os
from pathlib import Path
from random import randint

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from agriadapt.segmentation import settings
from agriadapt.segmentation.data.data import ImageImporter
from agriadapt.segmentation.helpers.metricise import Metricise
from agriadapt.segmentation.models.slim_squeeze_unet import SlimSqueezeUNetCofly, SlimSqueezeUNet
from agriadapt.segmentation.models.slim_unet import SlimUNet
from shapely import Polygon, Point
from numpy import floor, ceil


class SingleImageInference:
    def __init__(
        self,
        dataset,
        image_resolution,
        model_architecture,
        model_path,
        results_save_path=None,
        fixed_image=-1,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        #self.project_path = Path(settings.PROJECT_DIR)
        #self.project_path = "D:\\Desktop\\GithubClone\\agriadapt-develop\\"
        self.project_path = "./"
        self.model_path = model_path
        if dataset == "infest":
            self.image_dir = "agriadapt/segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        elif dataset == "geok":
            self.image_dir = "agriadapt/segmentation/data/geok/test/images/"
        else:
            raise ValueError("Invalid dataset selected.")
        assert model_architecture in ["slim", "squeeze"]
        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
        #model_key = f"{dataset}_{model_architecture}_{image_resolution[0]}"
        model_key = "geok_slim_final"  # Zeqi:
        if is_trans:
            model_key += "_trans"
        if is_best_fitting:
            model_key += "_opt"
        if model_architecture == "slim":
            self.model = SlimUNet(out_channels=2)
        elif dataset == "cofly" or is_trans:
            self.model = SlimSqueezeUNetCofly(out_channels=2)
        elif dataset == "geok":
            self.model = SlimSqueezeUNet(out_channels=2)

        # self.model.load_state_dict(
        #     torch.load(
        #         Path(settings.PROJECT_DIR)
        #         / f"segmentation/training/garage/{model_key}.pt"
        #     )
        # )
        # Zeqi:
        # self.model.load_state_dict(
        #     torch.load(
        #         self.project_path + f"agriadapt/segmentation/training/garage/{model_key}.pt"
        #     )
        # )
        self.model.load_state_dict(
            torch.load(
                self.project_path + self.model_path
            )
        )

        self.fixed_image = fixed_image
        self.save_image = save_image
        #self.adaptive_width = AdaptiveWidth(model_key)
        self.tensor_to_image = ImageImporter(dataset).tensor_to_image
        self.random_image_index = -1

    def _get_random_image_path(self):
        images = os.listdir(self.project_path + self.image_dir)
        if self.fixed_image < 0:
            self.random_image_index = randint(0, len(images) - 1)
            return images[self.random_image_index]
        else:
            return images[
                self.fixed_image if self.fixed_image < len(images) else len(images) - 1
            ]

    # def _yolov7_label(self, label, image_width, image_height):
    #     """
    #     Implement an image mask generation according to this:
    #     https://roboflow.com/formats/yolov7-pytorch-txt
    #     """
    #     # Deconstruct a row
    #     print(f"Label:", label)
    #     class_id, center_x, center_y, width, height = [
    #         float(x) for x in label.split(" ")
    #     ]
    #
    #     # Get center pixel
    #     center_x = center_x * image_width
    #     center_y = center_y * image_height
    #
    #     # Get border pixels
    #     top_border = int(center_x - (width / 2 * image_width))
    #     bottom_border = int(center_x + (width / 2 * image_width))
    #     left_border = int(center_y - (height / 2 * image_height))
    #     right_border = int(center_y + (height / 2 * image_height))
    #
    #     # Generate pixels
    #     pixels = []
    #     for x in range(left_border, right_border):
    #         for y in range(top_border, bottom_border):
    #             pixels.append((x, y))
    #
    #     return int(class_id), pixels

    def _yolov7_label(self, label, image_width, image_height):
        """
        Implement an image mask generation according to this:
        https://roboflow.com/formats/yolov7-pytorch-txt
        """
        # Deconstruct a row

        label = label.split(" ")
        # We consider lettuce as the background, so we skip lettuce label extraction (for now at least).
        if label[0] == "0":
            return None, None
        # Some labels are in a rectangle format, while others are presented as polygons... great fun.
        # Rectangles
        if len(label) == 5:
            class_id, center_x, center_y, width, height = [float(x) for x in label]

            # Get center pixel
            center_x = center_x * image_width
            center_y = center_y * image_height

            # Get border pixels
            top_border = int(center_x - (width / 2 * image_width))
            bottom_border = int(center_x + (width / 2 * image_width))
            left_border = int(center_y - (height / 2 * image_height))
            right_border = int(center_y + (height / 2 * image_height))

            # Generate pixels
            pixels = []
            for x in range(left_border, right_border):
                for y in range(top_border, bottom_border):
                    pixels.append((x, y))
        # Polygons
        else:
            class_id = label[0]
            # Create a polygon object
            points = [
                (float(label[i]) * image_width, float(label[i + 1]) * image_height)
                for i in range(1, len(label), 2)
            ]
            poly = Polygon(points)
            # We limit the area in which we search for points to make the process a tiny bit faster.
            pixels = []
            for x in range(
                int(floor(min([x[1] for x in points]))),
                int(ceil(max([x[1] for x in points]))),
            ):
                for y in range(
                    int(floor(min([x[0] for x in points]))),
                    int(ceil(max([x[0] for x in points]))),
                ):
                    if Point(y, x).within(poly):
                        pixels.append((x, y))

        return int(class_id), pixels



    def _get_single_image(self):
        file_name = self._get_random_image_path()
        print(f"Image name: {file_name} \nImage Index: {self.random_image_index}")
        img = Image.open(self.project_path + self.image_dir + file_name)
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        # Constructing the segmentation mask
        # We init the whole tensor as the background
        mask = torch.cat(
            (
                torch.ones(1, image_width, image_height),
                torch.zeros(1, image_width, image_height),
            ),
            0,
        )
        # Then, label by label, add to other classes and remove from background.
        file_name = file_name[:-3] + "txt"
        with open(
            self.project_path + self.image_dir.replace("images", "labels") + file_name
        ) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = self._yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to("cuda:0")
        mask = mask.to("cuda:0")
        img = img[None, :]
        mask = mask[None, :]

        return img, mask

    def _generate_images(self, X, y, y_pred):
        if not os.path.exists("results"):
            os.mkdir("results")
        # Generate an original rgb image with predicted mask overlay.
        x_mask = torch.tensor(
            torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8
        )
        x_mask = x_mask[0]

        # Draw predictions
        y_pred = y_pred[0]
        mask = torch.argmax(y_pred.clone().detach(), dim=0)
        weed_mask = torch.where(mask == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)

        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_pred.jpg")

        # Draw ground truth
        mask = y.clone().detach()[0]
        weed_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask[2] == 1, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)
        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_true.jpg")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def infer(self, width=0.25):
        # Get a random single image from test dataset.
        # Set the fixed attribute to always obtain the same image
        image, mask = self._get_single_image()

        # Select and set the model width
        # width = self.adaptive_width.get_image_width(
        #     self.tensor_to_image(image.cpu())[0]
        # )
        # width = {"knn": 0.5}
        # print(f"Model width: {width}")
        self.model.set_width(width)

        # Get a prediction
        y_pred = self.model.forward(image)
        #print(y_pred.shape)

        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # Generate overlayed segmentation masks (ground truth and prediction)
        if self.save_image:
            self._generate_images(image, mask, y_pred)

        return results

    def infer_from_rl(self, image_path, width=0.25):
        image, mask = self._get_single_image_from_rl(image_path)
        self.model.set_width(width)
        y_pred = self.model.forward(image)
        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)
        return results

    def _get_single_image_from_rl(self, image_path):
        img = Image.open(image_path)
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        # Constructing the segmentation mask
        # We init the whole tensor as the background
        mask = torch.cat(
            (
                torch.ones(1, image_width, image_height),
                torch.zeros(1, image_width, image_height),
            ),
            0,
        )
        # Then, label by label, add to other classes and remove from background.
        label_path = image_path[:-3].replace("images", "labels") + "txt"
        with open(
            label_path
        ) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = self._yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to("cuda:0")
        mask = mask.to("cuda:0")
        img = img[None, :]
        mask = mask[None, :]

        return img, mask


if __name__ == "__main__":
    # Run this once to download the new dataset
    # setup_env()

    si = SingleImageInference(
        # geok (new dataset)
        # dataset="geok",
        dataset="geok",
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            512,
            512,
        ),
        # slim or squeeze
        model_architecture="slim",
        model_path="SNN_models/geok_slim_final.pt",
        # Set to a positive integer to select a specific image from the dataset, otherwise random
        fixed_image=22,
        # Do you want to generate a mask/image overlay
        save_image=True,
        # Was segmentation model trained using transfer learning
        is_trans=False,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )
    # for i in range(5):
    #     print(i)
    #     try:
    #         results = si.infer()
    #         print(results)
    #     except:
    #         print("error")

    # results = si.infer(width=0.25)
    # print("0.25", results)
    # results = si.infer(width=0.5)
    # print("0.5", results)
    # results = si.infer(width=0.75)
    # print("0.75", results)
    # results = si.infer(width=1.0)
    # print("1.0", results)


    results = si.infer_from_rl(image_path="./image_data/geok_grouped/valid\images\group_5\image07.jpg", width=0.25)
    print("0.25", results)
    results = si.infer_from_rl(image_path="./image_data/geok_grouped/valid\images\group_5\image07.jpg",width=0.5)
    print("0.5", results)
    results = si.infer_from_rl(image_path="./image_data/geok_grouped/valid\images\group_5\image07.jpg",width=0.75)
    print("0.75", results)
    results = si.infer_from_rl(image_path="./image_data/geok_grouped/valid\images\group_5\image07.jpg",width=1.0)
    print("1.0", results)
