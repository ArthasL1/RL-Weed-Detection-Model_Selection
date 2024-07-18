import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import cv2
from numpy import floor, ceil
from sklearn.model_selection import train_test_split
from torch import Generator, tensor, argmax, ones, zeros, cat, unique, flatten
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from shapely import Polygon, Point

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

import agriadapt.segmentation.settings as settings


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class ImageImporter:
    def __init__(
        self,
        dataset,
        sample=False,
        validation=False,
        smaller=False,
        only_training=False,
        only_test=False,
        tobacco_i=0,
        augmentations=None,
    ):
        assert dataset in ["agriadapt", "cofly", "infest", "geok", "tobacco"]
        self._dataset = dataset
        # Reduced number of random images in training if set.
        self.sample = sample
        # If True, return validation instead of testing set (where applicable)
        self.validation = validation
        # Make the images smaller
        self.smaller = smaller
        # Only return the train dataset (second part of the returned tuple is empty)
        self.only_training = only_training
        # Only return the valid/test dataset (first part of the returned tuple is empty)
        self.only_test = only_test
        # Only used for the tobacco dataset. It denotes which of the fields is used as validation/testing.
        self.tobacco_i = tobacco_i

        self.project_path = Path(settings.PROJECT_DIR)

    def get_dataset(self):
        if self._dataset == "agriadapt":
            return self._get_agriadapt()
        elif self._dataset == "cofly":
            return self._get_cofly()
        elif self._dataset == "infest":
            return self._get_infest()
        elif self._dataset == "geok":
            return self._get_geok()
        elif self._dataset == "tobacco":
            return self._get_tobacco()

    def _get_agriadapt(self):
        """
        This method only returns raw images as there are no labelled masks for this data for now.
        NOTE: There's a lot of images, if we don't batch import this, RAM will not be happy.
        """
        images = sorted(
            os.listdir(self.project_path / "data/agriadapt/UAV_IMG/UAV_IMG/")
        )
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize((1280, 720))
        X, y = [], []
        for file_name in images:
            tens = smaller(
                create_tensor(
                    Image.open(
                        self.project_path
                        / "data/agriadapt/UAV_IMG/UAV_IMG/"
                        / file_name
                    )
                )
            )
            X.append(tens)
            y.append(tens)
        return ImageDataset(X, y)

    def _get_cofly(self):
        """
        Import images and their belonging segmentation masks (one-hot encoded).
        """
        images = sorted(
            os.listdir(self.project_path / "segmentation/data/cofly/images/images/")
        )
        random.seed(42069)
        idx = [x for x in range(len(images))]
        random.shuffle(idx)
        cut = int(len(images) * 0.8)
        train_images = [images[x] for x in idx[:cut]]
        test_images = [images[x] for x in idx[cut:]]
        return self._get_cofly_train(train_images), self._get_cofly_test(test_images)

    def _get_cofly_train(self, images):
        create_tensor = transforms.ToTensor()
        if self.smaller:
            smaller = transforms.Resize(self.smaller)

        X, y = [], []

        for file_name in images:
            img = create_tensor(
                Image.open(
                    self.project_path
                    / "segmentation/data/cofly/images/images/"
                    / file_name
                )
            )
            if self.smaller:
                img = smaller(img)

            # Data augmentation
            imgh = transforms.RandomHorizontalFlip(p=1)(img)
            imgv = transforms.RandomVerticalFlip(p=1)(img)
            imghv = transforms.RandomVerticalFlip(p=1)(imgh)

            X.append(img)
            X.append(imgh)
            X.append(imgv)
            X.append(imghv)

            # Open the mask
            mask = Image.open(
                self.project_path / "segmentation/data/cofly/labels/labels/" / file_name
            )
            if self.smaller:
                mask = smaller(mask)
            mask = tensor(np.array(mask))
            # Merge weeds classes to a single weeds class
            mask = torch.where(
                mask > 0,
                1,
                0,
            )

            maskh = transforms.RandomHorizontalFlip(p=1)(mask)
            maskv = transforms.RandomVerticalFlip(p=1)(mask)
            maskhv = transforms.RandomVerticalFlip(p=1)(maskh)
            y.append(self._cofly_prep_mask(mask))
            y.append(self._cofly_prep_mask(maskh))
            y.append(self._cofly_prep_mask(maskv))
            y.append(self._cofly_prep_mask(maskhv))

        return ImageDataset(X, y)

    def _get_cofly_test(self, images):
        create_tensor = transforms.ToTensor()
        if self.smaller:
            smaller = transforms.Resize(self.smaller)

        X, y = [], []

        for file_name in images:
            img = create_tensor(
                Image.open(
                    self.project_path
                    / "segmentation/data/cofly/images/images/"
                    / file_name
                )
            )
            if self.smaller:
                img = smaller(img)

            X.append(img)

            # Open the mask
            mask = Image.open(
                self.project_path / "segmentation/data/cofly/labels/labels/" / file_name
            )
            if self.smaller:
                mask = smaller(mask)
            mask = tensor(np.array(mask))
            # Merge weeds classes to a single weeds class
            mask = torch.where(
                mask > 0,
                1,
                0,
            )
            y.append(self._cofly_prep_mask(mask))

        return ImageDataset(X, y)

    @staticmethod
    def tensor_to_image(tensor_images):
        images = []
        for elem in tensor_images:
            elem = (elem.numpy() * 255).astype(np.uint8)
            elem = elem.transpose(1, 2, 0)
            image = cv2.cvtColor(elem, cv2.COLOR_RGB2BGR)
            images.append(image)
        return images

    def _cofly_prep_mask(self, mask):
        return (
            torch.nn.functional.one_hot(
                mask,
                num_classes=2,
            )
            .permute(2, 0, 1)
            .float()
        )

    def _get_infest(self):
        """
        Import images and convert labels coordinates to actual masks and return a train/test split datasets.
        There are two classes in the segmentation mask labels:
        0 -> weeds
        1 -> lettuce
        The indices of the segmentation mask are 1 and 2 respectively.
        Therefore, we create a 3-channel segmentation mask that separately recognises both weeds and lettuce.
        mask[0] -> background
        mask[1] -> weeds
        mask[2] -> lettuce
        """
        if self.validation:
            return self._fetch_infest_split(split="train"), self._fetch_infest_split(
                split="valid"
            )
        else:
            if self.only_test:
                return None, self._fetch_infest_split(split="test")
            else:
                return self._fetch_infest_split(
                    split="train"
                ), self._fetch_infest_split(split="test")

    def _fetch_infest_split(
        self,
        data_dir="segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/",
        split="train",
    ):
        images = sorted(os.listdir(self.project_path / data_dir / split / "images/"))
        create_tensor = transforms.ToTensor()
        X, y = [], []

        if self.sample and split == "train":
            images = random.sample(images, self.sample)

        for file_name in images:
            img = Image.open(
                self.project_path / data_dir / split / "images/" / file_name
            )
            if self.smaller:
                smaller = transforms.Resize(self.smaller)
                img = smaller(img)
            tens = create_tensor(img)
            X.append(tens)
            image_width = tens.shape[1]
            image_height = tens.shape[2]

            # Constructing the segmentation mask
            # We init the whole tensor as background
            # TODO: if we do transfer learning from cofly, we need the background and weeds masks (no lettuce)
            # That means that we have 1 as the first argument of zeros (as we only have weeds -- no lettuce)
            mask = cat(
                (
                    ones(1, image_width, image_height),
                    zeros(2, image_width, image_height),
                ),
                0,
            )
            # Then, label by label, add to other classes and remove from background.
            file_name = file_name[:-3] + "txt"
            with open(
                self.project_path / data_dir / split / "labels/" / file_name
            ) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(
                        label, image_width, image_height
                    )
                    if class_id > 1:
                        continue
                    # TODO: another thing to keep in mind with transfer learning from cofly
                    # Change values based on received pixels
                    print(class_id)
                    for pixel in pixels:
                        mask[0][pixel[0]][pixel[1]] = 0
                        mask[class_id][pixel[0]][pixel[1]] = 1
            y.append(mask)

        return ImageDataset(X, y)

    def _get_geok(self):
        """
        Retrieve the geok dataset with background and weeds labels.
        """
        data_dir = "segmentation/data/geok/"
        test_set_name = "valid" if self.validation else "test"
        if self.only_training:
            return self._fetch_geok_split(split="train", data_dir=data_dir), None
        elif self.only_test:
            return None, self._fetch_geok_split(split=test_set_name, data_dir=data_dir)
        else:
            return self._fetch_geok_split(
                split="train", data_dir=data_dir
            ), self._fetch_geok_split(split=test_set_name, data_dir=data_dir)

    def _fetch_geok_split(
        self,
        data_dir,
        split,
    ):
        images = sorted(os.listdir(self.project_path / data_dir / split / "images/"))
        create_tensor = transforms.ToTensor()
        X, y = [], []

        if self.sample and split == "train":
            images = random.sample(images, self.sample)

        for file_name in images:
            img = Image.open(
                self.project_path / data_dir / split / "images/" / file_name
            )
            if self.smaller:
                smaller = transforms.Resize(self.smaller)
                img = smaller(img)
            tens = create_tensor(img)
            image_width = tens.shape[1]
            image_height = tens.shape[2]

            # Constructing the segmentation mask
            # We init the whole tensor as background
            # That means that we have 1 as the first argument of zeros (as we only have weeds -- no lettuce)
            mask = cat(
                (
                    ones(1, image_width, image_height),
                    zeros(1, image_width, image_height),
                ),
                0,
            )
            # Then, label by label, add to other classes and remove from background.
            file_name = file_name[:-3] + "txt"
            with open(
                self.project_path / data_dir / split / "labels/" / file_name
            ) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(
                        label, image_width, image_height
                    )
                    if not pixels:
                        continue
                    for pixel in pixels:
                        mask[0][pixel[0]][pixel[1]] = 0
                        mask[class_id][pixel[0]][pixel[1]] = 1

            X.append(tens)
            y.append(mask)

        return ImageDataset(X, y)

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

    def _get_tobacco(self):
        field_directory = (
            self.project_path
            / "segmentation/data/tobacco/Tobacco Aerial Dataset V1/Ready for traintest tobacco data 352x480"
        )
        fields = [
            "119",
            "120/test/RGB",
            "133/test/RGB",
            "134/test/RGB",
            "147",
            "154/test/RGB",
            "163/test/RGB",
            "171/test/RGB",
        ]

        X, y = self._fetch_tobacco_split(
            field_directory / fields[self.tobacco_i], [], []
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, train_size=0.8
        )

        return ImageDataset(X_train, y_train), ImageDataset(X_test, y_test)

    def _get_tobacco_per_field(self):
        """
        Retrieve data from all tobacco fields. One field is used as a test, others to train the models.
        """
        field_directory = (
            self.project_path
            / "segmentation/data/tobacco/Tobacco Aerial Dataset V1/Ready for traintest tobacco data 352x480"
        )
        fields = [
            "119",
            "120/test/RGB",
            "133/test/RGB",
            "134/test/RGB",
            "147",
            "154/test/RGB",
            "163/test/RGB",
            "171/test/RGB",
        ]
        if not self.only_test:
            X_train, y_train = [], []
            for i, field in enumerate(fields):
                if i == self.tobacco_i:
                    continue
                self._fetch_tobacco_split(field_directory / field, X_train, y_train)
            train = ImageDataset(X_train, y_train)
        else:
            train = None

        X_test, y_test = self._fetch_tobacco_split(
            field_directory / fields[self.tobacco_i], [], []
        )
        test = ImageDataset(X_test, y_test)

        return train, test

    def _fetch_tobacco_split(self, dir, X, y):
        img_dir = dir / "data"
        mask_dir = dir / "maskref"
        images = sorted(
            os.listdir(
                img_dir,
            )
        )
        if self.sample:
            images = random.sample(images, self.sample)
        masks = sorted(os.listdir(mask_dir))
        create_tensor = transforms.ToTensor()
        for image, mask in zip(images, masks):
            if image != mask and not self.sample:
                raise ValueError("Image and mask name do not match.")
            img = create_tensor(Image.open(img_dir / image))
            # We get values 0, 127, and 256. We transform them to 0, 1, 2 (background, tobacco, weeds)
            msk = torch.round(create_tensor(Image.open(mask_dir / image)) * 2)
            if self.smaller:
                smaller = transforms.Resize(self.smaller)
                img = smaller(img)
                # Have to round again, as this is technically an image.
                msk = torch.round(smaller(msk))
            msk = self._construct_tobacco_mask(msk)
            X.append(img)
            y.append(msk)
        return X, y

    def _construct_tobacco_mask(self, mask_class):
        """
        We have three different classes -> background (0), tobacco (1), and weeds (2).
        Therefore, we need to construct a 3-channel binary mask for each class category.
        Alternatively we can only create a 2-channel one, counting tobacco as the background (this is currently implemented).
        """
        width, height = mask_class.shape[1:3]
        mask = cat(
            (
                ones(1, width, height),
                zeros(1, width, height),
            ),
            0,
        )
        # Then, label by label, add to other classes and remove from background.
        for x in range(width):
            for y in range(height):
                if mask_class[0][x][y] == 2:
                    mask[0][x][y] = 0
                    mask[1][x][y] = 1
        return mask


if __name__ == "__main__":
    ii = ImageImporter("geok", sample=1, smaller=(512, 512), only_training=True)
    train, test = ii.get_dataset()

    for X, y in iter(train):
        x_mask = torch.tensor(torch.mul(X, 255), dtype=torch.uint8)
        weeds_mask = torch.tensor(y[1], dtype=torch.bool)
        image = draw_segmentation_masks(
            x_mask,
            weeds_mask,
            colors=["red"],
            alpha=0.5,
        )
        # plt.imshow(X.permute(1, 2, 0))
        # plt.show()
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        0 / 0

    # ii = ImageImporter("cofly")
    # train, test = ii.get_dataset()
    # X, y = next(iter(train))
    # y = argmax(y, dim=0)
    # plt.imshow(y)
    # plt.show()
