import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_and_copy_images_masks(img_dir, mask_dir, output_dir="../image_data/tobacco_split"):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    if len(images) != len(masks):
        raise ValueError("The number of images and masks must be the same.")

    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, random_state=42, train_size=0.8
    )

    output_dir = Path(output_dir)
    train_images_dir = output_dir / 'train' / 'images'
    train_labels_dir = output_dir / 'train' / 'labels'
    test_images_dir = output_dir / 'test' / 'images'
    test_labels_dir = output_dir / 'test' / 'labels'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_name, mask_name in zip(X_train, y_train):
        shutil.copy(img_dir / image_name, train_images_dir / image_name)
        shutil.copy(mask_dir / mask_name, train_labels_dir / mask_name)

    for image_name, mask_name in zip(X_test, y_test):
        shutil.copy(img_dir / image_name, test_images_dir / image_name)
        shutil.copy(mask_dir / mask_name, test_labels_dir / mask_name)

    print(f"Files have been successfully copied to {output_dir}")


if __name__ == "__main__":
    split_and_copy_images_masks("../image_data/tobacco/images", '../image_data/tobacco/labels')
