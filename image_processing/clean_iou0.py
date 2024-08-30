import os
import shutil
from run_single_inference import SingleImageInference

# Define source and destination base paths
source_base_path = "../image_data/tobacco_split"
destination_base_path = "../image_data/tobacco_split_cleaned_iou0"

# Define the dataset splits

# splits = ["train", "valid", "test"]
splits = ["train", "test"]


snn_dataset = "tobacco"
snn_interface = SingleImageInference(
    # geok (new dataset)
    dataset=snn_dataset,
    # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
    image_resolution=(
        512,
        512,
    ),
    # slim or squeeze
    model_architecture="squeeze",
    model_path=f"../SNN_models/{snn_dataset}_squeeze_final.pt",
    # Set to a positive integer to select a specific image from the dataset, otherwise random
    fixed_image=22,
    # Do you want to generate a mask/image overlay
    save_image=False,
    # Was segmentation model trained using transfer learning
    is_trans=False,
    # Was segmentation model trained with find_best_fitting (utilising
    # model that has the highest difference in iou between widths
    is_best_fitting=False,
)


# Create the directory structure for the cleaned data
for split in splits:
    os.makedirs(os.path.join(destination_base_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, split, "images", "iou0"), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, split, "labels", "iou0"), exist_ok=True)

# Process each split
for split in splits:
    images_path = os.path.join(source_base_path, split, "images")
    labels_path = os.path.join(source_base_path, split, "labels")
    dest_images_path = os.path.join(destination_base_path, split, "images")
    dest_labels_path = os.path.join(destination_base_path, split, "labels")

    # Iterate over all image files in the current split
    for image_name in os.listdir(images_path):
        if image_name.endswith(".jpg"):  # Ensure only JPG files are processed
            image_path = os.path.join(images_path, image_name)
            label_name = image_name.replace(".jpg", ".txt")
            label_path = os.path.join(labels_path, label_name)
        elif image_name.endswith(".png"):
            image_path = os.path.join(images_path, image_name)
            label_name = image_name
            label_path = os.path.join(labels_path, label_name)

        # Infer IoU from image
        iou = snn_interface.infer_from_rl(image_path, 0.25, snn_dataset)["test/iou/weeds"]

        # Determine destination paths
        if iou == 0:
            dest_image_subfolder = "iou0"
            dest_label_subfolder = "iou0"
        else:
            dest_image_subfolder = ""
            dest_label_subfolder = ""

        dest_image_path = os.path.join(dest_images_path, dest_image_subfolder, image_name)
        dest_label_path = os.path.join(dest_labels_path, dest_label_subfolder, label_name)

        # Copy the image and label to the appropriate destination
        shutil.copy(image_path, dest_image_path)
        if os.path.exists(label_path):  # Ensure the label file exists
            shutil.copy(label_path, dest_label_path)
