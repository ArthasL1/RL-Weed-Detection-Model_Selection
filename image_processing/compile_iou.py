import os
import json
from run_single_inference import SingleImageInference
import numpy as np

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
    model_architecture="slim",
    model_path=f"../SNN_models/{snn_dataset}_slim_final.pt",
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

# Define the widths to process
widths = [0.25, 0.50, 0.75, 1.00]

# Root directory
root_dir = '../image_data/tobacco_split_cleaned_iou0'

# Process each of the train, valid, test directories
for dataset in ['train', 'valid', 'test']:
    dataset_path = os.path.join(root_dir, dataset)
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    results = []

    # Iterate over each image file in the images directory
    for image_file in os.listdir(images_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_path, image_file)
            image_name = os.path.splitext(image_file)[0]
            label_path = os.path.join(labels_path, image_name + '.txt')
        elif image_file.endswith(".png"):
            image_path = os.path.join(images_path, image_file)
            image_name = os.path.splitext(image_file)[0]
            label_name = image_name + '.png'
            label_path = os.path.join(labels_path, label_name)
        else:
            continue

        # Check if the corresponding label file exists
        if os.path.exists(label_path):
            ious = []
            # Compute IoU for each width
            for width in widths:
                iou = snn_interface.infer_from_rl(image_path, width, snn_dataset)["test/iou/weeds"]
                ious.append(float(iou))  # Convert to standard float

            # Calculate average IoU and the range (max - min)
            avg_iou = float(np.mean(ious))
            iou_range = float(np.max(ious) - np.min(ious))

            # Store the results
            results.append({
                'image': image_name,
                'ious': {str(width): float(iou) for width, iou in zip(widths, ious)},
                'avg_iou': avg_iou,
                'iou_range': iou_range
            })

    # Save results to a JSON file
    with open(os.path.join(root_dir, f'{dataset}_slim_iou_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
