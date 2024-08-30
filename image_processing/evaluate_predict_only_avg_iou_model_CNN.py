import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms


# Custom Dataset class for loading images
class IoUDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        if self.transform:
            image = self.transform(image)
        return image, image_path


# Simple CNN model to predict avg_iou
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x





# Denormalize the target values
def denormalize_targets(normalized, mean, std):
    return normalized * std + mean




# Function to load IoU results from JSON files
def load_iou_results(dataset_name, root_dir='../image_data/geok_cleaned_iou0'):
    json_file = os.path.join(root_dir, f'{dataset_name}_iou_results.json')
    with open(json_file, 'r') as f:
        iou_results = json.load(f)
    return {item['image']: item for item in iou_results}


# Function to predict the avg_iou for a single image
def predict_avg_iou(image_path, iou_model, norm_params_path):
    iou_model.eval()
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 512x512
        transforms.ToTensor(),  # Convert image to tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        output = iou_model(image).squeeze()  # Squeeze to get scalar value

    # Load normalization parameters
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    target_mean = np.array(norm_params['mean'])
    target_std = np.array(norm_params['std'])

    # Denormalize the prediction
    denormalized_output = denormalize_targets(output.item(), target_mean, target_std)

    return denormalized_output

if __name__ == "__main__":

    # Load normalization parameters
    with open('norm_params_only_avg_iou.json', 'r') as f:
        norm_params = json.load(f)
    target_mean = np.array(norm_params['mean'])
    target_std = np.array(norm_params['std'])

    # Load the trained model
    model = CNNModel()
    model.load_state_dict(torch.load('best_predict_only_avg_iou_model_CNN.pth'))
    model.eval()

    # Transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load data
    datasets = ['train', 'valid', 'test']
    root_dir = '../image_data/geok_cleaned_iou0'
    criterion = nn.MSELoss()

    for dataset in datasets:
        image_paths = []
        labels = load_iou_results(dataset, root_dir)
        images_dir = os.path.join(root_dir, dataset, 'images')

        # Get all image paths
        for image_file in os.listdir(images_dir):
            if image_file.endswith('.jpg'):
                image_paths.append(os.path.join(images_dir, image_file))

        # Randomly select 20 images
        selected_images = random.sample(image_paths, 20)
        datasetL = IoUDataset(selected_images, transform=transform)
        dataloader = DataLoader(datasetL, batch_size=1, shuffle=False)

        # Store predictions and actual values
        all_predictions = []
        all_actuals = []
        all_denormalized_predictions = []

        print(f"\nEvaluating on {dataset} set:")

        for images, image_paths in dataloader:
            with torch.no_grad():
                output = model(images).squeeze()  # Squeeze to ensure the output shape is a scalar

            # Denormalize prediction
            denormalized_output = denormalize_targets(output.item(), target_mean, target_std)

            # Extract the image name correctly
            image_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
            actual = labels[image_name]
            predicted_avg_iou = denormalized_output
            actual_avg_iou = actual['avg_iou']

            # Print predictions and actual values
            print(f"Image: {image_paths[0]}")
            print(f"Predicted avg_iou: {predicted_avg_iou:.4f}")
            print(f"Actual avg_iou: {actual_avg_iou:.4f}")

            # Store for MSE loss calculation
            all_predictions.append(output.item())  # Save the non-denormalized output for MSE calculation
            all_denormalized_predictions.append(denormalized_output)
            all_actuals.append(actual_avg_iou)

        # Convert to numpy arrays for MSE loss calculation
        all_predictions = np.array(all_predictions)
        all_denormalized_predictions = np.array(all_denormalized_predictions)
        all_actuals = np.array(all_actuals)

        # Normalize actual values for MSE loss comparison
        all_actuals_normalized = (all_actuals - target_mean) / target_std

        # Calculate MSE loss using predictions before denormalization and actual values after normalization
        mse_loss_normalized = criterion(torch.tensor(all_predictions), torch.tensor(all_actuals_normalized))
        print(f"\nMSE Loss (using predictions before denormalization and actual values after normalization) for "
              f"{dataset} set: {mse_loss_normalized.item():.4f}\n")

        # Calculate MSE loss using predictions after denormalization and actual values before normalization
        mse_loss_denormalized = criterion(torch.tensor(all_denormalized_predictions), torch.tensor(all_actuals))
        print(f"MSE Loss (using predictions after denormalization and actual values before normalization) for "
              f"{dataset} set: {mse_loss_denormalized.item():.4f}\n")


    print(np.float32(predict_avg_iou("../image_data/geok_cleaned_iou0/test/images/"
                                     "frame_0116_jpg.rf.30f0b85b04aaf0c5c0aafe20ea6632a2.jpg", model,
                                     "norm_params_only_avg_iou.json")))

