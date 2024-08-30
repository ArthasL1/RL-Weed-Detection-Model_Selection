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


# Simple CNN model to predict avg_iou and iou_range
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load normalization parameters
with open('norm_params.json', 'r') as f:
    norm_params = json.load(f)
target_mean = np.array(norm_params['mean'])
target_std = np.array(norm_params['std'])


# Denormalize the target values
def denormalize_targets(normalized, mean, std):
    return normalized * std + mean


# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('best_predict_iou_model_CNN.pth'))
model.eval()

# Transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])


# Function to load IoU results from JSON files
def load_iou_results(dataset_name, root_dir):
    json_file = os.path.join(root_dir, f'{dataset_name}_iou_results.json')
    with open(json_file, 'r') as f:
        iou_results = json.load(f)
    return {item['image']: item for item in iou_results}


# Load data
datasets = ['train', 'valid', 'test']
root_dir = '../image_data/geok_cleaned_iou0'
criterion = nn.MSELoss()

for dataset_name in datasets:
    image_paths = []
    labels = load_iou_results(dataset_name, root_dir)
    images_dir = os.path.join(root_dir, dataset_name, 'images')

    # Get all image paths
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.jpg'):
            image_paths.append(os.path.join(images_dir, image_file))

    # Randomly select 20 images
    selected_images = random.sample(image_paths, 20)
    dataset = IoUDataset(selected_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Store predictions and actual values
    all_predictions = []
    all_actuals = []

    print(f"\nEvaluating on {dataset_name} set:")

    for images, image_paths in dataloader:
        with torch.no_grad():
            outputs = model(images)

        # Denormalize predictions
        denormalized_outputs = denormalize_targets(outputs, target_mean, target_std)
        for output, image_path in zip(denormalized_outputs, image_paths):
            #image_name = os.path.basename(image_path).split('.')[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            actual = labels[image_name]
            predicted_avg_iou, predicted_iou_range = output.detach().numpy()
            actual_avg_iou = actual['avg_iou']
            actual_iou_range = actual['iou_range']

            # Print predictions and actual values
            print(f"Image: {image_path}")
            print(f"Predicted avg_iou: {predicted_avg_iou:.4f}, Predicted iou_range: {predicted_iou_range:.4f}")
            print(f"Actual avg_iou: {actual_avg_iou:.4f}, Actual iou_range: {actual_iou_range:.4f}")

            # Store for MSE loss calculation
            all_predictions.append(outputs.numpy())
            all_actuals.append([actual_avg_iou, actual_iou_range])

    # Convert to numpy arrays for MSE loss calculation
    all_predictions = np.array(all_predictions).squeeze()
    all_actuals = np.array(all_actuals)

    # Normalize actual values for MSE loss comparison
    all_actuals_normalized = (all_actuals - target_mean) / target_std

    # Calculate MSE loss
    mse_loss = criterion(torch.tensor(all_predictions), torch.tensor(all_actuals_normalized))
    print(f"\nMSE Loss for {dataset_name} set: {mse_loss.item():.4f}\n")
