import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


# Load IoU data from JSON files
def load_data(root_dir):
    data = []
    with open(os.path.join(root_dir, 'train_iou_results.json'), 'r') as f:
        dataset_results = json.load(f)
        for item in dataset_results:
            image_path = os.path.join(root_dir, 'train', 'images', item['image'] + '.jpg')
            data.append({
                'image_path': image_path,
                'avg_iou': item['avg_iou'],
                'iou_range': item['iou_range']
            })
    return pd.DataFrame(data)


# Custom Dataset class for loading images and targets
class IoUDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        avg_iou = self.dataframe.iloc[idx, 1]
        iou_range = self.dataframe.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([avg_iou, iou_range], dtype=torch.float)


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


# Normalize the target values
def normalize_targets(df):
    mean = df.mean()
    std = df.std()
    df_normalized = (df - mean) / std
    return df_normalized, mean, std


# Denormalize the target values
def denormalize_targets(normalized, mean, std):
    return normalized * std + mean


# Define root directory
root_dir = '../image_data/geok_cleaned_iou0'

# Load data
data = load_data(root_dir)

# Transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Normalize targets
target_columns = ['avg_iou', 'iou_range']
data[target_columns], target_mean, target_std = normalize_targets(data[target_columns])

# Save normalization parameters
norm_params = {
    'mean': target_mean.tolist(),
    'std': target_std.tolist()
}
with open('norm_params.json', 'w') as f:
    json.dump(norm_params, f)

# Split data into training and validation sets
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets
train_dataset = IoUDataset(train_df, transform=transform)
valid_dataset = IoUDataset(valid_df, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 100
patience = 10
early_stop_count = 0
best_val_loss = float('inf')
best_model_path = 'best_predict_iou_model_CNN.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train_samples = 0

    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)  # Multiply by batch size
        total_train_samples += images.size(0)  # Count the number of samples

    avg_train_loss = running_loss / total_train_samples  # Compute average loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for images, targets in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)  # Multiply by batch size
            total_val_samples += images.size(0)  # Count the number of samples

    avg_val_loss = val_loss / total_val_samples  # Compute average loss
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= patience:
        print("Early stopping triggered")
        break

print("Training completed. Best model saved.")