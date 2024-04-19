import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# Load the first few rows of the train_split.csv file
train_split_path = '/mnt/data/train_split.csv'
train_split_df = pd.read_csv(train_split_path)

# Load the first few rows of the val_split.csv file
val_split_path = '/mnt/data/val_split.csv'
val_split_df = pd.read_csv(val_split_path)

class Sport10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image_class = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, image_class

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Assuming the root directory for images is called 'sport10'
root_dir = '/path/to/sport10'

# Create dataset instances
train_dataset = Sport10Dataset(csv_file=train_split_path, root_dir=root_dir, transform=transform)
val_dataset = Sport10Dataset(csv_file=val_split_path, root_dir=root_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
