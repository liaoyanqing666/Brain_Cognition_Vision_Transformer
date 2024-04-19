import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class Sport10DataHandler:
    def __init__(self, train_csv, val_csv, root_dir, batch_size=4, num_workers=4):
        """
        Initialize the data handler for Sport10 dataset.

        Args:
            train_csv (str): Path to the training CSV file.
            val_csv (str): Path to the validation CSV file.
            root_dir (str): Directory containing the images.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of workers for the dataloaders.
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.train_dataset = self._create_dataset(train_csv, transform=self.transform)
        self.val_dataset = self._create_dataset(val_csv, transform=self.transform)
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)

    def _create_dataset(self, csv_file, transform):
        return Sport10Dataset(csv_file=csv_file, root_dir=self.root_dir, transform=transform)

    def _create_dataloader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

class Sport10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
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

# Example usage
if __name__ == "__main__":
    train_csv = '/mnt/data/train_split.csv'
    val_csv = '/mnt_data/val_split.csv'
    root_dir = '/path/to/sport10'
    data_handler = Sport10DataHandler(train_csv, val_csv, root_dir)

    # Use data_handler.train_loader and data_handler.val_loader as needed
