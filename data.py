import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

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


# https://cs.stanford.edu/~acoates/stl10/
# DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
class STL10Dataset(Dataset):
    def __init__(self, root_dir,X_path='train_X.bin',y_path='train_y.bin', transform=None):
        """
        Args:
            root_dir (string): 图片数据集的根目录路径
            transform (callable, optional): 数据预处理操作
        """
        self.root_dir = root_dir
        self.transform = transform
        # 读取数据文件
        with open(os.path.join(root_dir, X_path), 'rb') as f:
            self.images = np.fromfile(f, dtype=np.uint8)
            self.images = np.reshape(self.images, (-1, 3, 96, 96))
            self.images = np.transpose(self.images, (0, 2, 3, 1))  # 调整维度顺序，从(C, H, W)到(H, W, C)
        with open(os.path.join(root_dir, y_path), 'rb') as f:
            self.labels = np.fromfile(f, dtype=np.uint8)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # Load the first few rows of the train_split.csv file
    train_split_path = '/mnt/data/train_split.csv'
    train_split_df = pd.read_csv(train_split_path)

    # Load the first few rows of the val_split.csv file
    val_split_path = '/mnt/data/val_split.csv'
    val_split_df = pd.read_csv(val_split_path)

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


    
    # 创建数据集实例
    dataset = STL10Dataset(root_dir='data/stl10_binary/', transform=transform)
    
    # 创建数据加载器实例
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 遍历数据加载器以获取数据
    for images, labels in dataloader:
        # 在这里执行你的训练或评估步骤
        # images 是一个形状为 (batch_size, channels, height, width) 的张量
        # labels 是一个形状为 (batch_size,) 的张量
        # 在这里执行你的训练或评估步骤
        pass


