import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UCIHARDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.split = split
        base_path = f"{data_path}/UCI HAR Dataset/{split}"

        # Load features and labels
        X = np.loadtxt(f"{base_path}/X_{split}.txt")
        y = np.loadtxt(f"{base_path}/y_{split}.txt") - 1  # 0-indexed

        # Load raw inertial signals (9 channels: 3 acc + 3 gyro + 3 total_acc)
        signals_path = f"{base_path}/Inertial Signals"
        acc_x = np.loadtxt(f"{signals_path}/body_acc_x_{split}.txt")
        acc_y = np.loadtxt(f"{signals_path}/body_acc_y_{split}.txt")
        acc_z = np.loadtxt(f"{signals_path}/body_acc_z_{split}.txt")
        gyro_x = np.loadtxt(f"{signals_path}/body_gyro_x_{split}.txt")
        gyro_y = np.loadtxt(f"{signals_path}/body_gyro_y_{split}.txt")
        gyro_z = np.loadtxt(f"{signals_path}/body_gyro_z_{split}.txt")

        # Stack signals: (N, 6, 128) - 6 channels, 128 timesteps
        self.signals = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)
        self.labels = y.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.signals[idx]), torch.LongTensor([self.labels[idx]])[0]

def get_dataloaders(data_path, batch_size=64):
    train_dataset = UCIHARDataset(data_path, 'train')
    test_dataset = UCIHARDataset(data_path, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
