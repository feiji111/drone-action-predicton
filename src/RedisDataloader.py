import redis
from torch.utils.data import DataLoader, Dataset


class RedisDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index) -> _T_co:
        return super().__getitem__(index)
