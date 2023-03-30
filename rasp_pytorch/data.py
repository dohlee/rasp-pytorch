import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset 

class CavityData(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

class CavityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4):
        super().__init__()