import pandas as pd
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset
from biopandas.pdb import PandasPdb

class CavityData(Dataset):
    def __init__(self, df, pdb_dir):
        super().__init__()
        self.meta = pd.read_csv(df)
        self.pdb_list = [PandasPdb().read_pdb(os.path.join(pdb_dir, f'{pdbid}.pdb')) for pdbid in self.meta['id']]
    
    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, i):
        return self.pdb_list[i]

class CavityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4):
        super().__init__()