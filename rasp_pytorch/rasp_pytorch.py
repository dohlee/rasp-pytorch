import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import pytorch_lightning as pl

class RaSPCavityModel(pl.LightningModule):
    def __init__(self, n_atom_types=5): # but n_atom_types = 5 in the manuscript
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv3d(n_atom_types, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            Rearrange('b c x y z -> (b x y z) c'),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 20),
        )

    def forward(self, x):
        return self.main(x)
    
    def training_step(self, batch):
        bsz = batch.shape[0]

        # dummy for now
        label = torch.randint(0, 20, (bsz,))

        out = self(x)
        loss = F.cross_entropy(out, label)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    bsz = 8
    x = torch.randn([bsz, 5, 18, 18, 18])
    model = RaSPCavityModel()

    print(model.training_step(x))