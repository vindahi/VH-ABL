import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import math
import numpy as np



class Fusion(nn.Module):
    def __init__(self, fusion_dim, nbit):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )

    def forward(self, x, y):
        fused_feat = self.fusion(torch.cat([x, y], dim=-1))
        hash_code = self.hash(fused_feat)
        return hash_code


class BitImportanceNetwork(nn.Module):
    def __init__(self, nbit):
        super(BitImportanceNetwork, self).__init__()
        self.nbit = nbit
        self.importance_net = nn.Sequential(
            nn.Linear(nbit, nbit//2),
            nn.ReLU(),
            nn.Linear(nbit//2, nbit)
        )
        
    def forward(self, h):
        w = self.importance_net(h)
        return w
        return output
