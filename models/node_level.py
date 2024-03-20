"""
    get node-level patient embedding
"""
import torch
from dataloader import EHRDataset, collate_fn
from torch import nn
from torch.utils.data import DataLoader

from models.base import NetBase
from models.fusion import Diag2Patient


class TransformerLayer(torch.nn.Module):
    def __init__(self, config, fusion=None):
        super().__init__()
        self.embed_dim = config["code_embed_dim"]
        self.num_head = config["tran_num_head"]
        self.num_layer = config["tran_num_layer"]
        self.agg_type = config["tran_agg_type"]
        encoder_layer = nn.TransformerEncoderLayer(
            self.embed_dim,
            self.num_head,
            dim_feedforward=512,
            dropout=config["tran_attn_dropout"],
        )
        self.net = nn.TransformerEncoder(encoder_layer, self.num_layer)
        if fusion:
            self.fusion = fusion
        else:
            self.fusion = Diag2Patient(
                config["tran_agg_type"],
                input_dim=self.embed_dim,
                hidden_dim=self.embed_dim // 4,
            )

    def forward(self, x, mask):
        out = self.net(x, src_key_padding_mask=mask).permute((1, 0, 2))
        # batch_size, length, dim
        return self.fusion(out, mask)


class Transformer(NetBase):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config["code_embed_dim"]
        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.net = TransformerLayer(config)
        self.fc = nn.Sequential(
            nn.Linear(self.net.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        mask = x == 0
        x = self.embed(x).permute((1, 0, 2))
        x = self.net(x, mask)
        return self.fc(x)

    def _return_dl(self, name):
        dataset = EHRDataset(
            self.config["dataset"],
            self.config["dataset_path"],
            self.config["fold"],
            name,
            self.config["task"],
        )
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return dl

    def train_dataloader(self):
        return self._return_dl("train")

    def val_dataloader(self):
        return [self._return_dl("validation"), self._return_dl("test")]

    def test_dataloader(self):
        return self._return_dl("test")


class MLP(NetBase):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = config["fc_dropout"]
        self.hidden = config["hidden_dim"]
        self.fc = nn.Linear(self.input_dim, 2)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.input_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(256, self.hidden),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden, 16),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(16, 2),
        # )

    def forward(self, x):
        return self.fc(x)

    def _return_dl(self, name):
        dataset = EHRDataset(
            self.config["dataset"],
            self.config["dataset_path"],
            self.config["fold"],
            name,
            self.config["task"],
            return_idx=False,
        )
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )
        return dl

    def train_dataloader(self):
        return self._return_dl("train")

    def val_dataloader(self):
        return [self._return_dl("validation"), self._return_dl("test")]

    def test_dataloader(self):
        return self._return_dl("test")
