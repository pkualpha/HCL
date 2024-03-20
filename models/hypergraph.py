import torch
import torch.nn.functional as F
from dataloader import EHRDataset, collate_fn
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.utils import softmax
from torch_scatter import scatter
from utils import get_adjacency

from models.base import NetBase
from models.fusion import Diag2Patient


class HypergraphLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["code_embed_dim"]
        self.num_layer = config["hgnn_num_layer"]
        self.fusion = Diag2Patient(
            config["hgnn_agg_type"],
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim // config["hgnn_fusion_dim_div"],
        )
        self.act_name = config["hgnn_activation"]
        if self.act_name == "elu":
            self.act = nn.ELU()
        elif self.act_name == "relu":
            self.act = nn.ReLU()
        elif self.act_name == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif self.act_name == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.act_name == "tanh":
            self.act = nn.Tanh()

    def forward(self, x, adj, embedding):
        # concat or add?
        E = embedding[1:, :]
        final = [E]
        for i in range(self.num_layer):
            E = torch.sparse.mm(adj, E)
            if self.act_name != "none":
                E = self.act(E)
            final.append(E)
        E = torch.mean(torch.stack(final), 0)

        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # for i in range(self.num_layer):
        #     E += torch.sparse.mm(adj, E)

        E = torch.cat([torch.zeros(1, E.shape[1], device=E.device), E])
        out = E[x]
        return self.fusion(out, x == 0)


class HATLayer(torch.nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.embed_dim = fusion.input_dim
        self.fusion1 = fusion
        self.W1 = self.fusion1.W
        self.fusion2 = Diag2Patient("attn", input_dim=self.embed_dim * 2, hidden_dim=fusion.hidden_dim)
        self.W2 = self.fusion2.W
        self.edge_norm = nn.LayerNorm(self.embed_dim)
        self.node_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout()

    def forward(self, Xv_in, v, e):
        """
        input:
            embed : (code_num, dim)
            v : the row index for the sparse incident matrix H, |V| x |E|
            e : the col index for the sparse incident matrix H, |V| x |E|
        output:
            Xv : updated embedding (code_num, dim)
        """
        # node to edge
        alpha1 = self.W1(Xv_in)  # (node, 1)
        alpha1 = alpha1[v]  # (nnz, 1)
        alpha1 = softmax(alpha1, e, dim=0)  # (nnz, 1)
        Xve = Xv_in[v]  # (nnz, D)
        Xve *= alpha1  # (nnz, D)
        Xe = scatter(Xve, e, dim=0, reduce="sum")  # (B, D)
        Xe = self.edge_norm(Xe)
        # edge to norm
        Xev = Xe[e]  # (nnz, D)
        alpha2 = torch.cat((Xev, Xve), dim=1)  # (nnz, 2D)
        alpha2 = self.W2(alpha2)
        alpha2 = softmax(alpha2, v, dim=0)  # (nnz, 1)
        Xev *= alpha2
        Xv = scatter(Xev, v, dim=0, reduce="sum", dim_size=Xv_in.shape[0])

        Xv = self.dropout(Xv) + Xv_in
        return Xv


class HAT(NetBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_layer = config["hgnn_num_layer"]
        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]
        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.fusion = Diag2Patient("attn", input_dim=self.embed_dim, hidden_dim=self.embed_dim // 2)
        self.net = nn.ModuleList([HATLayer(self.fusion) for i in range(self.num_layer)])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (batch_size. seq_len)
        e = []
        v = []
        for i in range(x.shape[0]):
            p = sum(x[i] != 0)
            e.extend([i for j in range(p)])
            v.append(x[i][:p])
        v = torch.cat(v)
        e = torch.tensor(e, device=x.device)

        for hat in self.net:
            Xv = hat(self.embed.weight, v, e)
        Xv = self.norm(Xv)
        alpha1 = self.fusion.W(Xv)  # (node, 1)
        alpha1 = alpha1[v]  # (nnz, 1)
        alpha1 = softmax(alpha1, e, dim=0)  # (nnz, 1)
        Xve = Xv[v]  # (nnz, D)
        Xve *= alpha1  # (nnz, D)
        Xe = scatter(Xve, e, dim=0, reduce="sum")  # (B, D)
        return self.fc(Xe)


class Hypergraph(NetBase):
    def __init__(self, config):
        super().__init__(config)
        self.adjs = {}
        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]
        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.net = HypergraphLayer(config)
        self.fc = nn.Sequential(
            nn.Linear(self.net.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(32, 2),
        )
        self.get_adjs()

    def get_adjs(self):
        for name in ["train", "validation", "test"]:
            dataset = EHRDataset(
                self.config["dataset"],
                self.config["dataset_path"],
                self.config["fold"],
                name,
                self.config["task"],
            )
            self.adjs[name] = get_adjacency(dataset.x)
        for name in ["validation", "test"]:
            # 70/15/15 split
            self.adjs[name] = (self.adjs[name] * 3 + self.adjs["train"] * 14) / 17

    def forward(self, x, adj):
        x = self.net(x, adj, self.embed.weight)
        return self.fc(x)

    def evaluate_step(self, batch, name):
        x, y = batch
        out = self(x, self.adjs[name])
        loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return {"loss": loss, "y": y.detach(), "proba": proba}
        else:
            return {f"{name}_loss": loss.detach(), "y": y.detach(), "proba": proba}

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
        self.adjs[name] = self.adjs[name].to(self.device)
        return dl

    def train_dataloader(self):
        return self._return_dl("train")

    def val_dataloader(self):
        return self._return_dl("validation")

    def test_dataloader(self):
        return self._return_dl("test")
