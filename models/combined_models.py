import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import softmax
from torch_scatter import scatter

from dataloader import EHRDataset, collate_fn, get_graph_dataset
from models.base import SSL, NetBase
from models.fusion import Diag2Patient, MergePatient
from models.graph_level import define_gnn_encoder
from models.hypergraph import HATLayer, HypergraphLayer
from models.node_level import TransformerLayer
from utils import get_adjacency


class HCL(NetBase):
    """
    Transformer + HAT + Patient GNN + Patient SSL
    """

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]
        self.num_layer = config["hgnn_num_layer"]

        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.fusion = Diag2Patient("attn", input_dim=self.embed_dim, hidden_dim=self.embed_dim // 2)
        self.hyper_net = nn.ModuleList([HATLayer(self.fusion) for i in range(self.num_layer)])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.trans_net = TransformerLayer(config, self.fusion)
        self.gnn_net = define_gnn_encoder(config)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(16, 2),
        )
        self.agg_type = config["patient_agg_type"]
        if self.agg_type != "drop":
            self.merge = MergePatient(self.agg_type, input_dim=self.embed_dim, input_num=3)

        # patient graph
        self.masks = {}
        (
            self.data,
            self.masks["train"],
            self.masks["validation"],
            self.masks["test"],
        ) = get_graph_dataset(config)
        self.sample_size = [config["sample_size2"], config["sample_size1"]]

        self.ssl_method = config["ssl_method"]
        assert self.ssl_method in ["none", "simclr", "supcon"]
        if self.ssl_method != "none":
            self.alpha = config["alpha"]
            self.SSL = SSL(config)

    def forward(self, x, pyg_adj, batch_size, edge_weight, label=None):
        x_embed = self.embed(x)
        # Patient GNN
        x_patient_embed = self.fusion(x_embed, (x == 0))
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]

        # Transformer
        x = x[:batch_size]
        mask = x == 0
        embedding = x_embed[:batch_size].permute((1, 0, 2))
        trans_x = self.trans_net(embedding, mask)

        # HAT
        e = []
        v = []
        for i in range(x.shape[0]):
            p = sum(x[i] != 0)
            e.extend([i for j in range(p)])
            v.append(x[i][:p])
        v = torch.cat(v)
        e = torch.tensor(e, device=x.device)

        for hat in self.hyper_net:
            Xv = hat(self.embed.weight, v, e)
        Xv = self.norm(Xv)
        alpha1 = self.fusion.W(Xv)  # (node, 1)
        alpha1 = alpha1[v]  # (nnz, 1)
        alpha1 = softmax(alpha1, e, dim=0)  # (nnz, 1)
        Xve = Xv[v]  # (nnz, D)
        Xve *= alpha1  # (nnz, D)
        hyper_x = scatter(Xve, e, dim=0, reduce="sum")  # (B, D)

        if self.agg_type == "drop":
            out = trans_x
        else:
            out = self.merge(hyper_x, trans_x, graph_x)

        if self.ssl_method == "none":
            return self.fc(out)
        else:
            ssl_loss1 = self.SSL(hyper_x, trans_x, label)
            ssl_loss2 = self.SSL(hyper_x, graph_x, label)
            return self.fc(out), ssl_loss1 + ssl_loss2

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, y, edge_weight = self.prepare_batch(batch)

        if self.ssl_method != "none":
            out, ssl_loss = self(
                x,
                adjs,
                batch_size,
                edge_weight,
                y if name == "train" else None,
            )
            loss1 = self.loss(out, y)
            loss2 = ssl_loss
            loss = loss1 + self.alpha * loss2
        else:
            out = self(x, adjs, batch_size, edge_weight)
            loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()

        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            if self.ssl_method != "none":
                self.log("clf_loss", loss1, on_step=True, on_epoch=True)
                self.log("ssl_loss", loss2, on_step=True, on_epoch=True)
            return {"loss": loss, "y": y.detach(), "proba": proba}
        else:
            return {f"{name}_loss": loss.detach(), "y": y.detach(), "proba": proba}

    def _return_dl(self, name):
        loader = NeighborSampler(
            self.data.edge_index,
            node_idx=self.masks[name],
            sizes=self.sample_size,
            batch_size=self.config["bs"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        return loader


class HyperTransGNN(NetBase):
    """
    Transformer + Hypergraph + Patient GNN + Patient SSL
    """

    def __init__(self, config):
        super().__init__(config)
        self.adjs = {}
        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]

        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.hyper_net = HypergraphLayer(config)
        self.trans_net = TransformerLayer(config)
        self.gnn_net = define_gnn_encoder(config)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 2),
        )
        self.merge = MergePatient(config["patient_agg_type"], input_dim=self.embed_dim, input_num=3)
        self.gnn_merge = Diag2Patient(
            config["gnn_merge_type"],
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim // 4,
        )

        # hypergraph
        self.get_adjs()

        # patient graph
        self.masks = {}
        (
            self.data,
            self.masks["train"],
            self.masks["validation"],
            self.masks["test"],
        ) = get_graph_dataset(self.config)
        self.sample_size = [config["sample_size2"], config["sample_size1"]]

        self.ssl_method = config["ssl_method"]
        assert self.ssl_method in ["none", "simclr", "supcon"]
        if self.ssl_method != "none":
            self.alpha = config["alpha"]
            self.SSL = SSL(config)

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
            # self.adjs[name] = (self.adjs[name] + self.adjs['train']) / 2
            # 70/15/15 split
            self.adjs[name] = (self.adjs[name] * 3 + self.adjs["train"] * 14) / 17

    def forward(self, x, adj, pyg_adj, batch_size, edge_weight, label=None):
        x_embed = self.embed(x)
        x_patient_embed = self.gnn_merge(x_embed, (x == 0))
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]

        x = x[:batch_size]
        mask = x == 0
        embedding = x_embed[:batch_size].permute((1, 0, 2))
        hyper_x = self.hyper_net(x, adj, self.embed.weight)
        trans_x = self.trans_net(embedding, mask)
        out = self.merge(hyper_x, trans_x, graph_x)

        if self.ssl_method == "none":
            return self.fc(out)
        else:
            ssl_loss1 = self.SSL(hyper_x, trans_x, label)
            ssl_loss2 = self.SSL(hyper_x, graph_x, label)
            return self.fc(out), ssl_loss1 + ssl_loss2

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, y, edge_weight = self.prepare_batch(batch)

        if self.ssl_method != "none":
            out, ssl_loss = self(
                x,
                self.adjs[name],
                adjs,
                batch_size,
                edge_weight,
                y if name == "train" else None,
            )
            loss1 = self.loss(out, y)
            loss2 = ssl_loss
            loss = loss1 + self.alpha * loss2
        else:
            out = self(x, self.adjs[name], adjs, batch_size, edge_weight)
            loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            if self.ssl_method != "none":
                self.log("clf_loss", loss1, on_step=True, on_epoch=True)
                self.log("ssl_loss", loss2, on_step=True, on_epoch=True)
            return {"loss": loss, "y": y.detach(), "proba": proba}
        else:
            return {f"{name}_loss": loss.detach(), "y": y.detach(), "proba": proba}

    def _return_dl(self, name):
        loader = NeighborSampler(
            self.data.edge_index,
            node_idx=self.masks[name],
            sizes=self.sample_size,
            batch_size=self.config["bs"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        self.adjs[name] = self.adjs[name].to(self.device)
        return loader

    def train_dataloader(self):
        return self._return_dl("train")

    def val_dataloader(self):
        return self._return_dl("validation")

    def test_dataloader(self):
        return self._return_dl("test")
