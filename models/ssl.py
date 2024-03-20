import torch
import torch.nn.functional as F
from dataloader import EHRDataset, collate_fn
from torch import nn
from torch.utils.data import DataLoader
from utils import get_adjacency

from models.base import sslNetBase
from models.fusion import Diag2Patient, MergePatient
from models.graph_level import define_gnn_encoder
from models.hypergraph import HypergraphLayer
from models.node_level import TransformerLayer


class HATssl(sslNetBase):
    """
    Transformer + HAT + Patient GNN + Patient SSL
    """

    def __init__(self, config):
        super().__init__(config)
        self.hyper_net = HypergraphLayer(config)
        self.trans_net = TransformerLayer(config)
        self.gnn_net = define_gnn_encoder(config)
        self.gnn_merge = Diag2Patient("mean")

        # hypergraph
        self.adjs = {}
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
            self.adjs[name] = (self.adjs[name] + self.adjs["train"]) / 2
        # self.adjs[name] = self.adjs[name].to(self.device)

    def forward(self, x, adj, pyg_adj, batch_size, edge_weight):
        # print('1\n', x)
        x_embed = self.embed(x)
        # print('2\n', x_embed)
        x_patient_embed = self.gnn_merge(x_embed, (x == 0))
        # print('3\n', x_patient_embed)
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]

        x = x[:batch_size]
        mask = x == 0
        embedding = x_embed[:batch_size].permute((1, 0, 2))
        hyper_x = self.hyper_net(x, adj, self.embed.weight)
        trans_x = self.trans_net(embedding, mask)

        # for x in [hyper_x, trans_x, graph_x]:
        #     print('#' * 30)
        #     print(x)
        out = torch.cat([hyper_x, trans_x, graph_x], axis=-1)

        ssl_loss1 = self.SSL(hyper_x, trans_x)
        ssl_loss2 = self.SSL(hyper_x, graph_x)
        return out, ssl_loss1, ssl_loss2

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, dead_y, read_y, edge_weight = self.prepare_batch(batch)
        self.adjs[name] = self.adjs[name].to(self.device)
        out, ssl_loss1, ssl_loss2 = self(x, self.adjs[name], adjs, batch_size, edge_weight)
        loss = ssl_loss1 + ssl_loss2

        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            self.log("ssl_loss1", ssl_loss1, on_step=True, on_epoch=True)
            self.log("ssl_loss2", ssl_loss2, on_step=True, on_epoch=True)
            return {
                "loss": loss,
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }
        else:
            return {
                f"{name}_loss": loss.detach(),
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }


class TransGNNssl(sslNetBase):
    """
    Transformer + Patient GNN + Patient SSL + Linear classification protocal
    """

    def __init__(self, config):
        super().__init__(config)
        self.trans_net = TransformerLayer(config)
        self.gnn_net = define_gnn_encoder(config)
        self.gnn_merge = Diag2Patient(
            config["gnn_merge_type"],
            input_dim=config["code_embed_dim"],
            hidden_dim=config["code_embed_dim"],
        )

    def forward(self, x, pyg_adj, batch_size, edge_weight):
        x_embed = self.embed(x)
        x_patient_embed = self.gnn_merge(x_embed, (x == 0))
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]

        x = x[:batch_size]
        mask = x == 0
        embedding = x_embed[:batch_size].permute((1, 0, 2))
        trans_x = self.trans_net(embedding, mask)

        out = torch.cat([trans_x, graph_x], axis=-1)

        ssl_loss = self.SSL(graph_x, trans_x)
        return out, ssl_loss

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, dead_y, read_y, edge_weight = self.prepare_batch(batch)
        out, loss = self(x, adjs, batch_size, edge_weight)

        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return {
                "loss": loss,
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }
        else:
            return {
                f"{name}_loss": loss.detach(),
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }


class HyperTransGNNssl(sslNetBase):
    """
    Transformer + Hypergraph + Patient GNN + Patient SSL
    """

    def __init__(self, config):
        super().__init__(config)
        self.hyper_net = HypergraphLayer(config)
        self.trans_net = TransformerLayer(config)
        self.gnn_net = define_gnn_encoder(config)
        self.gnn_merge = Diag2Patient("mean")

        # hypergraph
        self.adjs = {}
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
            self.adjs[name] = (self.adjs[name] + self.adjs["train"]) / 2
        # self.adjs[name] = self.adjs[name].to(self.device)

    def forward(self, x, adj, pyg_adj, batch_size, edge_weight):
        # print('1\n', x)
        x_embed = self.embed(x)
        # print('2\n', x_embed)
        x_patient_embed = self.gnn_merge(x_embed, (x == 0))
        # print('3\n', x_patient_embed)
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]

        x = x[:batch_size]
        mask = x == 0
        embedding = x_embed[:batch_size].permute((1, 0, 2))
        hyper_x = self.hyper_net(x, adj, self.embed.weight)
        trans_x = self.trans_net(embedding, mask)

        # for x in [hyper_x, trans_x, graph_x]:
        #     print('#' * 30)
        #     print(x)
        out = torch.cat([hyper_x, trans_x, graph_x], axis=-1)

        ssl_loss1 = self.SSL(hyper_x, trans_x)
        ssl_loss2 = self.SSL(hyper_x, graph_x)
        return out, ssl_loss1, ssl_loss2

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, dead_y, read_y, edge_weight = self.prepare_batch(batch)
        self.adjs[name] = self.adjs[name].to(self.device)
        out, ssl_loss1, ssl_loss2 = self(x, self.adjs[name], adjs, batch_size, edge_weight)
        loss = ssl_loss1 + ssl_loss2

        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            self.log("ssl_loss1", ssl_loss1, on_step=True, on_epoch=True)
            self.log("ssl_loss2", ssl_loss2, on_step=True, on_epoch=True)
            return {
                "loss": loss,
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }
        else:
            return {
                f"{name}_loss": loss.detach(),
                "dead_y": dead_y,
                "read_y": read_y,
                "out": out.detach(),
            }


# todo
class HyperTrans(sslNetBase):
    """
    Transformer + Hypergraph + Patient SSL
    """

    def __init__(self, config):
        super().__init__(config)
        self.adjs = {}
        self.task = config["task"]
        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]
        self.ssl = config["ssl"]
        if self.ssl:
            self.alpha = config["alpha"]
        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.hyper_net = HypergraphLayer(config)
        self.trans_net = TransformerLayer(config)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 2),
        )
        self.merge = MergePatient(config["patient_agg_type"], input_dim=self.embed_dim, input_num=2)
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
            self.adjs[name] = (self.adjs[name] + self.adjs["train"]) / 2

    def forward(self, x, adj):
        hyper_x = self.hyper_net(x, adj, self.embed.weight)
        embedding = self.embed(x).permute((1, 0, 2))
        mask = x == 0
        trans_x = self.trans_net(embedding, mask)
        out = self.merge(hyper_x, trans_x)
        if self.ssl:
            ssl_loss = SSL(hyper_x, trans_x, self.config["ssl_method"])
            return self.fc(out), ssl_loss
        else:
            return self.fc(out)

    def evaluate_step(self, batch, name):
        x, y = batch
        if self.ssl:
            out, ssl_loss = self(x, self.adjs[name])
            loss1 = self.loss(out, y)
            loss2 = ssl_loss
            loss = loss1 + self.alpha * loss2
        else:
            out = self(x, self.adjs[name])
            loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            if self.ssl:
                self.log("clf_loss", loss1, on_step=True, on_epoch=True)
                self.log("ssl_loss", loss2, on_step=True, on_epoch=True)
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
            pin_memory=False,
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
