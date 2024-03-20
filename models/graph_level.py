"""
    get graph-level patient embedding
"""
from pathlib import Path
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import EHRDataset, collate_fn, get_graph_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from utils import load_json, print_metrics_binary, seed_everything, write_json

from models.base import NetBase
from models.fusion import Diag2Patient


def define_gnn_encoder(config):
    gnn_name = config['gnn_name']
    if gnn_name == 'gat':
        return GATLayer(config)
    elif gnn_name == 'sage':
        return SAGELayer(config)
    else:
        raise NotImplementedError("only implemented for GAT, GraphSAGE models.")


class GATLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.featdrop = config['gnn_featdrop']
        self.num_layers = config['gnn_layers']
        in2 = config['gat_nhid'] * config['gat_n_heads']
        self.convs = torch.nn.ModuleList(
            [
                GATConv(
                    config['code_embed_dim'],
                    config['gat_nhid'],
                    heads=config['gat_n_heads'],
                    dropout=config['gat_attndrop'],
                ),
                GATConv(
                    in2,
                    config['code_embed_dim'],
                    heads=config['gat_n_out_heads'],
                    concat=False,
                    dropout=config['gat_attndrop'],
                ),
            ]
        )

    def forward(self, x, adjs, edge_weight=None):
        x = F.dropout(x, p=self.featdrop, training=self.training)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)

        return x

    # def inference(self, x_all, subgraph_loader, device):
    #     for i in range(self.num_layers):
    #         xs = []
    #         for batch_size, n_id, adj in subgraph_loader:
    #             edge_index, _, size = adj.to(device)
    #             x = x_all[n_id].to(device)
    #             x_target = x[: size[1]]
    #             x = self.convs[i]((x, x_target), edge_index)
    #             if i != self.num_layers - 1:
    #                 x = F.elu(x)
    #             xs.append(x)

    #         x_all = torch.cat(xs, dim=0)
    #     else:
    #         return x_all


class SAGELayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = 2
        self.featdrop = config['gnn_featdrop']
        dim = config['code_embed_dim']
        self.convs = torch.nn.ModuleList(
            [
                SAGEConv(dim, dim),
                SAGEConv(dim, dim),
            ]
        )

    def forward(self, x, adjs, edge_weight=None):
        x = F.dropout(x, p=self.featdrop, training=self.training)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)

        return x


class GNN(NetBase):
    """
    Patient GNN
    """

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config['code_embed_dim']
        self.input_dim = config['input_dim']

        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)
        self.gnn_net = define_gnn_encoder(config)
        # self.gnn_net = GATLayer(config)

        self.fc = nn.Sequential(
            nn.Linear(config['code_embed_dim'], 128),
            nn.ReLU(),
            nn.Dropout(config['fc_dropout']),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config['fc_dropout']),
            nn.Linear(32, 2),
        )

        if config['gnn_merge_type'] == 'mean':
            self.gnn_merge = Diag2Patient('mean')
        elif config['gnn_merge_type'] == 'attn':
            self.gnn_merge = Diag2Patient(
                'attn', input_dim=self.embed_dim, hidden_dim=self.embed_dim // 4
            )

        # patient graph
        self.masks = {}
        (
            self.data,
            self.masks['train'],
            self.masks['validation'],
            self.masks['test'],
        ) = get_graph_dataset(self.config)
        self.sample_size = [config['sample_size2'], config['sample_size1']]

    def forward(self, x, pyg_adj, batch_size, edge_weight):
        x_embed = self.embed(x)
        x_patient_embed = self.gnn_merge(x_embed, (x == 0))
        graph_x = self.gnn_net(x_patient_embed, pyg_adj, edge_weight)[:batch_size]
        return self.fc(graph_x)

    def evaluate_step(self, batch, name):
        batch_size, n_id, adjs = batch
        x, y, edge_weight = self.prepare_batch(batch)

        out = self(x, adjs, batch_size, edge_weight)
        loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == 'train':
            self.log('train_loss', loss, on_step=True, on_epoch=True)
            return {'loss': loss, 'y': y.detach(), 'proba': proba}
        else:
            return {f'{name}_loss': loss.detach(), 'y': y.detach(), 'proba': proba}

    def _return_dl(self, name):
        loader = NeighborSampler(
            self.data.edge_index,
            node_idx=self.masks[name],
            sizes=self.sample_size,
            batch_size=self.config['bs'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=False,
        )
        return loader

    def train_dataloader(self):
        return self._return_dl('train')

    def val_dataloader(self):
        return self._return_dl('validation')

    def test_dataloader(self):
        return self._return_dl('test')


# used for whole graph inference

# class GNNModule(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.gnn_name = config['gnn_name']

#         self.gnn_encoder = define_gnn_encoder(config['gnn_name'])(config)
#         self.node_encoder = define_node_encoder(config['node_name'])(config)
#         self.embed_dim = config['embed_dim']
#         # fusion.py
#         self.merge = nn.Sequential(
#             nn.Linear(self.embed_dim * 2, self.embed_dim),
#             nn.ELU(),
#             nn.Dropout(),
#             nn.Linear(self.embed_dim, 16),
#             nn.ELU(),
#             nn.Dropout(),
#             nn.Linear(16, 2),
#         )

#     def forward(self, x, adjs, batch_size, edge_weight):
#         # get patient embedding
#         inner_embed = self.node_encoder(x)
#         inter_embed = self.gnn_encoder(inner_embed, adjs, edge_weight)
#         embed = torch.cat([inner_embed[:batch_size, :], inter_embed], axis=-1)
#         return self.merge(embed)

#     def infer_by_batch(self, plain_dataloader, device):
#         outs = []
#         for x, y in plain_dataloader:
#             out = self.node_encoder(x.to(device))
#             outs.append(out)
#         return torch.cat(outs, dim=0)

#     def inference(self, x_all, edge_weight, plain_dataloader, subgraph_loader, device):
#         # collect all inner-patient embeddings by minibatching:
#         inner_embed = self.infer_by_batch(plain_dataloader, device)

#         # collect all inter-patient embeddings
#         inter_embed = self.gnn_encoder.inference(
#             inner_embed,
#             subgraph_loader,
#             device,
#         )

#         embed = torch.cat([inner_embed, inter_embed], axis=-1)
#         return self.merge(embed)


# class GNNBase(pl.LightningModule):
#     """
#     Basic GNN model
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.batch_size = config['bs']
#         self.seed = config['seed']
#         self.loss_weight = torch.tensor(config['loss_weight']).float()
#         self.loss = nn.CrossEntropyLoss(weight=self.loss_weight)
#         self.learning_rate = config['lr']

#         self.dataset, self.train_loader, self.subgraph_loader = get_data(self.config)
#         self.plain_dataloader = DataLoader(
#             self.dataset.ehr_data,
#             batch_size=self.batch_size,
#             num_workers=self.config['num_workers'],
#             shuffle=False,
#         )
#         self.net = GNNModule(self.config)

#     def forward(self, x, adjs, batch_size, edge_weight):
#         return self.net(x, adjs, batch_size, edge_weight)

#     def training_step(self, batch, batch_idx):
#         # these are train-masked already (from train-dataloader)
#         batch_size, n_id, adjs = batch
#         x = self.dataset.data.x[n_id].to(self.device)
#         edge_weight = self.dataset.data.edge_attr.to(self.device)
#         y = self.dataset.data.y[n_id[:batch_size]].to(self.device)

#         out = self(x, adjs, batch_size, edge_weight)
#         loss = self.loss(out, y)
#         proba = F.softmax(out, dim=-1).detach()
#         self.log('train_loss', loss, on_step=True, on_epoch=True)
#         return {'loss': loss, 'y': y.detach(), 'proba': proba}

#     def evaluate_step(self, name):
#         x = self.dataset.data.x.to(self.device)
#         edge_weight = self.dataset.data.edge_attr.to(self.device)
#         truth = self.dataset.data.y

#         out = self.net.inference(
#             x, edge_weight, self.plain_dataloader, self.subgraph_loader, self.device
#         )

#         if name == 'val':
#             y = truth[self.dataset.data.val_mask].to(self.device)
#             out = out[self.dataset.data.val_mask]
#         elif name == 'test':
#             y = truth[self.dataset.data.test_mask].to(self.device)
#             out = out[self.dataset.data.test_mask]

#         loss = self.loss(out, y).detach()
#         proba = F.softmax(out, dim=-1).detach()
#         return {f'{name}_loss': loss, 'y': y.detach(), 'proba': proba}

#     def validation_step(self, batch, batch_idx):
#         # there's just one step for the validation epoch:
#         # - we're not using batch / batch_idx (it's just dummy)
#         return self.evaluate_step('val')

#     def test_step(self, batch, batch_idx):
#         return self.evaluate_step('test')

#     def evaluate_epoch(self, outputs, name):
#         if name == 'train':
#             proba = np.concatenate([i['proba'].cpu().numpy() for i in outputs], axis=0)
#             y = np.concatenate([i['y'].cpu().numpy() for i in outputs], axis=0)
#         else:
#             proba = outputs[0]['proba'].cpu().numpy()
#             y = outputs[0]['y'].cpu().numpy()

#         res = print_metrics_binary(y, proba, verbose=0)
#         if name == 'val':
#             self.log('val_loss', outputs[0]['val_loss'])
#             self.log('auroc', res['auroc'])
#             self.log('auprc', res['auprc'])

#         if type(self.logger).__name__ != 'WandbLogger':
#             config_path = f'{self.logger.log_dir}/config.json'
#             if not os.path.exists(config_path):
#                 write_json(self.config, path=config_path, verbose=0)

#             path = f'{self.logger.log_dir}/{name}_res.csv'
#             if os.path.exists(path):
#                 with open(path, "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(
#                         [
#                             "{:.5f}".format(res['acc']),
#                             "{:.5f}".format(res['auroc']),
#                             "{:.5f}".format(res['auprc']),
#                             "{:.5f}".format(res['f1']),
#                             "{:.5f}".format(res['minpse']),
#                         ]
#                     )
#             else:
#                 with open(path, "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(['acc', 'auroc', 'auprc', 'f1', 'minpse'])

#         pprint(res)
#         return res

#     def training_epoch_end(self, outputs):
#         self.evaluate_epoch(outputs, 'train')

#     def validation_epoch_end(self, outputs):
#         self.evaluate_epoch(outputs, 'val')

#     def test_epoch_end(self, outputs):
#         return self.evaluate_epoch(outputs, 'test')

#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
#         sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
#         return {
#             'optimizer': opt,
#             'lr_scheduler': sch,
#             'monitor': 'val_loss',
#         }

#     def train_dataloader(self):
#         return self.train_loader

#     def val_dataloader(self):
#         return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

#     def on_train_start(self):
#         seed_everything(self.seed)

#     def get_progress_bar_dict(self):
#         # don't show the version number
#         items = super().get_progress_bar_dict()
#         items.pop("v_num", None)
#         return items
