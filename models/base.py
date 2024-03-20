"""
    get node-level patient embedding
"""
import csv
import datetime
import enum
import os
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from dataloader import EHRDataset, SSLDataset, collate_fn, get_graph_dataset
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import NeighborSampler
from utils import (load_pkl, print_metrics_binary, seed_everything, write_json,
                   write_pkl)

from models.loss import define_loss


class NetBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(config)
        self.input_dim = config["input_dim"]
        self.batch_size = config["bs"]
        self.seed = config["seed"]
        self.learning_rate = config["lr"]
        self.loss = define_loss(config)

    def forward(self, x):
        raise NotImplementedError()

    def evaluate_step(self, batch, name):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return {"loss": loss, "y": y.detach(), "proba": proba}
        else:
            return {f"{name}_loss": loss.detach(), "y": y.detach(), "proba": proba}

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.evaluate_step(batch, "validation" if dataloader_idx == 0 else "val_test")

    def test_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "test")

    def prepare_batch(self, batch):
        # prepare data for patient-graph based model
        batch_size, n_id, adjs = batch
        x = self.data.x[n_id]
        # not using collate_fn when using NeighborSampler
        # need to add idx by 1
        # idx start from 1, 0 is the padding idx
        batch_x = [x[i].nonzero().squeeze(1) + 1 for i in range(x.size(0))]
        max_len = max([x.shape[-1] for x in batch_x])
        xs = [torch.cat([i, torch.zeros(max_len - i.shape[-1], dtype=int)]) for i in batch_x]
        x = torch.stack(xs).to(self.device)
        y = self.data.y[n_id[:batch_size]].to(self.device)
        # edge weight not needed for GAT actually
        edge_weight = self.data.edge_attr.to(self.device)
        return x, y, edge_weight

    def evaluate_epoch(self, outputs, name):
        # do not use .cpu() frequently
        proba = np.concatenate([i["proba"].cpu().numpy() for i in outputs], axis=0)
        y = np.concatenate([i["y"].cpu().numpy() for i in outputs], axis=0)
        # print(name, y.shape, proba.shape)
        # print(y)
        # print(proba)
        try:
            res = print_metrics_binary(y, proba, verbose=0)
            self.log(f"{name}_auroc", res["auroc"], on_epoch=True)
            self.log(f"{name}_auprc", res["auprc"], on_epoch=True)
            self.log(f"{name}_acc", res["acc"], on_epoch=True)
            self.log(f"{name}_f1", res["f1"], on_epoch=True)
            self.log(f"{name}_minpse", res["minpse"], on_epoch=True)

            if type(self.logger).__name__ == "WandbLogger" and name == "test":
                self.logger.experiment.log(
                    {
                        f"{name}_cf": wandb.plot.confusion_matrix(
                            probs=proba,
                            y_true=y,
                            class_names=["Alive", "Dead"],
                        ),
                        # f"{name}_prc": wandb.plot.pr_curve(y, proba, labels=["Alive", "Dead"]),
                        # f"{name}_roc": wandb.plot.roc_curve(y, proba, labels=["Alive", "Dead"]),
                    },
                )

            if not type(self.logger).__name__ in [
                "NoneType",
                "WandbLogger",
                "DummyLogger",
            ]:
                config_path = f"{self.logger.log_dir}/config.json"
                if not os.path.exists(config_path):
                    write_json(self.config, path=config_path, verbose=0)

                path = f"{self.logger.log_dir}/{name}_res.csv"
                if os.path.exists(path):
                    with open(path, "a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                "{:.5f}".format(res["acc"]),
                                "{:.5f}".format(res["auroc"]),
                                "{:.5f}".format(res["auprc"]),
                                "{:.5f}".format(res["f1"]),
                                "{:.5f}".format(res["minpse"]),
                            ]
                        )
                else:
                    with open(path, "a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["acc", "auroc", "auprc", "f1", "minpse"])

            print()
            pprint(res)
            return res
        except Exception as e:
            self.log(f"{name}_auroc", 0, on_epoch=True)
            print(e)
            return

    def training_epoch_end(self, outputs):
        try:
            self.evaluate_epoch(outputs, "train")
        except Exception as e:
            print(e)
            pass

    def validation_epoch_end(self, outputs):
        try:
            t = ["validation", "val_test"]
            for i, output in enumerate(outputs):
                self.evaluate_epoch(output, t[i])
        except Exception as e:
            print(e)
            pass

    def test_epoch_end(self, outputs):
        return self.evaluate_epoch(outputs, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.config["lr_sch"] == "none":
            return opt
        if self.config["lr_sch"] == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, mode="max")
        elif self.config["lr_sch"] == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=self.config["patience"], mode="max"
            )
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "validation_auroc",
        }

    def on_train_start(self):
        seed_everything(self.seed)

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

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class sslNetBase(pl.LightningModule):
    """
    Transformer + Hypergraph + Patient GNN + Patient SSL
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config["model_name"]
        self.batch_size = config["bs"]
        self.seed = config["seed"]
        self.learning_rate = config["lr"]

        self.embed_dim = config["code_embed_dim"]
        self.input_dim = config["input_dim"]

        self.embed = nn.Embedding(self.input_dim + 1, self.embed_dim, padding_idx=0)

        # patient graph
        self.masks = {}
        (
            self.data,
            self.masks["train"],
            self.masks["validation"],
            self.masks["test"],
        ) = get_graph_dataset(self.config)
        self.dead_y, self.read_y = self._get_y()

        self.sample_size = [config["sample_size2"], config["sample_size1"]]

        self.SSL = SSL(config)
        self.cluster_evaluate = config["cluster_evaluate"]

        # self.fc = nn.Linear()

    def _get_y(self):
        paths = [
            os.path.join(
                self.config["dataset_path"],
                self.config["dataset"],
                "fold_{}".format(self.config["fold"]),
                f"small_{split}_csr.pkl",
            )
            for split in ["train", "test", "validation"]
        ]
        data = [load_pkl(p) for p in paths]
        dead_y = torch.tensor(np.concatenate([d[1] for d in data])).long()
        read_y = torch.tensor(np.concatenate([d[2] for d in data])).long()
        return dead_y, read_y

    def forward(self):
        raise NotImplementedError()

    def prepare_batch(self, batch):
        batch_size, n_id, adjs = batch
        x = self.data.x[n_id]
        # idx start from 1, 0 is the padding idx
        batch_x = [x[i].nonzero().squeeze(1) + 1 for i in range(x.size(0))]
        max_len = max([x.shape[-1] for x in batch_x])
        xs = [torch.cat([i, torch.zeros(max_len - i.shape[-1], dtype=int)]) for i in batch_x]
        x = torch.stack(xs).to(self.device)
        # y = self.data.y[n_id[:batch_size]]
        read_y = self.read_y[n_id[:batch_size]]
        dead_y = self.dead_y[n_id[:batch_size]]

        # edge weight not needed for GAT actually
        edge_weight = self.data.edge_attr.to(self.device)
        return x, dead_y, read_y, edge_weight

    def evaluate_step(self, batch, name):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "train")

    def test_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "test")

    def evaluate_epoch(self, outputs, name):
        # do not use .cpu() frequently
        out = np.concatenate([i["out"].cpu().numpy() for i in outputs], axis=0)
        dead_Y = np.concatenate([i["dead_y"].cpu().numpy() for i in outputs], axis=0)
        read_Y = np.concatenate([i["read_y"].cpu().numpy() for i in outputs], axis=0)

        if name == "test":
            self.ssl_load_path = os.path.join(
                self.config["ssl_store_path"],
                "{}_{}_{}".format(self.config["dataset"], self.config["fold"], self.name),
            )
            write_pkl([out, dead_Y, read_Y], self.ssl_load_path)

        # cluster
        if self.cluster_evaluate:
            cluster_res = evaluate_cluster(self.config, self.embed.weight)
            for code_type, d in cluster_res.items():
                for k, v in d.items():
                    self.log(f"{name}_{code_type}_{k}", v, on_epoch=True)

        # linear clf protocal
        # record the mean metrics of k runs and the cf of the last run.
        k_run = self.config["ssl_k_run"]
        startTime = datetime.datetime.now()
        for task in ["ihm", "readmission"]:
            Y = dead_Y if task == "ihm" else read_Y
            all_res = []
            for i in range(k_run):
                X_train, X_test, y_train, y = train_test_split(out, Y, test_size=0.3, shuffle=True, stratify=Y)
                clf = LogisticRegression(n_jobs=8).fit(X_train, y_train)
                proba = clf.predict_proba(X_test)
                all_res.append(print_metrics_binary(y, proba, verbose=0))
            res = {}
            for k in ["auroc", "auprc", "acc", "f1", "minpse"]:
                res[k] = np.mean([all_res[i][k] for i in range(k_run)])
            res["cf"] = all_res[-1]["cf"]
            if task == "ihm":
                # keep the same name for LR scheduler and early stop
                self.log(f"{name}_auroc", res["auroc"], on_epoch=True)
            else:
                self.log(f"{task}_{name}_auroc", res["auroc"], on_epoch=True)
            self.log(f"{task}_{name}_auprc", res["auprc"], on_epoch=True)
            self.log(f"{task}_{name}_acc", res["acc"], on_epoch=True)
            self.log(f"{task}_{name}_f1", res["f1"], on_epoch=True)
            self.log(f"{task}_{name}_minpse", res["minpse"], on_epoch=True)

            if type(self.logger).__name__ == "WandbLogger" and name == "test":
                self.logger.experiment.log(
                    {
                        f"{task}_{name}_cf": wandb.plot.confusion_matrix(
                            probs=proba,
                            y_true=y,
                            class_names=["Alive", "Dead"],
                        ),
                        # f"{task}_{name}_prc": wandb.plot.pr_curve(y, proba, labels=["Alive", "Dead"]),
                        # f"{task}_{name}_roc": wandb.plot.roc_curve(y, proba, labels=["Alive", "Dead"]),
                    },
                )
            print("\n", task)
            pprint(res)

        endTime = datetime.datetime.now()
        print("Used time for evaluation:", endTime - startTime)
        return res

    def training_epoch_end(self, outputs):
        try:
            self.evaluate_epoch(outputs, "train")
        except Exception as e:
            print(e)
            pass

    def test_epoch_end(self, outputs):
        return self.evaluate_epoch(outputs, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.config["lr_sch"] == "none":
            return opt
        if self.config["lr_sch"] == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, mode="max")
        elif self.config["lr_sch"] == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, mode="max")
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "train_auroc",
        }

    def _return_dl(self):
        # use all data in unsupervised learning exp
        # masks are not needed
        loader = NeighborSampler(
            self.data.edge_index,
            # node_idx=self.masks[name],
            sizes=self.sample_size,
            batch_size=self.config["bs"],
            shuffle=True,
            pin_memory=False,
            num_workers=self.config["num_workers"],
        )
        return loader

    def train_dataloader(self):
        return self._return_dl()

    def test_dataloader(self):
        return self._return_dl()

    def on_train_start(self):
        seed_everything(self.seed)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    # @staticmethod
    # def load_model(log_dir, **hparams):
    #     """
    #     :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
    #     :param config: list of named arguments, used to update the model hyperparameters
    #     """
    #     assert os.path.exists(log_dir)
    #     # load hparams
    #     with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
    #         config = yaml.load(fp, Loader=yaml.Loader)
    #         config.update(hparams)

    #     dataset, train_loader, subgraph_loader = get_data(config)

    #     model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
    #     print(f'Loading model {model_path.parent.stem}')
    #     args = {
    #         'config': dict(config),
    #         'dataset': dataset,
    #         'train_loader': train_loader,
    #         'subgraph_loader': subgraph_loader,
    #     }
    #     model = Model.load_from_checkpoint(checkpoint_path=str(model_path), **args)
    #     return model, config, dataset, train_loader, subgraph_loader


def evaluate_cluster(config, w):
    # clustering
    w = w.detach().cpu().clone().numpy()
    res = {}
    if config["dataset"] == "eicu":
        dx_code2cluster, dx_code, dx_idx2cluster = get_code2cluster(config["dx_map_path"])
        proc_code2cluster, proc_code, proc_idx2cluster = get_code2cluster(config["proc_map_path"])

        # idx start from 1
        proc_embed = [w[i + 1] for i in range(len(dx_code), len(dx_code) + len(proc_code))]
        proc_cluster_label = list(proc_code2cluster.values())
        kmeans = KMeans(n_clusters=max(proc_cluster_label) + 1).fit(proc_embed)
        res["proc"] = get_cluster_metrics(proc_cluster_label, kmeans.labels_)

        # idx start from 1
        dx_embed = [w[i + 1] for i in range(len(dx_code))]
        dx_cluster_label = list(dx_code2cluster.values())
        kmeans = KMeans(n_clusters=max(dx_cluster_label) + 1).fit(dx_embed)
        res["dx"] = get_cluster_metrics(dx_cluster_label, kmeans.labels_)
    elif config["dataset"] == "mimic-iii":
        pass
    return res


def get_cluster_metrics(label, pred):
    return {
        "nmi": metrics.normalized_mutual_info_score(label, pred),
        "ami": metrics.adjusted_mutual_info_score(label, pred),
        "ri": metrics.rand_score(label, pred),
        "ari": metrics.adjusted_rand_score(label, pred),
    }


def get_code2cluster(code_path):
    """
    return:
    code2cluster: code idx to cluster idx
    code: code string to code idx
    idx2cluster: cluster idx to cluster string
    """
    code = load_pkl(code_path)
    code_list = [k.split("|")[0] for k, v in code.items()]
    code_set = set(code_list)

    cluster2idx = {}
    idx2cluster = {}
    for i, v in enumerate(code_set):
        cluster2idx[v] = i
        idx2cluster[i] = v

    # code idx (start from zero) to cluster idx
    code2cluster = {}

    for k, v in code.items():
        code2cluster[v] = cluster2idx[k.split("|")[0]]

    return code2cluster, code, idx2cluster


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SSL(nn.Module):
    """
    Noise Contrastive, BarlowTwins, SimCLR (SupCon)
    Todo: SimSiam, BYOL
    """

    def __init__(self, config):
        super().__init__()
        self.method = config["ssl_method"]
        self.is_projector = config["is_projector"]
        if self.is_projector:
            sizes = [config["code_embed_dim"]] + list(map(int, config["projector"].split("-")))
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector = nn.Sequential(*layers)

        if self.method == "barlow":
            self.lambd = config["barlow_lambd"]
            bn_size = sizes[-1] if self.is_projector else config["code_embed_dim"]
            self.bn = nn.BatchNorm1d(bn_size, affine=False)
        elif self.method == "simclr" or self.method == "supcon":
            self.supcon = SupConLoss(temperature=config["simclr_temper"])
        elif self.method == "noise_contrast":
            self.noise_type = config["noise_type"]

    def forward(self, h1, h2, labels=None):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        if self.is_projector:
            z1 = self.projector(h1)
            z2 = self.projector(h2)
        else:
            z1 = h1
            z2 = h2

        if self.method == "barlow":
            c = self.bn(z1).T @ self.bn(z2)
            c.div_(z1.shape[0])
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag
        elif self.method == "simclr" or self.method == "supcon":
            features = torch.stack((z1, z2), dim=1)  # (bsz, n_views, feat_size)
            if self.method == "supcon":
                loss = self.supcon(features, labels)
            else:
                loss = self.supcon(features)

            # def mask_correlated_samples(batch_size):
            #     N = 2 * batch_size
            #     mask = torch.ones((N, N), dtype=bool)
            #     mask = mask.fill_diagonal_(0)
            #     for i in range(batch_size):
            #         mask[i, batch_size + i] = 0
            #         mask[batch_size + i, i] = 0
            #     return mask

            # batch_size = z1.shape[0]
            # N = 2 * batch_size
            # mask = mask_correlated_samples(batch_size)

            # z = torch.cat((z1, z2), dim=0)
            # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

            # sim_i_j = torch.diag(sim, batch_size)
            # sim_j_i = torch.diag(sim, -batch_size)

            # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            # negative_samples = sim[mask].reshape(N, -1)

            # labels = torch.zeros(N).to(z1.device).long()
            # logits = torch.cat((positive_samples, negative_samples), dim=1)
            # loss = self.criterion(logits, labels)
            # loss /= N
        elif self.method == "noise_contrast":

            def score(x1, x2):
                return torch.sum(torch.mul(x1, x2), 1)

            pos = score(z1, z2)
            if self.noise_type == "row_shuffle":
                neg1 = score(z2, row_shuffle(z1))
                neg2 = score(z1, row_shuffle(z2))
            elif self.noise_type == "row_column_shuffle":
                neg1 = score(z2, row_column_shuffle(z1))
                neg2 = score(z1, row_column_shuffle(z2))
            one = torch.FloatTensor(neg1.shape[0]).fill_(1).to(z1.device)
            # one = zeros = torch.ones(neg1.shape[0])
            loss = torch.sum(
                -torch.log(1e-8 + torch.sigmoid(pos))
                - torch.log(1e-8 + (one - torch.sigmoid(neg1)))
                - torch.log(1e-8 + (one - torch.sigmoid(neg2)))
            )

        return loss


# def simclr_loss(x1, x2):
#     T = 0.5
#     batch_size, _ = x1.size()

#     x1_abs = x1.norm(dim=1)
#     x2_abs = x2.norm(dim=1)

#     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum(
#         'i,j->ij', x1_abs, x2_abs
#     )
#     sim_matrix = torch.exp(sim_matrix / T)
#     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#     loss = -torch.log(loss).mean()
#     return loss


# def info_nce_loss(self, features):
#     labels = torch.cat(
#         [torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0
#     )
#     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#     labels = labels.to(self.args.device)

#     features = F.normalize(features, dim=1)

#     similarity_matrix = torch.matmul(features, features.T)

#     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
#     labels = labels[~mask].view(labels.shape[0], -1)
#     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#     # assert similarity_matrix.shape == labels.shape

#     # select and combine multiple positives
#     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#     # select only the negatives the negatives
#     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#     logits = torch.cat([positives, negatives], dim=1)
#     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

#     logits = logits / self.args.temperature
#     return logits, labels


# def noise_contrast(embed1, embed2, method='row_shuffle'):
#     def score(x1, x2):
#         return torch.sum(torch.mul(x1, x2), 1)

#     pos = score(embed1, embed2)
#     if method == 'row_shuffle':
#         neg1 = score(embed2, row_shuffle(embed1))
#         neg2 = score(embed1, row_shuffle(embed2))
#     elif method == 'row_column_shuffle':
#         neg1 = score(embed2, row_column_shuffle(embed1))
#         neg2 = score(embed1, row_column_shuffle(embed2))
#     one = torch.FloatTensor(neg1.shape[0]).fill_(1).to(self.device)
#     # one = zeros = torch.ones(neg1.shape[0])
#     con_loss = torch.sum(
#         -torch.log(1e-8 + torch.sigmoid(pos))
#         - torch.log(1e-8 + (one - torch.sigmoid(neg1)))
#         - torch.log(1e-8 + (one - torch.sigmoid(neg2)))
#     )
#     return con_loss


def row_shuffle(embedding):
    corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
    return corrupted_embedding


def row_column_shuffle(embedding):
    corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
    corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
    return corrupted_embedding


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self, temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # z = torch.cat((z1, z2), dim=0)
        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()
        # loss
        # loss = - (temperature / base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        # return loss


class LinEva(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        x, dead_y, read_y = load_pkl(config["ssl_load_path"])
        if config["ssl_eva_task"] == "ihm":
            y = dead_y
        else:
            y = read_y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
        bs = 2048
        self.train_dl = DataLoader(
            SSLDataset(x_train, y_train),
            batch_size=bs,
            shuffle=True,
            pin_memory=True,
            # num_workers=self.config['num_workers'],
        )
        self.test_dl = DataLoader(
            SSLDataset(x_test, y_test),
            batch_size=bs,
            shuffle=True,
            pin_memory=False,
            # num_workers=self.config['num_workers'],
        )

        self.learning_rate = config["lr"]
        self.loss = define_loss(config)
        self.fc = nn.Linear(x.shape[1], 2)

    def forward(self, x):
        return self.fc(x)

    def evaluate_step(self, batch, name):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        proba = F.softmax(out, dim=-1).detach()
        if name == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return {"loss": loss, "y": y.detach(), "proba": proba}
        else:
            return {f"{name}_loss": loss.detach(), "y": y.detach(), "proba": proba}

    def training_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "train")

    def test_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "test")

    def evaluate_epoch(self, outputs, name):
        # do not use .cpu() frequently
        proba = np.concatenate([i["proba"].cpu().numpy() for i in outputs], axis=0)
        y = np.concatenate([i["y"].cpu().numpy() for i in outputs], axis=0)
        # print(name, y.shape, proba.shape)
        # print(y)
        # print(proba)
        try:
            res = print_metrics_binary(y, proba, verbose=0)
            self.log(f"{name}_auroc", res["auroc"], on_epoch=True)
            self.log(f"{name}_auprc", res["auprc"], on_epoch=True)
            self.log(f"{name}_acc", res["acc"], on_epoch=True)
            self.log(f"{name}_f1", res["f1"], on_epoch=True)
            self.log(f"{name}_minpse", res["minpse"], on_epoch=True)

            if type(self.logger).__name__ == "WandbLogger" and name == "test":
                self.logger.experiment.log(
                    {
                        f"{name}_cf": wandb.plot.confusion_matrix(
                            probs=proba,
                            y_true=y,
                            class_names=["Alive", "Dead"],
                        ),
                        # f"{name}_prc": wandb.plot.pr_curve(y, proba, labels=["Alive", "Dead"]),
                        # f"{name}_roc": wandb.plot.roc_curve(y, proba, labels=["Alive", "Dead"]),
                    },
                )

            print()
            pprint(res)
        except Exception as e:
            print(e)
            pass
        return res

    def training_epoch_end(self, outputs):
        try:
            self.evaluate_epoch(outputs, "train")
        except Exception as e:
            print(e)
            pass

    def test_epoch_end(self, outputs):
        return self.evaluate_epoch(outputs, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.config["lr_sch"] == "none":
            return opt
        if self.config["lr_sch"] == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, mode="max")
        elif self.config["lr_sch"] == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=self.config["patience"], mode="max"
            )
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "train_auroc",
        }

    def train_dataloader(self):
        return self.train_dl

    def test_dataloader(self):
        return self.test_dl

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def run_exp(config, model, model_name, project_name_prefix):
    pprint(config)
    DEBUG = config["debug"]
    if not DEBUG:
        wandb_logger = WandbLogger(
            project=project_name_prefix + config["dataset"],
            save_dir=config["wandb_dir"],
            # job_type='sweep',
            config=config,
            name=model_name,
        )
        print("save to", wandb_logger.save_dir)

    if config["save_model"]:
        checkpoint_callback = ModelCheckpoint(
            # dirpath=wandb.run.dir,
            monitor="validation_auroc",
            filename="{epoch:02d}-{auroc:.2f}-{auprc:.2f}",
            save_top_k=2,
            mode="max",
        )

    early_stop_callback = early_stopping.EarlyStopping(
        monitor="validation_auroc", patience=config["early_stop"], mode="max"
    )

    trainer = pl.Trainer(
        fast_dev_run=DEBUG,
        gpus=config["gpu"],
        max_epochs=config["epoch"],
        default_root_dir=os.path.join(config["log_dir"], model_name),
        callbacks=[early_stop_callback, checkpoint_callback] if config["save_model"] else [early_stop_callback],
        logger=True if DEBUG else wandb_logger,
        auto_select_gpus=True
        # profiler="simple",
        # precision=16 if config['use_amp'] else 32,
        # auto_scale_batch_size='power',
        # auto_lr_find=True,
        # deterministic=True,
        # resume_from_checkpoint=chkpt,
    )
    # trainer.tune(model)
    trainer.fit(model)
    ret = trainer.test(model)


def evaluation(config, task):
    DEBUG = config["debug"]
    model_name = config["model_name"]
    wandb_logger = WandbLogger(
        config=config,
        project="ehr-gnn-ssl-{}".format(config["dataset"]),
        save_dir=config["wandb_dir"],
        job_type="evalutation",
        # tags=['ssl'],
        name=f"{model_name}_eva",
    )
    early_stop_callback = early_stopping.EarlyStopping(monitor="train_auroc", patience=config["early_stop"], mode="max")
    config["ssl_eva_task"] = task
    model = LinEva(config)
    trainer = pl.Trainer(
        fast_dev_run=DEBUG,
        gpus=1,
        max_epochs=config["epoch"],
        default_root_dir=os.path.join(config["log_dir"], model_name),
        callbacks=[early_stop_callback],
        logger=False if DEBUG else wandb_logger,
    )
    trainer.fit(model)
    ret = trainer.test(model)
