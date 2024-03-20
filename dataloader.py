import os

import numpy as np
import torch
from scipy.sparse import vstack
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils import load_pkl


class EHRDataset(Dataset):
    def __init__(self, dataset, path, fold, split, task, return_idx=True):
        """
        return_idx: used for embedding based method.
        If return_idx is True, return x as index + 1 of the nonzero item in the vector.
        If return_idx is False, return x as the multi-hot vecter.
        """
        super().__init__()
        data_path = os.path.join(path, dataset, f"fold_{fold}", f"small_{split}_csr.pkl")
        self.x, self.dead_label, self.read_label = load_pkl(data_path)
        if task == "ihm":
            self.y = self.dead_label
        elif task == "readmission":
            self.y = self.read_label
        self.y = self.y.astype(int)
        self.num_patient = self.x.shape[0]
        self.num_diag = self.x.shape[1]
        self.return_idx = return_idx

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.return_idx:
            return self.x[index].nonzero()[1] + 1, self.y[index]
        else:
            return (
                torch.tensor(self.x[index].todense()).squeeze().float(),
                self.y[index],
            )


def collate_fn(batch):
    max_len = max(map(lambda x: x[0].shape[-1], batch))
    xs = map(lambda x: torch.tensor(x[0], dtype=int), batch)
    xs = [torch.cat([i, torch.zeros(max_len - i.shape[-1], dtype=int)]) for i in xs]
    x = torch.stack(xs)
    y = torch.LongTensor([x[1] for x in batch])
    return x, y


def get_graph_dataset(config):
    dpath = os.path.join(
        config["dataset_path"],
        config["dataset"],
        "fold_{}".format(config["fold"]),
        "k{}.pkl".format(config["k_neighbour"]),
    )
    (u, v, val) = load_pkl(dpath)
    ae = np.stack([u, v], axis=1)
    st = set([(ae[i][0], ae[i][1]) for i in range(ae.shape[0])])
    # std = set([(ae[i][0], ae[i][1]) for i in range(ae.shape[0])]+[(ae[i][1], ae[i][0]) for i in range(ae.shape[0])])

    nu = []
    nv = []
    nval = []
    for i in range(ae.shape[0]):
        if (v[i], u[i]) not in st:
            nu.append(v[i])
            nv.append(u[i])
            nval.append(val[i])

    nu = np.concatenate([u, np.array(nu)])
    nv = np.concatenate([v, np.array(nv)])
    nval = np.concatenate([val, np.array(nval)])

    edge = np.stack([nu, nv], 0)
    edge_index = torch.tensor(edge, dtype=torch.long)
    nval = np.concatenate([nval, nval])[:, None]
    edge_attr = torch.tensor(nval).float()

    datasets = [
        EHRDataset(
            config["dataset"],
            config["dataset_path"],
            config["fold"],
            name,
            config["task"],
        )
        for name in ["train", "test", "validation"]
    ]

    x = torch.tensor(vstack([d.x for d in datasets]).todense(), dtype=int)
    y = torch.tensor(np.concatenate([d.y for d in datasets])).long()
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    print(f"num_nodes {data.num_nodes}")
    print(f"num_features {data.num_features}")
    print(f"num_edges {data.num_edges}")
    print(f"num_edge_features {data.num_edge_features}")
    print(f"contains_isolated_nodes {data.contains_isolated_nodes()}")

    A, (train_mask, val_mask, test_mask) = load_pkl(
        os.path.join(
            config["dataset_path"],
            config["dataset"],
            "fold_{}".format(config["fold"]),
            "raw_A.pkl",
        ),
    )
    return (
        data,
        torch.BoolTensor(train_mask),
        torch.BoolTensor(val_mask),
        torch.BoolTensor(test_mask),
    )

    # sample_size = config['sample_size']
    # # train loader - only samples from the train nodes
    # train_loader = NeighborSampler(
    #     data.edge_index,
    #     node_idx=train_mask,
    #     sizes=sample_size,
    #     batch_size=config['bs'],
    #     shuffle=True,
    #     num_workers=config['num_workers'],
    # )
    # # val / test loader still samples subgraph
    # # TODO: samples from the entire graph
    # validation_loader = NeighborSampler(
    #     data.edge_index,
    #     node_idx=val_mask,
    #     sizes=sample_size,
    #     batch_size=config['bs'],
    #     shuffle=False,
    #     num_workers=config['num_workers'],
    # )

    # test_loader = NeighborSampler(
    #     data.edge_index,
    #     node_idx=test_mask,
    #     sizes=sample_size,
    #     batch_size=config['bs'],
    #     shuffle=False,
    #     num_workers=config['num_workers'],
    # )
    # return data, train_loader, validation_loader, test_loader


class SSLDataset(Dataset):
    def __init__(self, x, y):
        """
        return_idx: used for embedding based method.
        If return_idx is True, return x as index + 1 of the nonzero item in the vector.
        If return_idx is False, return x as the multi-hot vecter.
        """
        super().__init__()
        self.x = x
        self.y = y.astype(int)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index]).squeeze().float(),
            self.y[index],
        )
