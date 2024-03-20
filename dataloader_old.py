import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data, NeighborSampler
from torchvision import transforms

from utils import load_json


def slice_data(data, info, split):
    """Slice data according to the instances belonging to each split."""
    if split is None:
        return data
    elif split == 'train':
        return data[: info['train_len']]
    elif split == 'val':
        train_n = info['train_len']
        val_n = train_n + info['val_len']
        return data[train_n:val_n]
    elif split == 'test':
        val_n = info['train_len'] + info['val_len']
        test_n = val_n + info['test_len']
        return data[val_n:test_n]


def read_mm(datadir, name):
    """
    load raw data
    """
    info = load_json(Path(datadir) / (name + '_info.json'))
    dat_path = Path(datadir) / (name + '.dat')
    data = np.memmap(dat_path, dtype=np.float32, shape=tuple(info['shape']))
    return data, info


def collect_diag_labels(config, split=None):
    data_dir = config['data_dir']
    task = config['task']

    diag_data, diag_info = read_mm(data_dir, 'diagnoses')
    diag = slice_data(diag_data, diag_info, split)
    diag = np.array(diag)
    patient_id, x = diag[:, 0], diag[:, 1:]

    label_data, labels_info = read_mm(data_dir, 'labels')
    labels = slice_data(label_data, labels_info, split)
    idx2col = {'ihm': 1, 'los': 3, 'multi': [1, 3]}
    label_idx = idx2col[task]
    labels = labels[:, label_idx]
    y = np.array(labels, dtype=int)

    return x, y, patient_id, diag_info, labels_info


class EHRDataset(Dataset):
    """
    Dataset class for EHR data
    Features: diagnose
    Tasks: ihm, los

    Todo: add timeseries, flat, treatment, features
    """

    def __init__(self, config, split=None, semi=False):
        super().__init__()
        (
            self.x,
            self.y,
            self.patient_id,
            self.diag_info,
            self.labels_info,
        ) = collect_diag_labels(config, split=split)

        # ratio = config['label_ratio']
        # if split == 'train' and semi and ratio < 1:
        #     n = self.x.shape[0]
        #     rd = np.random.choice(range(n), size=int(n*ratio), replace=False)
        #     mask = np.zeros(n, dtype=int)
        #     mask[rd] = 1
        #     self.x = self.x[mask]
        #     self.y = self.y[mask]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def read_graph_edge_list(graph_dir, version):
    """
    return edge lists, and edge similarity scores from specified graph.
    """
    version2filename = {'default': 'k_closest_{}_k=3_adjusted_ns.txt'}

    file_name = version2filename[version]
    u_path = Path(graph_dir) / file_name.format('u')
    v_path = Path(graph_dir) / file_name.format('v')
    scores_path = Path(graph_dir) / file_name.format('scores')
    u_list = read_txt(u_path)
    v_list = read_txt(v_path)
    if os.path.exists(scores_path):
        scores = read_txt(scores_path, node=False)
    else:
        scores = None
    return u_list, v_list, scores


def read_txt(path, node=True):
    """
    read raw txt file into lists
    """
    with open(path, "r") as u:
        u_list = u.read()
    if node:
        return [int(n) for n in u_list.split('\n') if n != '']
    else:
        return [float(n) for n in u_list.split('\n') if n != '']


def get_edge_index(us, vs, scores=None):
    """
    return edge data according to pytorch-geometric's specified formats.
    """
    both_us = np.concatenate([us, vs])  # both directions
    both_vs = np.concatenate([vs, us])  # both directions
    edge = np.stack([both_us, both_vs], 0)
    edge_index = torch.tensor(edge, dtype=torch.long)
    if scores is None:
        num_edges = edge_index.shape[1]
        scores = np.random.rand(num_edges, 1)
    else:
        scores = np.concatenate([scores, scores])[:, None]
    scores = torch.tensor(scores).float()
    return edge_index, scores


def _sample_mask(idx, l):
    """Create sample mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def define_node_masks(N, train_n, val_n, ratio=1):
    """
    define node masks according to train / val / test split
    ratio defines the size of training set for semi-supervised learning
    """
    idx_train = range(train_n)
    idx_val = range(train_n, train_n + val_n)
    idx_test = range(train_n + val_n, N)
    train_mask = torch.BoolTensor(_sample_mask(idx_train, N))
    if ratio < 1:
        rd = np.random.choice(idx_train, size=int(train_n * (1 - ratio)), replace=False)
        train_mask[rd] = False
    val_mask = torch.BoolTensor(_sample_mask(idx_val, N))
    test_mask = torch.BoolTensor(_sample_mask(idx_test, N))
    return train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


class GraphDataset(Dataset):
    """
    Dataset class for graph data
    """

    def __init__(self, config):
        super().__init__()
        self.ehr_data = EHRDataset(config)

        N = self.ehr_data.diag_info['total']
        train_n = self.ehr_data.diag_info['train_len']
        val_n = self.ehr_data.diag_info['val_len']

        # Get the edges
        us, vs, edge_attr = read_graph_edge_list(
            config['graph_dir'], config['g_version']
        )
        edge_index, edge_attr = get_edge_index(us, vs, edge_attr)

        # define the graph and its features
        x = torch.from_numpy(self.ehr_data.x)
        y = torch.from_numpy(self.ehr_data.y)
        y = y.long() if config['task'] == 'ihm' else y.float()
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

        # define masks
        (
            data.train_mask,
            data.val_mask,
            data.test_mask,
            self.idx_train,
            self.idx_val,
            self.idx_test,
        ) = define_node_masks(N, train_n, val_n, config['label_ratio'])
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr,
            self.data.y,
        )


def get_data(config):
    """
    produce dataloaders for training and validating
    """
    dataset = GraphDataset(config)

    print(f'num_nodes {dataset.data.num_nodes}')
    print(f'num_features {dataset.data.num_features}')
    print(f'num_edges {dataset.data.num_edges}')
    print(f'num_edge_features {dataset.data.num_edge_features}')
    print(f'contains_isolated_nodes {dataset.data.contains_isolated_nodes()}')

    # train loader - only samples from the train nodes
    train_loader = NeighborSampler(
        dataset.data.edge_index,
        node_idx=dataset.data.train_mask,
        sizes=[30, 10],
        batch_size=config['bs'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    # val / test loader - samples from the entire graph
    subgraph_loader = NeighborSampler(
        dataset.data.edge_index,
        node_idx=None,
        sizes=[-1],
        # sizes=sample_sizes,
        batch_size=config['bs'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    return dataset, train_loader, subgraph_loader


class EHRDataModule(pl.LightningDataModule):
    pass


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
        It is used to separate setup logic for trainer.fit and trainer.test.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def transfer_batch_to_device(self, batch, device):
        x = batch['x']
        x = CustomDataWrapper(x)
        batch['x'] = x.to(device)
        return batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch['x'] = transforms(batch['x'])
        return batch
