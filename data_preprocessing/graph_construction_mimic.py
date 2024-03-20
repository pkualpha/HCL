import os

import numpy as np
from dataloader import EHRDataset
from global_settings import dataset_path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_pkl, write_pkl


def construct_graph(dataset, dataset_path):
    for i in range(10):
        get_graph_raw(dataset, dataset_path, i)
        for k in [3, 5, 7, 9, 11, 13, 15]:
            get_graph_knn(dataset, dataset_path, i, k)


def get_graph_knn(dataset, dataset_path, fold, k=5):
    """
    get patient graph by KNN
    """
    A, (train_mask, val_mask, test_mask) = load_pkl(os.path.join(dataset_path, dataset, f"fold_{fold}", "raw_A.pkl"))
    u = []
    v = []
    val = []
    for i in range(A.shape[0]):
        # do not contain self loop
        u.append(np.ones(k, dtype=int) * i)
        idx = np.argsort(np.array(A[i].todense()).squeeze())[-k - 1 : -1]
        v.append(idx)
        val.append(np.array(A[i, idx].todense()).squeeze())

    u = np.concatenate(u)
    v = np.concatenate(v)
    val = np.concatenate(val)

    write_pkl((u, v, val), os.path.join(dataset_path, dataset, f"fold_{fold}", f"k{k}.pkl"))


def get_graph_raw(dataset, dataset_path, fold):
    """
    get patient similarity matrix by TF-IDF and cosine similarity.
    """
    datasets = [
        EHRDataset(
            dataset,
            dataset_path,
            fold,
            name,
            "ihm",
        )
        for name in ["train", "test", "validation"]
    ]

    xs = [np.array(d.x.todense(), dtype=int) for d in datasets]
    print([i.shape for i in xs])

    xs = np.concatenate(xs)
    print(xs.shape)

    tf = xs.sum(axis=1)
    tfi = 1 / tf
    print(tf.shape, tf.max(), tf.min(), tfi.max(), tfi.min())
    tf = xs / np.expand_dims(tf, -1)
    print(tf.shape, tf.max(), tf[tf > 0].min(), tf[tf > 0].mean())

    idf = np.log(xs.shape[0] / (1 + xs.sum(axis=0)))
    print(idf.shape, idf.max(), idf.min(), idf.mean())

    M = tf * idf
    del (
        tf,
        idf,
        tfi,
        xs,
    )
    print(M.shape, M.max(), M[M > 0].min(), M[M > 0].mean())

    print("compute cosine similarity")
    Ms = csr_matrix(M)
    A = cosine_similarity(Ms, dense_output=False)
    del Ms, M
    print(A.shape)

    A.data *= A.data > 0.2
    A.eliminate_zeros()
    # cosine_similarity is in [0, 1]
    # ignore edges with low cos smilarity
    # only 9 nodes have less than 10 neighbours
    # 67167120 nonzero values, 2.65% of original fully connected adjacency matrix
    print(A.shape)

    # creating train/val/test mask for NeighbourSampler
    spl = [np.array(d.x.todense(), dtype=int).shape[0] for d in datasets]
    train_mask = np.zeros(sum(spl), dtype=int)
    val_mask = np.zeros(sum(spl), dtype=int)
    test_mask = np.zeros(sum(spl), dtype=int)

    train_mask[range(spl[0])] = 1
    val_mask[range(spl[0], spl[0] + spl[1])] = 1
    test_mask[range(spl[0] + spl[1], spl[0] + spl[1] + spl[2])] = 1

    write_pkl(
        (A, (train_mask, val_mask, test_mask)),
        os.path.join(dataset_path, dataset, f"fold_{fold}", "raw_A.pkl"),
    )


if __name__ == "__main__":
    construct_graph("mimic", dataset_path)
