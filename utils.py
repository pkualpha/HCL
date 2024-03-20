import json
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

warnings.filterwarnings("ignore")


def get_adjacency(x):
    H_T = x.astype(int)
    BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = BH_T.T
    H = H_T.T
    DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    DHBH_T = np.dot(DH, BH_T)

    adj = DHBH_T.tocoo()
    adj = adj + adj.T
    sadj = adj.multiply(1.0 / adj.sum(axis=1))

    values = sadj.data
    indices = np.vstack((sadj.row, sadj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    shape = sadj.shape
    A = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return A


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_metrics_binary(y_true, proba, verbose=0):
    pred = proba.argmax(axis=1)
    proba = proba[:, 1]
    cf = metrics.confusion_matrix(y_true, pred)
    try:
        auroc = metrics.roc_auc_score(y_true, proba)
    except ValueError:
        auroc = 0
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, proba)
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    acc = metrics.accuracy_score(y_true, pred)
    f1 = metrics.f1_score(y_true, pred)
    if verbose:
        print("confusion matrix:")
        print(cf)
        print("accuracy = {}".format(acc))
        print("f1 = {}".format(f1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
        "acc": acc,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
        "cf": cf,
    }


def write_pkl(data, path, verbose=1):
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("saved to ", path)


def write_json(data, path, sort_keys=False, verbose=1):
    with open(path, "w") as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=4)
    if verbose:
        print("saved to ", path)


def load_json(path):
    with open(path, "r") as json_file:
        info = json.load(json_file)
    return info


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
