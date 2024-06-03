from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn)
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from torch.utils.data import DataLoader
from pyhealth.data import Event, Visit, Patient
import torch

def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


def get_dataloader(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict,
    )

    return dataloader

def prototyping(X, k, max_iters=1000, tol=1e-4, alpha=0.1):
    prototypes = X[torch.randperm(X.size(0))[:k]]
    for i in range(max_iters):
        distances = torch.cdist(X, prototypes)
        labels = torch.argmin(distances, dim=1)
        prototypes_old = prototypes.clone()
        for j in range(k):
            cluster_points = X[labels == j]
            for m in range(len(cluster_points)):
                prototypes[j] = (1 - alpha) * prototypes[j] + alpha * cluster_points[m]

        if torch.norm(prototypes - prototypes_old) < tol:
            print(f"Converged after {i} iterations")
            break

    return labels, prototypes

def calc_label_freq(dataset):
    all_num = len(dataset)
    positive_num = len([sample for sample in dataset if sample['label'] == 1])
    negative_num = len([sample for sample in dataset if sample['label'] == 0])
    positive_frequency = positive_num / all_num
    negative_frequency = negative_num / all_num
    return positive_num, negative_num, positive_frequency, negative_frequency