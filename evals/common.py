import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class EarlyStopping:
    patience: int
    best_score: float = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait: int = 0

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_indices(indices: Sequence[int]) -> List[int]:
    return [int(i) for i in indices]


def validate_index_inputs(labeled_indices: Sequence[int], unlabeled_indices: Sequence[int], n_train: int) -> None:
    lb = normalize_indices(labeled_indices)
    ulb = normalize_indices(unlabeled_indices)
    if len(lb) == 0:
        raise ValueError("labeled_indices must not be empty")
    if min(lb + ulb) < 0 or max(lb + ulb) >= n_train:
        raise ValueError("indices out of range for training split")
    if len(set(lb).intersection(set(ulb))) > 0:
        raise ValueError("labeled_indices and unlabeled_indices must be disjoint")


def default_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def cifar_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def cifar_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


class CIFARLabeledSubset(Dataset):
    def __init__(self, base_data: np.ndarray, base_targets: Sequence[int], indices: Sequence[int], transform) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = base_data
        self.targets = np.asarray(base_targets, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img = self.data[base_idx]
        target = int(self.targets[base_idx])
        img = transforms.functional.to_pil_image(img)
        return self.transform(img), target


class CIFARUnlabeledSubset(Dataset):
    def __init__(self, base_data: np.ndarray, indices: Sequence[int], weak_transform, strong_transform) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = base_data
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img = self.data[base_idx]
        img = transforms.functional.to_pil_image(img)
        return idx, self.weak_transform(img), self.strong_transform(img)


def split_labeled_train_val(
    labeled_indices: Sequence[int],
    train_targets: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    lb = np.asarray(labeled_indices, dtype=np.int64)
    if len(lb) < 2 or val_ratio <= 0:
        return lb.tolist(), lb.tolist()

    y = np.asarray(train_targets, dtype=np.int64)[lb]
    try:
        lb_train, lb_val = train_test_split(
            lb,
            test_size=val_ratio,
            random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    except ValueError:
        lb_train, lb_val = train_test_split(lb, test_size=val_ratio, random_state=seed, stratify=None)
    return lb_train.tolist(), lb_val.tolist()


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return float(correct / max(total, 1))
