from typing import Dict, Optional, Sequence

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from .common import (
    CIFARLabeledSubset,
    EarlyStopping,
    cifar_test_transform,
    cifar_train_transform,
    default_device,
    evaluate_classifier,
    make_resnet18,
    set_seed,
    split_labeled_train_val,
    validate_index_inputs,
)


def train_eval_fsl(
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    *,
    data_dir: str = "./data",
    batch_size: int = 128,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Method 1: Fully supervised ResNet18 on labeled CIFAR-10 subset only."""
    set_seed(seed)
    device_obj = default_device(device)

    train_raw = datasets.CIFAR10(root=data_dir, train=True, download=True)
    validate_index_inputs(labeled_indices, unlabeled_indices, len(train_raw.data))

    train_idx, val_idx = split_labeled_train_val(labeled_indices, train_raw.targets, val_ratio, seed)
    train_ds = CIFARLabeledSubset(train_raw.data, train_raw.targets, train_idx, cifar_train_transform())
    val_ds = CIFARLabeledSubset(train_raw.data, train_raw.targets, val_idx, cifar_test_transform())

    test_base = datasets.CIFAR10(root=data_dir, train=False, download=True)
    test_ds = CIFARLabeledSubset(test_base.data, test_base.targets, list(range(len(test_base))), cifar_test_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = make_resnet18(num_classes=10).to(device_obj)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience)

    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device_obj, non_blocking=True)
            y = y.to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = evaluate_classifier(model, val_loader, device_obj)
        epochs_run = epoch + 1
        if stopper.step(val_acc, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    test_acc = evaluate_classifier(model, test_loader, device_obj)
    return {
        "method": "fsl",
        "test_acc": test_acc,
        "best_val_acc": float(stopper.best_score),
        "epochs_run": float(epochs_run),
        "num_labeled": float(len(labeled_indices)),
        "num_unlabeled": float(len(unlabeled_indices)),
    }
