import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .common import (
    CIFARLabeledSubset,
    CIFARUnlabeledSubset,
    CIFAR10_MEAN,
    CIFAR10_STD,
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


def _ensure_torchssl_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    torchssl_root = repo_root / "TorchSSL"
    if str(torchssl_root) not in sys.path:
        sys.path.insert(0, str(torchssl_root))


_ensure_torchssl_on_path()
from models.flexmatch.flexmatch_utils import consistency_loss  # noqa: E402
from datasets.augmentation.randaugment import RandAugment  # noqa: E402


def _update_classwise_acc(selected_label: torch.Tensor, num_classes: int) -> torch.Tensor:
    classwise_acc = torch.zeros(num_classes, device=selected_label.device)
    valid = selected_label[selected_label >= 0]
    if valid.numel() == 0:
        return classwise_acc
    counts = torch.bincount(valid, minlength=num_classes).float()
    max_count = torch.max(counts)
    if max_count > 0:
        classwise_acc = counts / max_count
    return classwise_acc


def train_eval_semi_sl_flexmatch(
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    *,
    data_dir: str = "./data",
    batch_size: int = 64,
    uratio: int = 7,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 0.03,
    weight_decay: float = 5e-4,
    lambda_u: float = 1.0,
    p_cutoff: float = 0.95,
    temperature: float = 1.0,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Method 3: Semi-supervised CIFAR-10 with FlexMatch-style supervised + consistency training."""
    set_seed(seed)
    device_obj = default_device(device)

    train_raw = datasets.CIFAR10(root=data_dir, train=True, download=True)
    validate_index_inputs(labeled_indices, unlabeled_indices, len(train_raw.data))

    lb_train_idx, lb_val_idx = split_labeled_train_val(labeled_indices, train_raw.targets, val_ratio, seed)

    weak = cifar_train_transform()
    strong = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_tf = cifar_test_transform()

    lb_train_ds = CIFARLabeledSubset(train_raw.data, train_raw.targets, lb_train_idx, weak)
    lb_val_ds = CIFARLabeledSubset(train_raw.data, train_raw.targets, lb_val_idx, eval_tf)
    ulb_ds = CIFARUnlabeledSubset(train_raw.data, unlabeled_indices, weak, strong)

    test_base = datasets.CIFAR10(root=data_dir, train=False, download=True)
    test_ds = CIFARLabeledSubset(test_base.data, test_base.targets, list(range(len(test_base))), eval_tf)

    lb_loader = DataLoader(lb_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    ulb_loader = DataLoader(
        ulb_ds,
        batch_size=batch_size * uratio,
        shuffle=True,
        drop_last=True,
        num_workers=max(1, num_workers * 2),
    )
    val_loader = DataLoader(lb_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = make_resnet18(num_classes=10).to(device_obj)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    selected_label = torch.full((len(ulb_ds),), -1, dtype=torch.long, device=device_obj)
    stopper = EarlyStopping(patience=patience)
    p_model = None
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        lb_iter = iter(lb_loader)
        ulb_iter = iter(ulb_loader)
        steps = max(1, max(len(lb_loader), len(ulb_loader)))

        for _ in range(steps):
            try:
                x_lb, y_lb = next(lb_iter)
            except StopIteration:
                lb_iter = iter(lb_loader)
                x_lb, y_lb = next(lb_iter)

            try:
                x_ulb_idx, x_ulb_w, x_ulb_s = next(ulb_iter)
            except StopIteration:
                ulb_iter = iter(ulb_loader)
                x_ulb_idx, x_ulb_w, x_ulb_s = next(ulb_iter)

            x_lb = x_lb.to(device_obj, non_blocking=True)
            y_lb = y_lb.to(device_obj, non_blocking=True)
            x_ulb_idx = x_ulb_idx.to(device_obj, non_blocking=True)
            x_ulb_w = x_ulb_w.to(device_obj, non_blocking=True)
            x_ulb_s = x_ulb_s.to(device_obj, non_blocking=True)

            classwise_acc = _update_classwise_acc(selected_label, num_classes=10)
            inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s], dim=0)
            logits = model(inputs)
            num_lb = x_lb.size(0)
            logits_lb = logits[:num_lb]
            logits_ulb_w, logits_ulb_s = logits[num_lb:].chunk(2)

            sup_loss = criterion(logits_lb, y_lb)
            unsup_loss, _, select, pseudo_lb, p_model = consistency_loss(
                logits_ulb_s,
                logits_ulb_w,
                classwise_acc,
                p_target=None,
                p_model=p_model,
                name="ce",
                T=temperature,
                p_cutoff=p_cutoff,
                use_hard_labels=True,
                use_DA=False,
            )

            selected_batch = select == 1
            if selected_batch.any():
                selected_label[x_ulb_idx[selected_batch]] = pseudo_lb[selected_batch]

            loss = sup_loss + lambda_u * unsup_loss
            optimizer.zero_grad(set_to_none=True)
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
        "method": "semi_sl_flexmatch",
        "test_acc": test_acc,
        "best_val_acc": float(stopper.best_score),
        "epochs_run": float(epochs_run),
        "num_labeled": float(len(labeled_indices)),
        "num_unlabeled": float(len(unlabeled_indices)),
    }
