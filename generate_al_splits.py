"""
Generate labeled/unlabeled index splits for each (method, budget) pair
and save them as individual JSON files in datasets/al_splits/.

Output: datasets/al_splits/{method}_{budget}.json
Format: {"labeled_indices": [...], "unlabeled_indices": [...]}

Budgets are cumulative: random_20.json includes the same 10 indices as
random_10.json plus 10 more selected by the strategy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_DEEPAL_DIR = _ROOT / "deep-active-learning"
if str(_DEEPAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPAL_DIR))

from data import get_CIFAR10          # type: ignore
from handlers import CIFAR10_Handler  # type: ignore
from utils import get_net             # type: ignore

from al_methods import STRATEGY_REGISTRY, select_indices, typiclust

BUDGETS = [10, 20, 30, 40, 50, 60, 100, 150, 200, 250, 300]
METHODS = sorted(STRATEGY_REGISTRY.keys()) + ["typiclust"]
EMBEDDINGS_PATH = str(_ROOT / "datasets" / "cifar10_train_embeddings.npz")
OUTPUT_DIR = _ROOT / "datasets" / "al_splits"
SEED = 42


def _sorted_ints(values) -> list[int]:
    return sorted(int(v) for v in values)


def _run_method(method: str) -> None:
    budgets = sorted(BUDGETS)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = get_CIFAR10(CIFAR10_Handler)
    dataset.initialize_labels(budgets[0])

    labeled = np.where(dataset.labeled_idxs)[0].tolist()
    unlabeled = np.where(~dataset.labeled_idxs)[0].tolist()

    net = None
    if method in STRATEGY_REGISTRY and method != "random":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_net("CIFAR10", device)

    current_labeled_count = len(labeled)

    for budget in budgets:
        increment = budget - current_labeled_count

        if increment > 0:
            if method in STRATEGY_REGISTRY:
                selected, unlabeled = select_indices(
                    dataset=dataset,
                    budget=increment,
                    strategy=method,
                    net=net,
                    train_before_query=(net is not None),
                    update_dataset_labels=True,
                )
                labeled = np.where(dataset.labeled_idxs)[0].tolist()

            elif method == "typiclust":
                selected, unlabeled = typiclust(
                    dataset=dataset,
                    budget=budget, # we always cluster with "budget" number of clusters, in this implementation, we do not incrementally add clusters, but re-cluster with the new budget at each step
                    embeddings_npz_path=EMBEDDINGS_PATH,
                )
                for idx in selected:
                    dataset.labeled_idxs[idx] = True
                labeled = np.where(dataset.labeled_idxs)[0].tolist()

            current_labeled_count = len(labeled)

        out_path = OUTPUT_DIR / f"{method}_{budget}.json"
        out_path.write_text(
            json.dumps({
                "labeled_indices": _sorted_ints(labeled),
                "unlabeled_indices": _sorted_ints(unlabeled),
            }),
            encoding="utf-8",
        )
        print(f"  saved {out_path.name}  ({len(labeled)} labeled)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        print(f"[{method}]")
        _run_method(method)
    print("Done.")


if __name__ == "__main__":
    main()
