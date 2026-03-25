from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from al_methods import STRATEGY_REGISTRY, select_indices, typiclust


_DEEPAL_DIR = Path(__file__).resolve().parent / "deep-active-learning"
if str(_DEEPAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPAL_DIR))

from data import Data, get_CIFAR10  # type: ignore  # noqa: E402
from handlers import CIFAR10_Handler  # type: ignore  # noqa: E402
from utils import get_net  # type: ignore  # noqa: E402


DEFAULT_BUDGETS = [10, 20, 30, 40, 50, 60, 100, 150, 200, 250, 300]
DEFAULT_METHODS = sorted(list(STRATEGY_REGISTRY.keys())) + ["typiclust"]


def _sorted_unique_ints(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def _clone_dataset(base: Data) -> Data:
    # Data stores X/Y arrays and a labeled mask; we can safely share X/Y.
    cloned = Data(base.X_train, base.Y_train, base.X_test, base.Y_test, base.handler)
    return cloned


def _set_initial_labels(dataset: Data, labeled_indices: list[int]) -> None:
    dataset.labeled_idxs[:] = False
    dataset.labeled_idxs[np.asarray(labeled_indices, dtype=np.int64)] = True


def _get_labeled_unlabeled(dataset: Data) -> tuple[list[int], list[int]]:
    labeled = np.arange(dataset.n_pool)[dataset.labeled_idxs].astype(np.int64).tolist()
    unlabeled = np.arange(dataset.n_pool)[~dataset.labeled_idxs].astype(np.int64).tolist()
    return _sorted_unique_ints(labeled), _sorted_unique_ints(unlabeled)


def _write_split(out_dir: Path, method: str, budget: int, labeled: list[int], unlabeled: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "labeled_indices": labeled,
        "unlabeled_indices": unlabeled,
    }
    (out_dir / f"{method}_{budget}.json").write_text(json.dumps(payload), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate labeled/unlabeled index splits for AL methods over multiple budgets."
    )
    p.add_argument("--budgets", type=int, nargs="*", default=DEFAULT_BUDGETS)
    p.add_argument("--methods", type=str, nargs="*", default=DEFAULT_METHODS)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("datasets") / "al_splits"),
    )
    p.add_argument(
        "--embeddings-npz",
        type=str,
        default=str(Path("datasets") / "cifar10_train_embeddings.npz"),
        help="Required for typiclust.",
    )
    p.add_argument("--train-before-query", action="store_true", default=True)
    p.add_argument("--no-train-before-query", dest="train_before_query", action="store_false")
    p.add_argument("--use-cuda", action="store_true", default=True)
    p.add_argument("--cpu", dest="use_cuda", action="store_false")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    budgets = _sorted_unique_ints(list(args.budgets))
    if not budgets:
        raise SystemExit("No budgets provided")

    methods = list(args.methods)
    valid_methods = set(STRATEGY_REGISTRY.keys()) | {"typiclust"}
    unknown = [m for m in methods if m not in valid_methods]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Valid: {sorted(valid_methods)}")

    out_dir = Path(args.output_dir)

    # Load CIFAR10 once; clone per method.
    base_dataset = get_CIFAR10(CIFAR10_Handler)

    init_budget = int(min(budgets))
    if init_budget < 0 or init_budget > base_dataset.n_pool:
        raise SystemExit(f"Invalid init budget: {init_budget}")

    rng = np.random.default_rng()
    init_labeled = rng.choice(base_dataset.n_pool, size=init_budget, replace=False).astype(np.int64).tolist()
    init_labeled = _sorted_unique_ints(init_labeled)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    for method in methods:
        dataset = _clone_dataset(base_dataset)
        _set_initial_labels(dataset, init_labeled)

        net = None
        if method in STRATEGY_REGISTRY and method != "random":
            net = get_net("CIFAR10", device)

        current_labeled_count = int(dataset.labeled_idxs.sum())

        for budget in budgets:
            target = int(budget)
            if target < current_labeled_count:
                labeled, unlabeled = _get_labeled_unlabeled(dataset)
                _write_split(out_dir, method, target, labeled, unlabeled)
                continue

            increment = target - current_labeled_count
            if increment > 0:
                if method in STRATEGY_REGISTRY:
                    _selected, _remaining = select_indices(
                        dataset=dataset,
                        budget=increment,
                        strategy=method,
                        net=net,
                        train_before_query=args.train_before_query,
                        update_dataset_labels=True,
                    )
                else:
                    selected, _remaining = typiclust(
                        dataset=dataset,
                        budget=increment,
                        embeddings_npz_path=str(args.embeddings_npz),
                        random_state=None,
                    )
                    for idx in selected:
                        dataset.labeled_idxs[int(idx)] = True

            labeled, unlabeled = _get_labeled_unlabeled(dataset)
            _write_split(out_dir, method, target, labeled, unlabeled)
            current_labeled_count = len(labeled)


if __name__ == "__main__":
    main()
