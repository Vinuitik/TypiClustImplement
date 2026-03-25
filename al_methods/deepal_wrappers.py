from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys


_DEEPAL_DIR = Path(__file__).resolve().parent.parent / "deep-active-learning"
if str(_DEEPAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPAL_DIR))

from query_strategies import (  # type: ignore  # noqa: E402
    BALDDropout,
    EntropySampling,
    EntropySamplingDropout,
    KCenterGreedy,
    KMeansSampling,
    LeastConfidence,
    LeastConfidenceDropout,
    MarginSampling,
    MarginSamplingDropout,
    RandomSampling,
)
from .typiclust import typiclust as typiclust_algo


STRATEGY_REGISTRY = {
    "random": RandomSampling,
    "least_confidence": LeastConfidence,
    "margin": MarginSampling,
    "entropy": EntropySampling,
    "least_confidence_dropout": LeastConfidenceDropout,
    "margin_dropout": MarginSamplingDropout,
    "entropy_dropout": EntropySamplingDropout,
    "kmeans": KMeansSampling,
    "kcenter": KCenterGreedy,
    "bald_dropout": BALDDropout,
}

_RANDOM_ONLY = {"random"}


def _to_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _get_unlabeled_idxs(dataset) -> list[int]:
    unlabeled_idxs, _ = dataset.get_unlabeled_data()
    return _to_int_list(unlabeled_idxs)


def select_indices(
    dataset,
    budget: int,
    strategy: str = "random",
    net=None,
    train_before_query: bool = False,
    update_dataset_labels: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Run one DeepAL query step and return:
    1) indices to label
    2) indices to keep unlabeled

    Parameters
    ----------
    dataset:
        A DeepAL-compatible dataset object (same interface as data.Data).
    budget:
        Number of samples to select.
    strategy:
        One of keys in STRATEGY_REGISTRY.
    net:
        DeepAL net object. Required for all strategies except random.
    train_before_query:
        If True, call strategy.train() before strategy.query().
    update_dataset_labels:
        If True, mark selected indices as labeled in dataset.labeled_idxs.
    """
    if strategy not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{strategy}'. Valid values: {valid}")

    if budget < 0:
        raise ValueError("budget must be >= 0")

    if strategy not in _RANDOM_ONLY and net is None:
        raise ValueError(f"strategy '{strategy}' requires a trained DeepAL net")

    unlabeled_before = _get_unlabeled_idxs(dataset)
    if not unlabeled_before or budget == 0:
        return [], unlabeled_before

    n_query = min(int(budget), len(unlabeled_before))
    strategy_cls = STRATEGY_REGISTRY[strategy]
    strategy_obj = strategy_cls(dataset, net)

    if train_before_query and strategy not in _RANDOM_ONLY:
        strategy_obj.train()

    selected = _to_int_list(strategy_obj.query(n_query))

    # Keep original unlabeled order while removing selected elements.
    selected_set = set(selected)
    remaining_unlabeled = [idx for idx in unlabeled_before if idx not in selected_set]

    if update_dataset_labels:
        strategy_obj.update(selected)

    return selected, remaining_unlabeled


def random(dataset, budget: int) -> tuple[list[int], list[int]]:
    return select_indices(dataset=dataset, budget=budget, strategy="random")


def least_confidence(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="least_confidence",
        net=net,
        train_before_query=train_before_query,
    )


def margin(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="margin",
        net=net,
        train_before_query=train_before_query,
    )


def entropy(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="entropy",
        net=net,
        train_before_query=train_before_query,
    )


def least_confidence_dropout(
    dataset, budget: int, net, train_before_query: bool = True
) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="least_confidence_dropout",
        net=net,
        train_before_query=train_before_query,
    )


def margin_dropout(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="margin_dropout",
        net=net,
        train_before_query=train_before_query,
    )


def entropy_dropout(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="entropy_dropout",
        net=net,
        train_before_query=train_before_query,
    )


def kmeans(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="kmeans",
        net=net,
        train_before_query=train_before_query,
    )


def kcenter(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="kcenter",
        net=net,
        train_before_query=train_before_query,
    )


def bald_dropout(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="bald_dropout",
        net=net,
        train_before_query=train_before_query,
    )


def adversarial_bim(dataset, budget: int, net, train_before_query: bool = True) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="adversarial_bim",
        net=net,
        train_before_query=train_before_query,
    )


def adversarial_deepfool(
    dataset, budget: int, net, train_before_query: bool = True
) -> tuple[list[int], list[int]]:
    return select_indices(
        dataset=dataset,
        budget=budget,
        strategy="adversarial_deepfool",
        net=net,
        train_before_query=train_before_query,
    )


def typiclust(
    dataset,
    budget: int,
    embeddings_npz_path: str = None,
    k: int = 20,
    random_state: int | None = 42,
) -> tuple[list[int], list[int]]:
    return typiclust_algo(
        dataset=dataset,
        budget=budget,
        embeddings_npz_path=embeddings_npz_path,
        k=k,
        random_state=random_state,
    )



