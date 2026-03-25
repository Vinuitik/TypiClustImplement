from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from al_methods import (
	STRATEGY_REGISTRY,
	select_indices,
	typiclust,
)
from evals.frameworks import train_eval_fsl


_DEEPAL_DIR = Path(__file__).resolve().parent / "deep-active-learning"
import sys

if str(_DEEPAL_DIR) not in sys.path:
	sys.path.insert(0, str(_DEEPAL_DIR))

from data import get_CIFAR10  # type: ignore  # noqa: E402
from handlers import CIFAR10_Handler  # type: ignore  # noqa: E402
from utils import get_net  # type: ignore  # noqa: E402


DEFAULT_BUDGETS = [10, 20, 30, 40, 50, 60, 100, 150, 200, 250, 300]
DEFAULT_METHODS = sorted(list(STRATEGY_REGISTRY.keys())) + ["typiclust", "typiclust_rp"]


def _sorted_unique_ints(values: list[int]) -> list[int]:
	return sorted(set(int(v) for v in values))


def _cifar10_dataset(seed: int):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	return get_CIFAR10(CIFAR10_Handler)


def _run_single_method(
	method: str,
	budgets: list[int],
	seed: int,
	train_before_query: bool,
	use_cuda: bool,
	embeddings_npz_path: str | None,
	typicality_npz_path: str | None,
	suppress_k: int,
	decay_factor: float,
	prop_k: int,
) -> tuple[str, dict[str, dict[str, list[int]]]]:
	budgets = _sorted_unique_ints(budgets)
	if not budgets:
		return method, {}

	dataset = _cifar10_dataset(seed)
	init_labels = budgets[0]
	dataset.initialize_labels(init_labels)

	labeled_indices = np.arange(dataset.n_pool)[dataset.labeled_idxs].astype(np.int64).tolist()
	unlabeled_indices = np.arange(dataset.n_pool)[~dataset.labeled_idxs].astype(np.int64).tolist()

	active_scores = None
	if method in {"typiclust", "typiclust_rp"} and typicality_npz_path is not None:
		with np.load(typicality_npz_path) as pack:
			if "typicality" in pack:
				active_scores = np.asarray(pack["typicality"], dtype=np.float32).copy()

	net = None
	if method in STRATEGY_REGISTRY and method != "random":
		device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
		net = get_net("CIFAR10", device)

	outputs: dict[str, dict[str, list[int]]] = {}
	current_labeled = len(labeled_indices)

	for budget in budgets:
		target_budget = int(budget)
		if target_budget < current_labeled:
			outputs[str(target_budget)] = {
				"labeled_indices": _sorted_unique_ints(labeled_indices),
				"unlabeled_indices": _sorted_unique_ints(unlabeled_indices),
				"accuracy": 0.0,
			}
			continue

		increment = target_budget - current_labeled
		if increment > 0:
			if method in STRATEGY_REGISTRY:
				selected, remaining = select_indices(
					dataset=dataset,
					budget=increment,
					strategy=method,
					net=net,
					train_before_query=train_before_query,
					update_dataset_labels=True,
				)
				_ = selected
				unlabeled_indices = remaining
				labeled_indices = np.arange(dataset.n_pool)[dataset.labeled_idxs].astype(np.int64).tolist()


			elif method == "typiclust_rp":
				anchors, after_anchor = typiclust_precomputed(
					labeled_indices=labeled_indices,
					unlabeled_indices=unlabeled_indices,
					budget=increment,
					embeddings_npz_path=embeddings_npz_path,
					typicality_npz_path=typicality_npz_path,
					suppress_k=suppress_k,
					decay_factor=decay_factor,
					random_state=seed + target_budget,
					active_scores=active_scores,
				)
				propagated, after_prop = typiclust_propagation(
					anchor_indices=anchors,
					unlabeled_indices=after_anchor,
					prop_k=prop_k,
					typicality_npz_path=typicality_npz_path,
				)
				labeled_indices = _sorted_unique_ints(labeled_indices + anchors + propagated)
				unlabeled_indices = after_prop
			elif method == "typiclust":
				selected, remaining = typiclust(
					dataset=dataset,
					budget=increment,
					embeddings_npz_path=embeddings_npz_path,
				)
				for a in selected:
					dataset.labeled_idxs[a] = True
				labeled_indices = np.arange(dataset.n_pool)[dataset.labeled_idxs].astype(np.int64).tolist()
				unlabeled_indices = remaining
			else:
				raise ValueError(f"Unknown method: {method}")

		current_labeled = len(labeled_indices)
		
		# Train and evaluate FSL (resnet18 from scratch) on selected indices
		metrics = train_eval_fsl(
			labeled_indices=labeled_indices,
			unlabeled_indices=unlabeled_indices,
			max_epochs=50, # Keep short for AL loop
		)
		acc = metrics.get("test_acc", 0.0)
		print(f"[{method}] Budget: {target_budget} | Test Acc: {acc:.4f}")

		outputs[str(target_budget)] = {
			"labeled_indices": _sorted_unique_ints(labeled_indices),
			"unlabeled_indices": _sorted_unique_ints(unlabeled_indices),
			"accuracy": acc,
		}

	return method, outputs


def generate_budget_indices_parallel(
	methods: list[str] | None = None,
	budgets: list[int] | None = None,
	seed: int = 42,
	train_before_query: bool = True,
	use_cuda: bool = True,
	max_workers: int | None = None,
	embeddings_npz_path: str | None = None,
	typicality_npz_path: str | None = None,
	suppress_k: int = 20,
	decay_factor: float = 0.1,
	prop_k: int = 5,
) -> dict[str, dict[str, dict[str, list[int]]]]:
	selected_methods = methods if methods is not None else DEFAULT_METHODS
	selected_budgets = budgets if budgets is not None else DEFAULT_BUDGETS

	results: dict[str, dict[str, dict[str, list[int]]]] = {}
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = {
			executor.submit(
				_run_single_method,
				method,
				selected_budgets,
				seed,
				train_before_query,
				use_cuda,
				embeddings_npz_path,
				typicality_npz_path,
				suppress_k,
				decay_factor,
				prop_k,
			): method
			for method in selected_methods
		}

		for fut in as_completed(futures):
			method_name = futures[fut]
			method, method_result = fut.result()
			results[method] = method_result
			print(f"[done] {method_name}")

	return results


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate labeled/unlabeled index lists for AL methods by budget.")
	parser.add_argument(
		"--budgets",
		type=int,
		nargs="*",
		default=DEFAULT_BUDGETS,
		help="Target cumulative labeled budgets.",
	)
	parser.add_argument(
		"--methods",
		type=str,
		nargs="*",
		default=DEFAULT_METHODS,
		help="Method names to run. Supports DeepAL registry keys + typiclust + typiclust_rp.",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train-before-query", action="store_true", default=True)
	parser.add_argument("--no-train-before-query", dest="train_before_query", action="store_false")
	parser.add_argument("--use-cuda", action="store_true", default=True)
	parser.add_argument("--cpu", dest="use_cuda", action="store_false")
	parser.add_argument("--max-workers", type=int, default=None)
	parser.add_argument("--embeddings-npz", type=str, default=None)
	parser.add_argument("--typicality-npz", type=str, default=None)
	parser.add_argument("--suppress-k", type=int, default=20)
	parser.add_argument("--decay-factor", type=float, default=0.1)
	parser.add_argument("--prop-k", type=int, default=5)
	parser.add_argument("--output", type=str, default="al_budget_indices_cifar10.json")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	results = generate_budget_indices_parallel(
		methods=list(args.methods),
		budgets=list(args.budgets),
		seed=args.seed,
		train_before_query=args.train_before_query,
		use_cuda=args.use_cuda,
		max_workers=args.max_workers,
		embeddings_npz_path=args.embeddings_npz,
		typicality_npz_path=args.typicality_npz,
		suppress_k=args.suppress_k,
		decay_factor=args.decay_factor,
		prop_k=args.prop_k,
	)

	payload: dict[str, Any] = {
		"dataset": "CIFAR10",
		"budgets": _sorted_unique_ints([int(b) for b in args.budgets]),
		"methods": list(args.methods),
		"results": results,
	}
	output_path = Path(args.output)
	output_path.write_text(json.dumps(payload), encoding="utf-8")
	print(f"Saved: {output_path}")


if __name__ == "__main__":
	main()
