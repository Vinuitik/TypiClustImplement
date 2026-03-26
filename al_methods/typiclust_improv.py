import numpy as np
from .typiclust import _load_embeddings


def typiclust_improv(
    dataset,
    budget: int,
    embeddings_npz_path: str = None,
    k: int = 20,
    random_state: int | None = 42,
) -> tuple[list[int], list[int]]:
    """
    TypiClust variant with diversity-penalised greedy selection.

    Steps:
      1. Precompute typicality = 1 / mean_knn_dist for every unlabeled point.
      2. Maintain running sum and sum-of-squares for O(1) std computation.
      3. Greedily pick the most typical point, then:
         a. Zero its typicality (prevents re-selection).
         b. Subtract 0.5 * current_std from each of its k nearest neighbors.
      4. Repeat until `budget` points are selected.
    """
    unlabeled_data, _ = dataset.get_unlabeled_data()
    unlabeled_idxs = [int(v) for v in unlabeled_data]
    embeddings = _load_embeddings(embeddings_npz_path)

    if budget >= len(unlabeled_idxs):
        return unlabeled_idxs, []

    unlabeled_embeddings = embeddings[unlabeled_idxs]
    n_samples = unlabeled_embeddings.shape[0]
    actual_k = min(k, n_samples - 1)

    # ------------------------------------------------------------------
    # Precompute typicality and KNN indices for every point
    # ------------------------------------------------------------------
    typicality = np.zeros(n_samples)
    knn_indices = np.empty((n_samples, actual_k), dtype=np.intp)

    for i in range(n_samples):
        dists = np.linalg.norm(unlabeled_embeddings - unlabeled_embeddings[i], axis=1)
        dists[i] = np.inf  # exclude self
        nn = np.argpartition(dists, actual_k - 1)[:actual_k]
        knn_indices[i] = nn
        typicality[i] = 1.0 / (np.mean(dists[nn]) + 1e-9)

    # ------------------------------------------------------------------
    # Running stats for O(1) std:  var = E[T²] - E[T]²
    # ------------------------------------------------------------------
    sum_T = float(np.sum(typicality))
    sum_T2 = float(np.dot(typicality, typicality))

    def current_std() -> float:
        mean = sum_T / n_samples
        var = sum_T2 / n_samples - mean * mean
        return float(np.sqrt(max(var, 0.0)))

    # ------------------------------------------------------------------
    # Greedy selection with neighbor penalisation
    # ------------------------------------------------------------------
    labeled_indexes: list[int] = []

    for _ in range(budget):
        best = int(np.argmax(typicality))
        labeled_indexes.append(unlabeled_idxs[best])

        # Zero out selected point and update running stats
        t_sel = typicality[best]
        sum_T -= t_sel
        sum_T2 -= t_sel * t_sel
        typicality[best] = 0.0

        # Penalise neighbors
        penalty = 0.5 * current_std()
        for nb in knn_indices[best]:
            t_old = typicality[nb]
            t_new = max(t_old - penalty, 0.0)
            sum_T += t_new - t_old
            sum_T2 += t_new * t_new - t_old * t_old
            typicality[nb] = t_new

    unlabeled_indexes = [idx for idx in unlabeled_idxs if idx not in labeled_indexes]
    return labeled_indexes, unlabeled_indexes
