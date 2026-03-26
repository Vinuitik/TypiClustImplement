import numpy as np
import torch
import faiss


def _load_embeddings(path: str) -> np.ndarray:
    if not path:
        raise ValueError("embeddings_npz_path is required for typiclust")
    data = np.load(path)
    return data[data.files[0]]


def _knn_search(embeddings: np.ndarray, k: int, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched KNN using PyTorch (GPU when available).
    Returns (distances, indices) each shaped (n, k), self excluded.
    Distances are actual L2 (not squared).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = embeddings.shape[0]
    actual_k = min(k, n - 1)

    emb_t = torch.from_numpy(embeddings).to(device)         # (n, d)

    all_dists = np.empty((n, actual_k), dtype=np.float32)
    all_idxs  = np.empty((n, actual_k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = emb_t[start:end]                             # (b, d)

        # torch.cdist → exact L2 distances, shape (b, n)
        dists = torch.cdist(batch, emb_t)

        # k+1 so we can drop self (self-distance ≈ 0, always first)
        topk_dists, topk_idxs = dists.topk(actual_k + 1, dim=1, largest=False)

        all_dists[start:end] = topk_dists[:, 1:].cpu().numpy()
        all_idxs[start:end]  = topk_idxs[:, 1:].cpu().numpy()

    return all_dists, all_idxs


def _faiss_kmeans(embeddings_f32: np.ndarray, k: int, seed: int) -> np.ndarray:
    """KMeans via FAISS CPU. Returns cluster label per point."""
    _, d = embeddings_f32.shape
    km = faiss.Kmeans(d, k, niter=20, nredo=3, gpu=False, seed=seed, verbose=False)
    km.train(embeddings_f32)
    _, labels = km.index.search(embeddings_f32, 1)
    return labels.flatten()


def typiclust(
    dataset,
    budget: int,
    embeddings_npz_path: str = None,
    k: int = 20,
    random_state: int | None = 42,
) -> tuple[list[int], list[int]]:
    """
    Selects points using the TypiClust algorithm.
    KNN is PyTorch GPU-accelerated; KMeans uses FAISS CPU.
    """
    unlabeled_data, _ = dataset.get_unlabeled_data()
    unlabeled_idxs = [int(v) for v in unlabeled_data]
    embeddings = _load_embeddings(embeddings_npz_path)

    if budget >= len(unlabeled_idxs):
        return unlabeled_idxs, []

    emb = np.ascontiguousarray(embeddings[unlabeled_idxs], dtype=np.float32)

    # -- Batch KNN (PyTorch GPU) → typicality for every point ------------
    dists, _ = _knn_search(emb, k)
    mean_dists = dists.mean(axis=1)                  # (n,)  already L2
    typicality = 1.0 / (mean_dists + 1e-9)           # (n,)

    # -- KMeans (FAISS CPU) -----------------------------------------------
    seed = random_state if random_state is not None else 0
    cluster_labels = _faiss_kmeans(emb, budget, seed)

    # -- Per cluster: pick highest-typicality point -----------------------
    labeled_indexes = []
    for c in range(budget):
        cluster_indices = np.where(cluster_labels == c)[0]
        if len(cluster_indices) == 0:
            continue
        best_idx = cluster_indices[np.argmax(typicality[cluster_indices])]
        labeled_indexes.append(unlabeled_idxs[int(best_idx)])

    unlabeled_indexes = [idx for idx in unlabeled_idxs if idx not in labeled_indexes]
    return labeled_indexes, unlabeled_indexes
