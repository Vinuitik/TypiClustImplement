import numpy as np
import faiss


def _load_embeddings(path: str) -> np.ndarray:
    if not path:
        raise ValueError("embeddings_npz_path is required for typiclust")
    data = np.load(path)
    return data[data.files[0]]


def _knn_search(embeddings_f32: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch KNN via FAISS (GPU when available).
    Returns (sq_distances, indices) each shaped (n, k), with self excluded.
    """
    n, d = embeddings_f32.shape
    actual_k = min(k, n - 1)

    cpu_index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index

    index.add(embeddings_f32)
    sq_dists, idxs = index.search(embeddings_f32, actual_k + 1)  # +1 to drop self
    return sq_dists[:, 1:], idxs[:, 1:]   # drop column 0 (self, dist=0)


def _faiss_kmeans(embeddings_f32: np.ndarray, k: int, seed: int) -> np.ndarray:
    """KMeans via FAISS (GPU when available). Returns cluster label per point."""
    n, d = embeddings_f32.shape
    use_gpu = faiss.get_num_gpus() > 0
    km = faiss.Kmeans(d, k, niter=20, nredo=3, gpu=use_gpu, seed=seed, verbose=False)
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
    Selects points using the TypiClust algorithm (FAISS GPU-accelerated).
    """
    unlabeled_data, _ = dataset.get_unlabeled_data()
    unlabeled_idxs = [int(v) for v in unlabeled_data]
    embeddings = _load_embeddings(embeddings_npz_path)

    if budget >= len(unlabeled_idxs):
        return unlabeled_idxs, []

    emb = np.ascontiguousarray(embeddings[unlabeled_idxs], dtype=np.float32)
    n_samples = emb.shape[0]

    # -- Batch KNN → typicality for every point --------------------------
    sq_dists, _ = _knn_search(emb, k)
    mean_dists = np.sqrt(np.maximum(sq_dists, 0.0)).mean(axis=1)  # (n,)
    typicality = 1.0 / (mean_dists + 1e-9)                         # (n,)

    # -- GPU KMeans -------------------------------------------------------
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
