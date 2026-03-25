import numpy as np
from sklearn.cluster import KMeans

def _load_embeddings(path: str) -> np.ndarray:
    if not path:
        raise ValueError("embeddings_npz_path is required for typiclust")
    data = np.load(path)
    return data[data.files[0]]

def typiclust(
    dataset,
    budget: int,
    embeddings_npz_path: str = None,
    k: int = 20,
    random_state: int | None = 42,
) -> tuple[list[int], list[int]]:
    """
    Selects points using the TypiClust algorithm.
    """
    unlabeled_data, _ = dataset.get_unlabeled_data()
    unlabeled_idxs = [int(v) for v in unlabeled_data]
    embeddings = _load_embeddings(embeddings_npz_path)

    if budget >= len(unlabeled_idxs):
        return unlabeled_idxs, []

    unlabeled_embeddings = embeddings[unlabeled_idxs]
    n_samples = unlabeled_embeddings.shape[0]

    # Cluster the embeddings into `budget` clusters
    kmeans = KMeans(n_clusters=budget, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(unlabeled_embeddings)

    labeled_indexes = []

    # For each cluster, find the point with the highest typicality
    for c in range(budget):
        cluster_indices = np.where(cluster_labels == c)[0]
        if len(cluster_indices) == 0:
            continue
            
        best_idx = -1
        highest_typicality = -1.0

        for idx in cluster_indices:
            point = unlabeled_embeddings[idx]
            # Compute Euclidean distances to all other points in unlabeled pool
            distances = np.linalg.norm(unlabeled_embeddings - point, axis=1)
            
            # Find K nearest neighbors (using partition for efficiency)
            actual_k = min(k, n_samples)
            smallest_distances = np.partition(distances, actual_k - 1)[:actual_k]
            
            mean_dist = np.mean(smallest_distances)
            typicality = 1.0 / (mean_dist + 1e-9)

            if typicality > highest_typicality:
                highest_typicality = typicality
                best_idx = idx
                
        if best_idx != -1:
            labeled_indexes.append(unlabeled_idxs[int(best_idx)])

    unlabeled_indexes = [idx for idx in unlabeled_idxs if idx not in labeled_indexes]

    return labeled_indexes, unlabeled_indexes
