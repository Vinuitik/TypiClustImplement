from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import normalize_indices, set_seed, validate_index_inputs


def _load_embedding_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        if "embeddings" not in data or "labels" not in data:
            raise ValueError("embedding npz must contain keys 'embeddings' and 'labels'")
        x = np.asarray(data["embeddings"])
        y = np.asarray(data["labels"])
    if x.shape[0] != y.shape[0]:
        raise ValueError("embeddings and labels size mismatch")
    return x, y


def train_eval_fsl_ssl_embeddings(
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    *,
    train_embeddings_npz: str,
    test_embeddings_npz: str,
    val_ratio: float = 0.2,
    seed: int = 0,
    c: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, float]:
    """Method 2: Logistic regression on frozen SSL embeddings using labeled indices only."""
    set_seed(seed)
    x_train, y_train = _load_embedding_npz(train_embeddings_npz)
    x_test, y_test = _load_embedding_npz(test_embeddings_npz)
    validate_index_inputs(labeled_indices, unlabeled_indices, x_train.shape[0])

    lb = np.asarray(normalize_indices(labeled_indices), dtype=np.int64)
    x_lb = x_train[lb]
    y_lb = y_train[lb]

    if len(lb) >= 2 and val_ratio > 0:
        try:
            x_fit, x_val, y_fit, y_val = train_test_split(
                x_lb,
                y_lb,
                test_size=val_ratio,
                random_state=seed,
                stratify=y_lb if len(np.unique(y_lb)) > 1 else None,
            )
        except ValueError:
            x_fit, x_val, y_fit, y_val = train_test_split(x_lb, y_lb, test_size=val_ratio, random_state=seed)
    else:
        x_fit, y_fit = x_lb, y_lb
        x_val, y_val = x_lb, y_lb

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    multi_class="auto",
                    solver="lbfgs",
                    n_jobs=None,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(x_fit, y_fit)
    val_acc = accuracy_score(y_val, clf.predict(x_val))
    test_acc = accuracy_score(y_test, clf.predict(x_test))

    return {
        "method": "fsl_ssl_embeddings",
        "test_acc": float(test_acc),
        "best_val_acc": float(val_acc),
        "epochs_run": 1.0,
        "num_labeled": float(len(labeled_indices)),
        "num_unlabeled": float(len(unlabeled_indices)),
    }
