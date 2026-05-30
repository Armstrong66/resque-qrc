"""
preprocessing/projection.py — Train-only PCA for shared model inputs.

When USE_SHARED_PCA is True, all reservoir/RNN models see the same reduced
feature space (fit on train split only).
"""

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED, QUBIT_PRIMARY, PCA_COMPONENTS, USE_SHARED_PCA
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class PCAProjector:
    """Fitted PCA mapping X (N, d_in) -> (N, n_components). Fit on train only."""
    pca: PCA
    n_components: int
    input_dim: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X).astype(np.float32)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "PCAProjector":
        with open(path, "rb") as f:
            return pickle.load(f)


def pca_n_components(n_components: int = None) -> int:
    return n_components or PCA_COMPONENTS or QUBIT_PRIMARY


def fit_pca_projector(X_train: np.ndarray,
                      n_components: int = None) -> PCAProjector:
    """Fit PCA on training windows only (no leakage)."""
    n_components = pca_n_components(n_components)
    n_components = min(n_components, X_train.shape[1], max(1, X_train.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_train)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info(
        f"PCA projector: {X_train.shape[1]}d -> {n_components}d "
        f"(explained variance {explained:.1%})"
    )
    return PCAProjector(pca=pca, n_components=n_components, input_dim=X_train.shape[1])


def project_splits(projector: PCAProjector,
                   X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        projector.transform(X_train),
        projector.transform(X_val),
        projector.transform(X_test),
    )


def maybe_project_dataset(ds, n_components: int = None):
    """
    If USE_SHARED_PCA, return (X_train, X_val, X_test, projector) projected;
    else return raw splits and None.
    """
    if not USE_SHARED_PCA:
        return ds.X_train, ds.X_val, ds.X_test, None
    proj = fit_pca_projector(ds.X_train, n_components)
    X_tr, X_vl, X_ts = project_splits(proj, ds.X_train, ds.X_val, ds.X_test)
    return X_tr, X_vl, X_ts, proj
