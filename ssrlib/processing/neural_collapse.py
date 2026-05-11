"""Neural Collapse metrics (Papyan, Han, Donoho 2020).

Reference:
    Papyan, V., Han, X.Y., Donoho, D.L. (2020). Prevalence of Neural Collapse
    during the terminal phase of deep learning training. PNAS.

Computes four interconnected phenomena observed in classifiers during the
"terminal phase of training" (TPT):

    NC1: variability collapse           — ΣW shrinks relative to ΣB
    NC2a: equinorm class means          — ‖μc − μG‖ becomes constant in c
    NC2b: equiangular class means       — pairwise cosines become constant
    NC2c: maximally equiangular         — cosines approach −1/(C−1) (Simplex ETF)
    NC3:  self-duality                  — classifier W ∝ class means M
    NC4:  Nearest-Class-Center (NCC)    — linear classifier == NCC decision rule

NC1–NC2 require only embeddings + labels.
NC3–NC4 additionally require classifier weights (and optionally bias).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._spectral import EPS
from .base import BaseProcessor


def _class_means(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (class_means [C, D], global_mean [D], classes [C])."""
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    global_mean = X.mean(axis=0)
    return means, global_mean, classes


def _within_class_covariance(
    X: np.ndarray, y: np.ndarray, class_means: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """ΣW = Ave_{i,c}[(h_i,c − μc)(h_i,c − μc)ᵀ]."""
    D = X.shape[1]
    sigma_w = np.zeros((D, D), dtype=np.float64)
    n_total = 0
    for ci, c in enumerate(classes):
        mask = y == c
        if not mask.any():
            continue
        diff = X[mask] - class_means[ci]
        sigma_w += diff.T @ diff
        n_total += int(mask.sum())
    return sigma_w / max(n_total, 1)


def _between_class_covariance(centered_means: np.ndarray) -> np.ndarray:
    """ΣB = (1/C) · M_centeredᵀ M_centered, with M_centered shape (C, D)."""
    C = centered_means.shape[0]
    return (centered_means.T @ centered_means) / max(C, 1)


class NeuralCollapseProcessor(BaseProcessor):
    """Neural Collapse metrics (NC1–NC4) from Papyan, Han, Donoho 2020.

    Output is a 1-D array. Without classifier weights its shape is (4,) with
    components ordered ``[nc1, nc2_equinorm, nc2_equiangle, nc2_max_equiangle]``.
    With ``classifier_weights`` provided the array has shape (6,) with two
    extra components appended: ``[nc3_selfdual, nc4_ncc_mismatch]``.

    The processor must be called with ``labels`` as a keyword argument:

        >>> proc = NeuralCollapseProcessor()
        >>> out = proc.process(X, labels=y)
        >>> out_full = proc.process(X, labels=y,
        ...                         classifier_weights=W, classifier_bias=b)

    ``EmbeddingProbe`` automatically forwards any extra context kwargs the
    processor's ``process`` signature accepts (see ``EmbeddingProbe.__call__``).
    """

    _COMPONENTS_NO_CLF = [
        "nc1",
        "nc2_equinorm",
        "nc2_equiangle",
        "nc2_max_equiangle",
    ]
    _COMPONENTS_WITH_CLF = _COMPONENTS_NO_CLF + ["nc3_selfdual", "nc4_ncc_mismatch"]

    def __init__(self, **kwargs):
        super().__init__("NeuralCollapse", **kwargs)
        self._metadata.update(
            {
                "processor_type": "neural_collapse",
                "metric": "nc1_nc2_nc3_nc4",
                "output_type": "nc_components",
                "reference": "Papyan, Han, Donoho 2020 (PNAS)",
                "components_order": list(self._COMPONENTS_NO_CLF),
            }
        )

    def process(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        classifier_weights: Optional[np.ndarray] = None,
        classifier_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if labels is None:
            raise ValueError(
                "NeuralCollapseProcessor requires labels. Pass labels= as a kwarg "
                "(or via EmbeddingProbe with labels=y in the probe call)."
            )
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"embeddings has {X.shape[0]} rows but labels has {y.shape[0]}"
            )

        means, global_mean, classes = _class_means(X, y)
        C = int(len(classes))
        if C < 2:
            raise ValueError(f"Need at least 2 classes for NC metrics, got {C}")

        centered_means = means - global_mean

        # ---- NC1: variability collapse ------------------------------------
        sigma_w = _within_class_covariance(X, y, means, classes)
        sigma_b = _between_class_covariance(centered_means)
        sigma_b_pinv = np.linalg.pinv(sigma_b)
        nc1 = float(np.trace(sigma_w @ sigma_b_pinv) / C)

        # ---- NC2a: equinorm coefficient of variation ----------------------
        norms = np.linalg.norm(centered_means, axis=1)
        avg_norm = float(norms.mean())
        nc2_equinorm = float(norms.std() / max(avg_norm, EPS))

        # Normalize centered means for cosine computations
        norms_safe = np.maximum(norms, EPS).reshape(-1, 1)
        M_unit = centered_means / norms_safe
        cos_mat = M_unit @ M_unit.T

        # Take strict upper triangle: pairwise cosines (excluding self)
        iu = np.triu_indices(C, k=1)
        cosines = cos_mat[iu]

        # ---- NC2b: equiangular ↦ std of pairwise cosines ------------------
        nc2_equiangle = float(cosines.std())

        # ---- NC2c: max-equiangular ↦ deviation from −1/(C−1) --------------
        target = -1.0 / (C - 1)
        nc2_max_equiangle = float(np.mean(np.abs(cosines - target)))

        components = [nc1, nc2_equinorm, nc2_equiangle, nc2_max_equiangle]
        component_names = list(self._COMPONENTS_NO_CLF)

        # ---- NC3 + NC4 (only if classifier weights provided) --------------
        nc3 = None
        nc4 = None
        if classifier_weights is not None:
            W = np.asarray(classifier_weights, dtype=np.float64)
            if W.ndim != 2 or W.shape != (C, X.shape[1]):
                raise ValueError(
                    f"classifier_weights must have shape (C, D) = "
                    f"({C}, {X.shape[1]}), got {W.shape}"
                )

            # NC3: ‖Wᵀ/‖Wᵀ‖_F − Ṁ/‖Ṁ‖_F‖_F²
            M_dot = centered_means.T  # (D, C)
            W_T = W.T  # (D, C)
            W_T_norm = W_T / max(float(np.linalg.norm(W_T)), EPS)
            M_dot_norm = M_dot / max(float(np.linalg.norm(M_dot)), EPS)
            nc3 = float(np.linalg.norm(W_T_norm - M_dot_norm) ** 2)

            # NC4: fraction of disagreement between linear classifier and NCC.
            # Use uncentered class means for NCC, as in the paper.
            b = (
                np.zeros(C, dtype=np.float64)
                if classifier_bias is None
                else np.asarray(classifier_bias, dtype=np.float64)
            )
            if b.shape != (C,):
                raise ValueError(
                    f"classifier_bias must have shape ({C},), got {b.shape}"
                )

            pred_lin = np.argmax(X @ W.T + b, axis=1)
            # ‖x − μ‖² = ‖x‖² − 2 x μᵀ + ‖μ‖²; the ‖x‖² term is constant in μ
            mean_sq_norms = (means * means).sum(axis=1)
            pred_ncc = np.argmin(-2 * X @ means.T + mean_sq_norms, axis=1)
            nc4 = float((pred_lin != pred_ncc).mean())

            components.extend([nc3, nc4])
            component_names = list(self._COMPONENTS_WITH_CLF)

        # Update metadata with both per-component values and the order array
        self._metadata.update(
            {
                "input_shape": X.shape,
                "n_classes": C,
                "n_features": int(X.shape[1]),
                "components_order": component_names,
                "nc1": nc1,
                "nc2_equinorm": nc2_equinorm,
                "nc2_equiangle": nc2_equiangle,
                "nc2_max_equiangle": nc2_max_equiangle,
                "has_classifier": classifier_weights is not None,
            }
        )
        if nc3 is not None:
            self._metadata.update({"nc3_selfdual": nc3, "nc4_ncc_mismatch": nc4})

        return np.array(components, dtype=np.float64)
