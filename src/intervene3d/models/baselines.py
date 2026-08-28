r"""Discriminative baselines: the controls that test whether the problem exists.

The single most important baseline in the whole project is the **static
classifier**.  If a model could name the physical mechanism from the reference
image alone, the interventional story would collapse.  In this benchmark the
matched counterfactuals are *pixel-identical* at the reference view by
construction, so any function of the reference view must be at chance -- and
this module demonstrates that empirically rather than merely asserting it.

Three families:

``SingleFrameClassifier``
    Trained on reference-view features only.  Expected: chance.
``PassiveMultiViewClassifier``
    Trained on reference-view features **plus** the response to one *fixed*,
    unchosen action.  It can see parallax, so it can separate a display from a
    real scene, but it cannot choose the action that would reveal a mirror.
``MultinomialLogisticRegression``
    The shared trainable core -- softmax regression with L2, full-batch Adam,
    pure NumPy, deterministic under a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT, CHANNEL_FRAME, CHANNEL_MARKER, GeometryFeature
from intervene3d.geometry.planes import homography_residual


# --------------------------------------------------------------------- features
def reference_features(feature: GeometryFeature) -> np.ndarray:
    """Summary statistics of a single view.

    Everything here is a function of the reference observation alone.  Because
    the benchmark's variants are pixel-identical at the reference view, this
    vector is *provably* identical across the causal variants of a base scene.
    """
    intr = feature.camera.intrinsics
    content = (feature.channel == CHANNEL_CONTENT) & feature.visible
    frame = (feature.channel == CHANNEL_FRAME) & feature.visible
    markers = (feature.channel == CHANNEL_MARKER) & feature.visible

    uv = feature.uv[content]
    depth = feature.depth[content]
    n_c = max(uv.shape[0], 1)

    def _s(x: np.ndarray, default: float = 0.0) -> float:
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else default

    return np.array(
        [
            uv.shape[0] / max(np.count_nonzero(feature.channel == CHANNEL_CONTENT), 1),
            np.count_nonzero(frame) / 4.0,
            float(np.count_nonzero(markers)),
            _s(uv[:, 0]) / intr.width if uv.size else 0.0,
            _s(uv[:, 1]) / intr.height if uv.size else 0.0,
            float(np.std(uv[:, 0])) / intr.width if uv.shape[0] > 1 else 0.0,
            float(np.std(uv[:, 1])) / intr.height if uv.shape[0] > 1 else 0.0,
            _s(depth) / 10.0,
            float(np.std(depth[np.isfinite(depth)])) / 10.0 if n_c > 1 else 0.0,
            _s(1.0 / np.clip(depth, 1e-3, None)),
            float(np.nanmin(depth)) / 10.0 if np.any(np.isfinite(depth)) else 0.0,
            float(np.nanmax(depth)) / 10.0 if np.any(np.isfinite(depth)) else 0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def response_features(reference: GeometryFeature, observed: GeometryFeature) -> np.ndarray:
    """Statistics of how the view *changed* after one action.

    This is exactly the information a passive multi-view model has: it observes
    the consequence of a motion it did not choose.
    """
    content = (reference.channel == CHANNEL_CONTENT)
    joint = content & reference.visible & observed.visible
    d = observed.uv[joint] - reference.uv[joint]
    mag = np.linalg.norm(d, axis=1) if d.size else np.zeros(1)

    vis_change = float(np.count_nonzero(reference.visible != observed.visible))
    marker_appear = float(
        np.count_nonzero((observed.channel == CHANNEL_MARKER) & observed.visible & ~reference.visible)
    )
    residual = homography_residual(reference.uv[joint], observed.uv[joint]) if int(np.count_nonzero(joint)) >= 4 else 0.0
    if not np.isfinite(residual):
        residual = 0.0

    return np.array(
        [
            float(np.mean(mag)),
            float(np.std(mag)) if mag.size > 1 else 0.0,
            float(np.max(mag)),
            float(np.ptp(mag)) if mag.size > 1 else 0.0,
            residual,
            np.log1p(residual),
            vis_change,
            marker_appear,
            float(np.count_nonzero(joint)),
            1.0,
        ],
        dtype=np.float64,
    )


# ------------------------------------------------------------------- classifier
@dataclass
class MultinomialLogisticRegression:
    """Softmax regression with L2, full-batch Adam.  Deterministic under ``seed``."""

    n_features: int
    n_classes: int
    learning_rate: float = 0.1
    l2: float = 1e-3
    epochs: int = 400
    seed: int = 0
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _scale: np.ndarray = field(default_factory=lambda: np.ones(0), repr=False)

    def __post_init__(self) -> None:
        if self.W.size == 0:
            self.W = np.zeros((self.n_features, self.n_classes))
            self.b = np.zeros(self.n_classes)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean) / self._scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._mean = x.mean(axis=0)
        self._scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        xs = self._standardise(x)
        onehot = np.zeros((x.shape[0], self.n_classes))
        onehot[np.arange(x.shape[0]), y] = 1.0

        m_w = np.zeros_like(self.W)
        v_w = np.zeros_like(self.W)
        m_b = np.zeros_like(self.b)
        v_b = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        history: list[float] = []
        for step in range(1, self.epochs + 1):
            probs = self._softmax(xs @ self.W + self.b)
            diff = probs - onehot
            gW = xs.T @ diff / max(x.shape[0], 1) + self.l2 * self.W
            gb = diff.mean(axis=0)
            m_w = b1 * m_w + (1 - b1) * gW
            v_w = b2 * v_w + (1 - b2) * gW**2
            m_b = b1 * m_b + (1 - b1) * gb
            v_b = b2 * v_b + (1 - b2) * gb**2
            self.W -= self.learning_rate * (m_w / (1 - b1**step)) / (np.sqrt(v_w / (1 - b2**step)) + eps)
            self.b -= self.learning_rate * (m_b / (1 - b1**step)) / (np.sqrt(v_b / (1 - b2**step)) + eps)
            history.append(float(-np.mean(np.log(np.clip(probs[np.arange(x.shape[0]), y], 1e-12, None)))))
        return history

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-300, None)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._mean.size == 0:  # untrained -> uniform, never a silent guess
            return np.full((np.asarray(x).shape[0], self.n_classes), 1.0 / self.n_classes)
        return self._softmax(self._standardise(np.asarray(x, dtype=np.float64)) @ self.W + self.b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features, "n_classes": self.n_classes,
            "learning_rate": self.learning_rate, "l2": self.l2, "epochs": self.epochs, "seed": self.seed,
            "W": self.W.tolist(), "b": self.b.tolist(),
            "mean": self._mean.tolist(), "scale": self._scale.tolist(),
        }


@dataclass
class DiscriminativeBaseline:
    """A trained classifier over a chosen feature space."""

    name: str
    labels: Sequence[str]
    model: MultinomialLogisticRegression
    uses_response: bool
    fixed_action: str | None = None
    train_loss: list[float] = field(default_factory=list)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "labels": list(self.labels),
            "uses_response": self.uses_response,
            "fixed_action": self.fixed_action,
            "final_train_loss": self.train_loss[-1] if self.train_loss else None,
            "n_features": self.model.n_features,
        }


def train_baseline(
    name: str,
    features: np.ndarray,
    labels_idx: np.ndarray,
    label_names: Sequence[str],
    *,
    uses_response: bool,
    fixed_action: str | None = None,
    seed: int = 0,
    epochs: int = 400,
) -> DiscriminativeBaseline:
    model = MultinomialLogisticRegression(
        n_features=features.shape[1], n_classes=len(label_names), epochs=epochs, seed=seed
    )
    history = model.fit(features, labels_idx)
    return DiscriminativeBaseline(name, list(label_names), model, uses_response, fixed_action, history)
