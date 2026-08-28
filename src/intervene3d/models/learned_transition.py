r"""A deliberately small learned transition model.

The research plan is explicit: *do not start with a giant transformer*.  This is
a two-hidden-layer MLP trained with Adam in pure NumPy (no deep-learning
dependency, fully deterministic under a seed, seconds to train on CPU).  It
predicts a **per-landmark residual**

.. math:: \hat F_{t+1} = \Phi_{\text{base}}(F_t, H_k, a) + g_\theta(F_t, H_k, a)

where ``Phi_base`` is either the analytical optical transition (the *hybrid*
model, ``base="analytical"``) or the identity (the *learned-only* ablation,
``base="identity"``).  The hybrid/learned-only contrast is one of the ablations
the research specification asks for, and this is the smallest implementation
that makes the contrast real rather than rhetorical.

Scope: this is a Gate-7 placeholder.  It operates on the landmark feature space,
not on a geometry-foundation-model latent, and is not claimed to be a world
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.interventions.actions import Action

_MECHANISM_ORDER = list(OpticalMechanism)
_N_MECH = len(_MECHANISM_ORDER)


def _mechanism_onehot(mechanism: OpticalMechanism) -> np.ndarray:
    v = np.zeros(_N_MECH)
    v[_MECHANISM_ORDER.index(mechanism)] = 1.0
    return v


def build_inputs(
    state: GeometryFeature,
    hypothesis: Hypothesis,
    action: Action,
    *,
    hypothesis_conditioning: bool = True,
) -> np.ndarray:
    """Per-landmark input matrix ``(N, D)`` for the content channel.

    Features are normalised so the MLP sees O(1) magnitudes: pixels by image
    size, depth by 10 m, translations by 1 m, rotations by 1 rad.
    """
    mask = state.channel == CHANNEL_CONTENT
    intr = state.camera.intrinsics
    uv = np.nan_to_num(state.uv[mask], nan=0.0)
    depth = np.nan_to_num(state.depth[mask], nan=0.0)
    vis = state.visible[mask].astype(np.float64)

    u = uv[:, 0] / max(intr.width, 1)
    v = uv[:, 1] / max(intr.height, 1)
    z = depth / 10.0
    inv_z = np.where(depth > 1e-6, 1.0 / np.maximum(depth, 1e-6), 0.0)

    a = np.concatenate([action.translation, action.rotation_euler])
    onehot = _mechanism_onehot(hypothesis.mechanism) if hypothesis_conditioning else np.zeros(_N_MECH)
    tail = np.concatenate([a, onehot])

    per_landmark = np.stack([u, v, z, inv_z, vis], axis=1)
    return np.concatenate([per_landmark, np.tile(tail, (per_landmark.shape[0], 1))], axis=1)


INPUT_DIM = 5 + 6 + _N_MECH
OUTPUT_DIM = 3  # (du, dv, d_depth)


@dataclass
class ResidualMLP:
    """Two-hidden-layer tanh MLP with Adam, implemented in NumPy."""

    hidden_dim: int = 32
    seed: int = 0
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    params: dict[str, np.ndarray] = field(default_factory=dict)
    _adam: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)
    _step: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.params:
            rng = np.random.default_rng(self.seed)
            h = self.hidden_dim
            self.params = {
                "W1": rng.normal(0.0, np.sqrt(2.0 / INPUT_DIM), (INPUT_DIM, h)),
                "b1": np.zeros(h),
                "W2": rng.normal(0.0, np.sqrt(2.0 / h), (h, h)),
                "b2": np.zeros(h),
                "W3": np.zeros((h, OUTPUT_DIM)),  # start as the identity residual (= base model)
                "b3": np.zeros(OUTPUT_DIM),
            }

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.tanh(z2)
        y = a2 @ p["W3"] + p["b3"]
        return y, {"x": x, "a1": a1, "a2": a2}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    # ----------------------------------------------------------------- training
    def _backward(self, cache: dict[str, np.ndarray], dy: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        x, a1, a2 = cache["x"], cache["a1"], cache["a2"]
        n = max(x.shape[0], 1)
        grads = {"W3": a2.T @ dy / n, "b3": dy.mean(axis=0)}
        da2 = dy @ p["W3"].T
        dz2 = da2 * (1.0 - a2**2)
        grads["W2"] = a1.T @ dz2 / n
        grads["b2"] = dz2.mean(axis=0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (1.0 - a1**2)
        grads["W1"] = x.T @ dz1 / n
        grads["b1"] = dz1.mean(axis=0)
        return grads

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for key, g in grads.items():
            g = g + self.weight_decay * self.params[key]
            m, v = self._adam.get(key, (np.zeros_like(g), np.zeros_like(g)))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            self._adam[key] = (m, v)
            m_hat = m / (1 - b1**self._step)
            v_hat = v / (1 - b2**self._step)
            self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 200,
        batch_size: int = 256,
        seed: int = 0,
        verbose_every: int = 0,
    ) -> list[float]:
        """Minimise the mean-squared residual error.  Returns the loss history."""
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        history: list[float] = []
        if n == 0:
            return history
        for epoch in range(epochs):
            perm = rng.permutation(n)
            total, count = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = x[idx], y[idx]
                pred, cache = self.forward(xb)
                diff = pred - yb
                loss = float(np.mean(diff**2))
                self._adam_step(self._backward(cache, 2.0 * diff / max(diff.shape[1], 1)))
                total += loss * xb.shape[0]
                count += xb.shape[0]
            history.append(total / max(count, 1))
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  epoch {epoch + 1:4d}  loss {history[-1]:.6f}")
        return history

    # ------------------------------------------------------------ serialisation
    def state_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "params": {k: v.tolist() for k, v in self.params.items()},
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> ResidualMLP:
        model = cls(
            hidden_dim=int(payload["hidden_dim"]),
            seed=int(payload["seed"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            params={k: np.asarray(v, dtype=np.float64) for k, v in payload["params"].items()},
        )
        return model

    def save(self, path) -> None:
        from intervene3d.utils.io import dump_json

        dump_json(path, self.state_dict())

    @classmethod
    def load(cls, path) -> ResidualMLP:
        from intervene3d.utils.io import load_json

        return cls.from_state_dict(load_json(path))
