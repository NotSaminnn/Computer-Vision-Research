"""Camera-intervention primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from intervene3d.geometry.se3 import euler_to_rotation, rotation_angle, se3_from_Rt


class ActionKind(str, Enum):
    """Canonical single-degree-of-freedom intervention families."""

    NONE = "none"
    TRANSLATE_X = "translate_x"  # lateral (the classical stereo baseline)
    TRANSLATE_Y = "translate_y"  # vertical
    TRANSLATE_Z = "translate_z"  # forward / backward
    YAW = "yaw"
    PITCH = "pitch"
    ROLL = "roll"
    COMPOUND = "compound"


@dataclass(frozen=True)
class Action:
    """A single controlled camera intervention ``do(C_{t+1} = C_t . delta)``.

    ``translation`` is in metres in the reference camera frame
    (``x`` right, ``y`` down, ``z`` forward);
    ``rotation_euler`` is ``(yaw, pitch, roll)`` in radians.
    """

    name: str
    kind: ActionKind = ActionKind.COMPOUND
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        t = np.asarray(self.translation, dtype=np.float64).reshape(3)
        r = np.asarray(self.rotation_euler, dtype=np.float64).reshape(3)
        object.__setattr__(self, "translation", t)
        object.__setattr__(self, "rotation_euler", r)
        object.__setattr__(self, "kind", ActionKind(self.kind))

    def delta_T(self) -> np.ndarray:
        """The SE(3) delta, expressed in the reference camera frame."""
        R = euler_to_rotation(*self.rotation_euler)
        return se3_from_Rt(R, self.translation)

    @property
    def translation_magnitude(self) -> float:
        """``|t|`` in metres -- the *baseline* of this intervention."""
        return float(np.linalg.norm(self.translation))

    @property
    def rotation_magnitude(self) -> float:
        """Geodesic rotation magnitude in radians."""
        return rotation_angle(euler_to_rotation(*self.rotation_euler))

    @property
    def is_null(self) -> bool:
        return self.translation_magnitude < 1e-12 and self.rotation_magnitude < 1e-12

    def scaled(self, factor: float, *, name: str | None = None) -> Action:
        """A proportionally scaled copy (used to sweep separability vs baseline)."""
        return Action(
            name if name is not None else f"{self.name}*{factor:g}",
            self.kind,
            self.translation * factor,
            self.rotation_euler * factor,
        )

    def perturbed(self, rng: np.random.Generator, *, translation_std: float, rotation_std: float) -> Action:
        """Additive execution noise ``Delta C + epsilon`` (generalisation study)."""
        return Action(
            f"{self.name}+noise",
            self.kind,
            self.translation + rng.normal(0.0, translation_std, size=3),
            self.rotation_euler + rng.normal(0.0, rotation_std, size=3),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "translation": self.translation.tolist(),
            "rotation_euler": self.rotation_euler.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Action:
        return cls(
            payload["name"],
            ActionKind(payload.get("kind", "compound")),
            np.asarray(payload.get("translation", np.zeros(3)), dtype=np.float64),
            np.asarray(payload.get("rotation_euler", np.zeros(3)), dtype=np.float64),
        )

    def __repr__(self) -> str:
        return (
            f"Action({self.name}, |t|={self.translation_magnitude:.3f} m, "
            f"|r|={np.degrees(self.rotation_magnitude):.2f} deg)"
        )


def null_action(name: str = "stay") -> Action:
    """The do-nothing intervention; the passive-observation control."""
    return Action(name, ActionKind.NONE, np.zeros(3), np.zeros(3))
