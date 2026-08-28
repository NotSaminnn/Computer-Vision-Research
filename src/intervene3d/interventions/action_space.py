"""The bounded, configurable action set ``A``.

Identifiability in this project is *relative to an action set*.  The action
space is therefore a first-class, serialised object: an identifiability label is
meaningless without the ``A`` it was computed against.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.interventions.actions import Action, ActionKind, null_action


@dataclass(frozen=True)
class ActionSpaceConfig:
    """Declarative description of the allowed interventions.

    Parameters
    ----------
    translation_steps:
        Signed magnitudes in metres applied along each enabled translation axis.
    rotation_steps_deg:
        Signed magnitudes in degrees applied about each enabled rotation axis.
    enabled_kinds:
        Which single-DoF families to instantiate.
    max_translation / max_rotation_deg:
        Hard bounds.  Any candidate exceeding them is rejected by
        :meth:`ActionSpace.validate`, and generation clips the step lists.
    include_null:
        Whether ``A`` contains the do-nothing action (needed as a control, and
        as the "passive" baseline's only option).
    """

    translation_steps: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    rotation_steps_deg: tuple[float, ...] = (5.0, 10.0)
    enabled_kinds: tuple[str, ...] = ("translate_x", "translate_y", "translate_z", "yaw", "pitch")
    max_translation: float = 0.35
    max_rotation_deg: float = 15.0
    include_null: bool = True
    symmetric: bool = True  # also generate the negated version of each step

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_steps": list(self.translation_steps),
            "rotation_steps_deg": list(self.rotation_steps_deg),
            "enabled_kinds": list(self.enabled_kinds),
            "max_translation": self.max_translation,
            "max_rotation_deg": self.max_rotation_deg,
            "include_null": self.include_null,
            "symmetric": self.symmetric,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ActionSpaceConfig:
        payload = dict(payload or {})
        return cls(
            translation_steps=tuple(float(x) for x in payload.get("translation_steps", (0.05, 0.10, 0.20, 0.30))),
            rotation_steps_deg=tuple(float(x) for x in payload.get("rotation_steps_deg", (5.0, 10.0))),
            enabled_kinds=tuple(payload.get("enabled_kinds", ("translate_x", "translate_y", "translate_z", "yaw", "pitch"))),
            max_translation=float(payload.get("max_translation", 0.35)),
            max_rotation_deg=float(payload.get("max_rotation_deg", 15.0)),
            include_null=bool(payload.get("include_null", True)),
            symmetric=bool(payload.get("symmetric", True)),
        )


_TRANSLATION_AXES: dict[ActionKind, np.ndarray] = {
    ActionKind.TRANSLATE_X: np.array([1.0, 0.0, 0.0]),
    ActionKind.TRANSLATE_Y: np.array([0.0, 1.0, 0.0]),
    ActionKind.TRANSLATE_Z: np.array([0.0, 0.0, 1.0]),
}
_ROTATION_AXES: dict[ActionKind, np.ndarray] = {
    ActionKind.YAW: np.array([1.0, 0.0, 0.0]),    # index into (yaw, pitch, roll)
    ActionKind.PITCH: np.array([0.0, 1.0, 0.0]),
    ActionKind.ROLL: np.array([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class ActionSpace:
    """A concrete, ordered, bounded set of candidate interventions."""

    actions: tuple[Action, ...]
    config: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("ActionSpace must contain at least one action")
        names = [a.name for a in actions]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Action names must be unique; duplicates: {dupes}")
        object.__setattr__(self, "actions", actions)
        self.validate()

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __getitem__(self, index: int) -> Action:
        return self.actions[index]

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.actions]

    def index_of(self, name: str) -> int:
        for i, a in enumerate(self.actions):
            if a.name == name:
                return i
        raise KeyError(f"no action named {name!r}")

    def validate(self) -> None:
        """Assert every action respects the configured motion bounds."""
        max_rot = np.radians(self.config.max_rotation_deg)
        for a in self.actions:
            if a.translation_magnitude > self.config.max_translation + 1e-9:
                raise ValueError(
                    f"action {a.name!r} translation {a.translation_magnitude:.4f} m exceeds "
                    f"max_translation {self.config.max_translation} m"
                )
            if a.rotation_magnitude > max_rot + 1e-9:
                raise ValueError(
                    f"action {a.name!r} rotation {np.degrees(a.rotation_magnitude):.3f} deg exceeds "
                    f"max_rotation {self.config.max_rotation_deg} deg"
                )

    def contains(self, action: Action) -> bool:
        return any(a.name == action.name for a in self.actions)

    @property
    def max_translation_action(self) -> Action:
        """Largest-baseline action -- the ``max_baseline`` intervention baseline."""
        return max(self.actions, key=lambda a: a.translation_magnitude)

    def non_null(self) -> list[Action]:
        return [a for a in self.actions if not a.is_null]

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "actions": [a.to_dict() for a in self.actions]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionSpace:
        return cls(
            tuple(Action.from_dict(a) for a in payload["actions"]),
            ActionSpaceConfig.from_dict(payload.get("config")),
        )

    @classmethod
    def from_config(cls, config: ActionSpaceConfig | dict[str, Any] | None) -> ActionSpace:
        """Instantiate the candidate action grid described by ``config``."""
        cfg = config if isinstance(config, ActionSpaceConfig) else ActionSpaceConfig.from_dict(config)
        actions: list[Action] = []
        if cfg.include_null:
            actions.append(null_action())

        signs: Sequence[float] = (1.0, -1.0) if cfg.symmetric else (1.0,)
        for kind_name in cfg.enabled_kinds:
            kind = ActionKind(kind_name)
            if kind in _TRANSLATION_AXES:
                axis = _TRANSLATION_AXES[kind]
                for step in cfg.translation_steps:
                    if abs(step) > cfg.max_translation + 1e-12:
                        continue
                    for sign in signs:
                        value = sign * step
                        actions.append(
                            Action(
                                f"{kind.value}{value:+.3f}m",
                                kind,
                                axis * value,
                                np.zeros(3),
                            )
                        )
            elif kind in _ROTATION_AXES:
                selector = _ROTATION_AXES[kind]
                for step_deg in cfg.rotation_steps_deg:
                    if abs(step_deg) > cfg.max_rotation_deg + 1e-12:
                        continue
                    for sign in signs:
                        value_deg = sign * step_deg
                        actions.append(
                            Action(
                                f"{kind.value}{value_deg:+.2f}deg",
                                kind,
                                np.zeros(3),
                                selector * np.radians(value_deg),
                            )
                        )
            else:
                raise ValueError(f"unsupported action kind {kind_name!r}")
        return cls(tuple(actions), cfg)

    @classmethod
    def lateral_sweep(
        cls,
        baselines: Sequence[float],
        *,
        max_translation: float | None = None,
    ) -> ActionSpace:
        """A pure lateral-translation sweep, used for MCRB estimation and for the
        separability-versus-baseline figure."""
        baselines = [float(b) for b in baselines]
        limit = max_translation if max_translation is not None else max(abs(b) for b in baselines)
        cfg = ActionSpaceConfig(
            translation_steps=tuple(sorted({abs(b) for b in baselines if b != 0.0})),
            rotation_steps_deg=(),
            enabled_kinds=("translate_x",),
            max_translation=limit,
            max_rotation_deg=0.0,
            include_null=any(b == 0.0 for b in baselines),
            symmetric=False,
        )
        actions = [
            Action(f"translate_x{b:+.4f}m", ActionKind.TRANSLATE_X, np.array([b, 0.0, 0.0]), np.zeros(3))
            if b != 0.0
            else null_action()
            for b in baselines
        ]
        return cls(tuple(actions), cfg)
