"""Hypothesis representation, validation, equality and serialisation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from intervene3d.geometry.planes import Aperture


class OpticalMechanism(str, Enum):
    """The five physical image-formation mechanisms of the hypothesis family."""

    DIRECT = "direct"
    REFLECTION = "reflection"
    TRANSMISSION = "transmission"
    EMISSIVE = "emissive"
    MIXED = "mixed"

    @property
    def symbol(self) -> str:
        return {
            "direct": "H_D",
            "reflection": "H_R",
            "transmission": "H_T",
            "emissive": "H_E",
            "mixed": "H_M",
        }[self.value]


MECHANISM_LABELS: dict[OpticalMechanism, str] = {
    OpticalMechanism.DIRECT: "Direct",
    OpticalMechanism.REFLECTION: "Mirror",
    OpticalMechanism.TRANSMISSION: "Glass",
    OpticalMechanism.EMISSIVE: "Display",
    OpticalMechanism.MIXED: "Mixed",
}

#: Parameters each mechanism requires.  Validation is strict: a hypothesis that
#: cannot be simulated must fail loudly at construction, not silently at run time.
_REQUIRED_PARAMS: dict[OpticalMechanism, tuple[str, ...]] = {
    OpticalMechanism.DIRECT: (),
    OpticalMechanism.REFLECTION: (),
    OpticalMechanism.TRANSMISSION: ("thickness", "refractive_index"),
    OpticalMechanism.EMISSIVE: ("display_mode",),
    OpticalMechanism.MIXED: ("reflectance",),
}

VALID_DISPLAY_MODES = ("static", "view_tracked")


@dataclass(frozen=True)
class Hypothesis:
    """A candidate physical explanation of an apparent geometric observation.

    Parameters
    ----------
    mechanism:
        Which image-formation process is hypothesised.
    interface:
        The finite interface (plane + rectangular aperture).  For every optical
        mechanism this is the physical surface -- the mirror, the screen, the
        pane.  For :attr:`OpticalMechanism.DIRECT` it is instead a purely
        **occluding opening** (think of a doorway or a window frame) and the
        contact geometry stays on the content, not on the plane.

        Giving ``H_D`` the same opening as the optical variants is what makes the
        benchmark matched: otherwise aperture clipping alone would betray the
        mechanism, and no reasoning about optics would be required.  ``None`` is
        permitted for ``H_D`` when no opening is modelled.
    params:
        Mechanism-specific parameters (see ``_REQUIRED_PARAMS``).
    name:
        Human-readable identifier; defaults to the mechanism symbol.
    """

    mechanism: OpticalMechanism
    interface: Aperture | None = None
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        mech = OpticalMechanism(self.mechanism)
        object.__setattr__(self, "mechanism", mech)
        object.__setattr__(self, "params", dict(self.params))
        if not self.name:
            object.__setattr__(self, "name", mech.symbol)
        self.validate()

    # ---------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise ``ValueError`` if this hypothesis cannot be simulated."""
        mech = self.mechanism
        # H_D may carry an occluding opening, or none at all.  Every optical
        # mechanism *requires* an interface: without a surface there is nothing
        # to reflect from, display on or transmit through.
        if mech is not OpticalMechanism.DIRECT and self.interface is None:
            raise ValueError(f"{mech.symbol} requires an optical interface (plane + aperture)")

        missing = [p for p in _REQUIRED_PARAMS[mech] if p not in self.params]
        if missing:
            raise ValueError(f"{mech.symbol} is missing required parameters: {missing}")

        if mech is OpticalMechanism.TRANSMISSION:
            if float(self.params["thickness"]) < 0.0:
                raise ValueError("transmission thickness must be non-negative")
            if float(self.params["refractive_index"]) < 1.0:
                raise ValueError("refractive_index must be >= 1.0")
        if mech is OpticalMechanism.EMISSIVE:
            mode = self.params["display_mode"]
            if mode not in VALID_DISPLAY_MODES:
                raise ValueError(f"display_mode must be one of {VALID_DISPLAY_MODES}, got {mode!r}")
        if mech is OpticalMechanism.MIXED:
            r = float(self.params["reflectance"])
            if not 0.0 <= r <= 1.0:
                raise ValueError("mixed reflectance must lie in [0, 1]")

    # ------------------------------------------------------------------- helpers
    @property
    def symbol(self) -> str:
        return self.mechanism.symbol

    @property
    def label(self) -> str:
        return MECHANISM_LABELS[self.mechanism]

    def with_interface(self, interface: Aperture | None) -> Hypothesis:
        return Hypothesis(self.mechanism, interface, dict(self.params), self.name)

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism.value,
            "name": self.name,
            "interface": None if self.interface is None else self.interface.to_dict(),
            "params": _jsonable(self.params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Hypothesis:
        interface = payload.get("interface")
        return cls(
            OpticalMechanism(payload["mechanism"]),
            None if interface is None else Aperture.from_dict(interface),
            dict(payload.get("params", {})),
            payload.get("name", ""),
        )

    # ------------------------------------------------------------------ equality
    def _key(self) -> str:
        """Canonical, hashable, float-rounded fingerprint used for equality."""
        return json.dumps(_round_floats(self.to_dict()), sort_keys=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hypothesis):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"Hypothesis({self.symbol}{', ' + extra if extra else ''})"


@dataclass(frozen=True)
class HypothesisSet:
    """An ordered, immutable collection of competing hypotheses.

    Order matters: it fixes the index convention of belief vectors, separability
    matrices and predicted-transition tensors throughout the codebase.
    """

    hypotheses: tuple[Hypothesis, ...]

    def __post_init__(self) -> None:
        hyps = tuple(self.hypotheses)
        if len(hyps) < 2:
            raise ValueError("A hypothesis set needs at least two competing hypotheses")
        names = [h.name for h in hyps]
        if len(set(names)) != len(names):
            raise ValueError(f"Hypothesis names must be unique, got {names}")
        object.__setattr__(self, "hypotheses", hyps)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self.hypotheses)

    def __getitem__(self, index: int) -> Hypothesis:
        return self.hypotheses[index]

    @property
    def names(self) -> list[str]:
        return [h.name for h in self.hypotheses]

    @property
    def labels(self) -> list[str]:
        return [h.label for h in self.hypotheses]

    @property
    def mechanisms(self) -> list[OpticalMechanism]:
        return [h.mechanism for h in self.hypotheses]

    def index_of_name(self, name: str) -> int:
        for i, h in enumerate(self.hypotheses):
            if h.name == name:
                return i
        raise KeyError(f"no hypothesis named {name!r} in {self.names}")

    def index_of_mechanism(self, mechanism: OpticalMechanism | str) -> int:
        mechanism = OpticalMechanism(mechanism)
        for i, h in enumerate(self.hypotheses):
            if h.mechanism is mechanism:
                return i
        raise KeyError(f"no hypothesis with mechanism {mechanism} in {self.mechanisms}")

    def uniform_prior(self) -> np.ndarray:
        return np.full(len(self), 1.0 / len(self), dtype=np.float64)

    def pairs(self) -> list[tuple[int, int]]:
        """All unordered index pairs ``i < j``."""
        n = len(self)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [h.to_dict() for h in self.hypotheses]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HypothesisSet:
        return cls(tuple(Hypothesis.from_dict(h) for h in payload["hypotheses"]))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _round_floats(obj: Any, ndigits: int = 9) -> Any:
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj
