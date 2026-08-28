"""Explicit representation of controlled observer (camera) interventions.

An action is ``a = Delta C in SE(3)`` applied **in the reference camera frame**:
``T_wc_new = T_wc_ref @ delta``.  The action space is bounded and configurable;
arbitrary motions are never assumed.
"""

from intervene3d.interventions.action_space import ActionSpace, ActionSpaceConfig
from intervene3d.interventions.actions import Action, ActionKind, null_action

__all__ = ["Action", "ActionKind", "null_action", "ActionSpace", "ActionSpaceConfig"]
