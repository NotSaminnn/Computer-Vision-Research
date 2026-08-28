# Notebooks

**Policy: no pipeline code lives here.** Everything executable is a module under
`src/intervene3d/` or a script under `scripts/`, so that every result is
reproducible from a command rather than from a saved cell ordering. Notebooks are
for exploration only, and anything worth keeping should be moved into the package
with a test.

A useful starting point:

```python
import sys; sys.path.insert(0, "../src")
import numpy as np
from intervene3d.config import validate_synthetic_config
from intervene3d.data.synthetic import (
    action_space_from_config, build_hypothesis_set, generate_base_scene, reference_observation,
)
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import AnalyticalTransitionModel

cfg  = validate_synthetic_config({})
base = generate_base_scene(np.random.default_rng([0, 0]), cfg, 0)
hyps = build_hypothesis_set(base.content.interface, cfg, np.random.default_rng([0, 0, 7]))
f0   = reference_observation(base.content, hyps[0]).feature

est = GeometrySeparabilityEstimator(
    AnalyticalTransitionModel(), DistanceWeights.from_dict(cfg["identifiability"]["distance"])
)
sep = est.pairwise_over_actions(
    f0, hyps, action_space_from_config(cfg["action_space"]),
    markers_cam=base.content.observer_markers_cam,
)
print(hyps.names)
print(sep.max(axis=0))   # I_A(H_i, H_j)
```

To load a finished run instead:

```python
from intervene3d.utils.io import load_json
metrics = load_json("../experiments/phase1_problem_existence/<run>/metrics/metrics.json")
```
