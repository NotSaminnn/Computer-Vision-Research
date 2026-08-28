# Dataset Matrix

**Audit date:** 2026-08-28 · **Machine-readable source of truth:**
`configs/datasets/external.yaml` · **Tooling:** `scripts/validate_datasets.py`,
`scripts/download_datasets.sh`

> **No external dataset was downloaded, and none is required by the smoke test or
> the Phase 1 experiment.** Everything executed in this repository so far runs on
> the self-contained synthetic benchmark.

## Status vocabulary

| Term | Meaning |
|---|---|
| `verified` | The official page was fetched on 2026-08-28 and the stated facts confirmed. |
| `ACCESS UNVERIFIED` | The paper is real and cited correctly, but the download mechanism, licence or terms were **not** confirmed. Not a claim that it is unavailable. |
| `manual` | A human must accept terms or obtain credentials. The tooling will never attempt an automated download. |
| `NOT DOWNLOADED` | Not present on this machine. The normal state. |

---

## 1. The matrix

| # | Dataset | Venue | Paper | Data source | Licence | Licence status | Auto-download | Auth | Storage | Role in Intervene3D | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **LayeredDepth** | ICCV 2025 | arXiv:2503.11633 `verified` | `github.com/princeton-vl/LayeredDepth` `verified` | benchmark data stated CC0 | `verified` | no (manual) | no | unknown | external test: interface vs behind-glass depth preserved? | ★★★★★ |
| 2 | **TransPhy3D** | arXiv 2512.23705 | `verified` | HF `Daniellesry/TransPhy3D` `verified` | not stated on pages consulted | `ACCESS UNVERIFIED` | no | unknown | unknown (~11k video sequences) | pretraining for the transition model | ★★★★★ |
| 3 | **3D Visual Illusion** | NeurIPS 2025 | arXiv:2505.13061 `verified` | `github.com/YaoChengTang/…` `verified` | not stated | `ACCESS UNVERIFIED` | no | unknown | unknown (~200k images) | closest existing analogue of our display/mirror variants | ★★★★★ |
| 4 | **Mirror3D** | CVPR 2021 | arXiv:2106.06629 `verified` | `github.com/3dlg-hcvc/mirror3d` `verified` | annotations only; frames governed by Matterport3D / ScanNet / NYUv2 | `ACCESS UNVERIFIED` | no | **yes — three separate agreements** | unknown | external test of physical mirror contact depth | ★★★★★ |
| 5 | **GIFT benchmark** | arXiv 2608.02068 | `verified` (paper) | no release page found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | the most directly comparable external evaluation, **if released** | ★★★★★ if available |
| 6 | **ClearPose** | ECCV 2022 | arXiv:2203.03890 `verified` | `github.com/opipari/ClearPose` `verified` | not stated | `ACCESS UNVERIFIED` | no | unknown | large (350k+ frames) | contact depth on real transparent objects | ★★★★☆ |
| 7 | **DREDS + STD** | ECCV 2022 | — | `github.com/PKU-EPIC/DREDS` `verified` | **CC BY-NC 4.0** | `verified` | no (non-commercial terms) | no | unknown | domain randomisation, unseen materials | ★★★★☆ |
| 8 | **Booster** | TPAMI 2023 | arXiv:2301.08245 `verified` | official page not reached | unknown | `ACCESS UNVERIFIED` | no | likely registration | unknown | independent external contact-depth test | ★★★★☆ |
| 9 | **MD-3k** | arXiv 2606.29600 | `verified` | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | 3,161 images (from GDD) | does the hypothesis generator keep multiple explanations alive? | ★★★★☆ |
| 10 | **DepthFocus synthetic** | CVPR 2026 | arXiv:2511.16993 `verified` | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | ~500k stereo pairs | optional pretraining | ★★★☆☆ |
| 11 | **MAGD** | MTAP 2024 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | 36 videos / 9,960 frames | mechanism stability through real camera motion | ★★★☆☆ |
| 12 | **MVMD** | WACV 2025 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | closest analogue of viewpoint-intervention evidence for mirrors | ★★★☆☆ |
| 13 | **PDI-Dataset** | arXiv 2605.15185 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | geometric coherence of the world model | ★★☆☆☆ |
| — | **Intervene3D-Synth** (ours) | — | this repo | `data/processed/` | MIT (this repo) | `verified` | **generated locally** | no | ~40 MB for 288 variants | **the only dataset used so far** | — |

**Storage is `unknown` wherever the official page was not reached.** Guessing a
figure would be fabricating dataset facts, which the integrity rules forbid.

---

## 2. Why a new benchmark is necessary

None of the above supplies, simultaneously:

- matched competing physical explanations of the *same* apparent geometry;
- controlled observer interventions applied identically across variants;
- ground-truth physical contact geometry;
- causal image-formation labels;
- **resolvable / non-resolvable labels**;
- minimum resolving motion.

The 3D Visual Illusion dataset comes closest (screens, mirrors, pictures), but its
categories are collected rather than *paired*, and it has no action set and no
resolvability supervision.

---

## 3. Intervene3D-Synth: what is actually generated

Produced by `scripts/generate_synthetic_data.py`; specification in
`configs/synthetic/*.yaml`.

**Per scene variant:**

| Field | Description |
|---|---|
| `scene_id`, `base_scene_id` | identity, and the matched-counterfactual group |
| `hypothesis` | the true mechanism: `direct` / `emissive` / `reflection` (+ `transmission`, `mixed` in the Phase 2 config) |
| `hypothesis_set_full` | the full competing family, serialised exactly |
| `camera` | reference intrinsics and pose `T_wc` |
| `interface` | optical interface plane and finite aperture |
| `ref_uv`, `ref_depth`, `ref_visible`, `ref_channel` | the reference observation (identical across variants) |
| `ref_contact_depth` | ground-truth first physical surface |
| `obs_*` | the pre-simulated observation for **every** action in `A` |
| `separability`, `identifiability` | oracle `Δ_ij(a)` tensor and `I_A` matrix |
| `resolvable` | **derived**, not declared: `min_{j≠*} I_A(H*,H_j) ≥ ε` |
| `mcrb`, `mcrb_analytic`, `mcrb_compensated` | operational, theoretical and homography-compensated resolving baselines |
| `z_near`, `z_far` | extremal content depths (the MCRB inputs) |
| `oracle_best_action` | the intervention an oracle would choose |
| `split` | assigned by base scene |

**Realised statistics** (`configs/synthetic/phase1.yaml`, seed 20260828):
288 variants over 96 base scenes; **178 resolvable / 110 non-identifiable
(38.2 %)**; median MCRB 0.0099 m.

**The dataset is not trivially solvable, by construction:**
- view-tracked displays are geometrically identical to a direct scene inside the
  aperture — no baseline resolves them;
- mirrors are unresolvable whenever no allowed action brings the virtual image of
  observer-attached structure inside the aperture;
- shallow scenes have an MCRB beyond the action bounds.

---

## 4. Leakage prevention

**Policy: split by underlying base scene. Never by frame, pose or rendering.**

Implemented in `data/splits.py`; assignment is a deterministic SHA-256 hash of
`base_scene_id` alone, so:

- all causal variants of a base scene always share a split;
- regenerating with more scenes never moves an existing scene between splits.

Enforced three ways: `validate_synthetic_config` rejects any policy other than
`base_scene`; `validate_dataset` runs a leakage check on every generated dataset;
and `tests/unit/test_reproducibility.py` covers determinism, stability under
growth, and the leakage detector itself.

---

## 5. Acquisition workflow

```bash
python scripts/validate_datasets.py --list          # registry with status and licence
bash   scripts/download_datasets.sh --dataset X     # prints exact instructions
python scripts/validate_datasets.py --dataset X     # after manual placement
python scripts/validate_datasets.py --all           # everything, synthetic and external
```

`download_datasets.sh` attempts an automated download only when **both** the
licence and the download permission are recorded as `verified`. As of this audit
**no registered dataset meets that bar**, so the script prints instructions for
all thirteen. That is the intended behaviour, not a missing feature.

---

## 6. Before using any external dataset (Gate 5)

1. Re-verify the official source, licence and terms — they change.
2. Record the licence verbatim in `configs/datasets/external.yaml`.
3. Confirm storage and any authentication before downloading.
4. Preserve original files; write derived data to `data/interim/` or `data/processed/`.
5. Write a `manifest.json` with per-file SHA-256 checksums and the source URL and version.
6. Never bypass an access restriction.
7. Keep several real benchmarks **out of training** so a zero-shot claim is possible.
