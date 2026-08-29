# Dataset Matrix

**Audit date:** 2026-08-28 · **Re-verified:** 2026-08-29 · **Machine-readable
source of truth:** `configs/datasets/external.yaml` · **Tooling:**
`scripts/validate_datasets.py`, `scripts/download_datasets.sh`

> **No external dataset is required by the smoke test or the Phase 1 experiment,
> and none has been used in any claimed result.** Every number in
> `docs/EXPERIMENT_PLAN.md` comes from the self-contained synthetic benchmark.
> External data acquired since 2026-08-29 is available for *future* evaluation
> and has not entered any reported metric.

## 0. What changed on 2026-08-29

The 2026-08-28 audit could not confirm licences or download mechanisms from the
project pages alone, so eight entries sat at `ACCESS UNVERIFIED`. Re-verification
went to the primary machine-readable sources instead — the Hugging Face dataset
API (`/api/datasets/<id>` for licence and gating, `/tree/main?recursive=true` for
per-file byte counts) and the official GitHub READMEs. Result:

| Dataset | Was | Now | Evidence |
|---|---|---|---|
| LayeredDepth | licence `verified`, auto-download `unknown` | **auto-download permitted** | `cardData.license = cc0-1.0`, `gated = false` |
| LayeredDepth-Syn | folded into the LayeredDepth row | **separate entry, permitted** | `bsd-3-clause` — *different licence from the benchmark repo*, so it cannot share a row |
| 3D Visual Illusion | `ACCESS UNVERIFIED` | **permitted** | `apache-2.0`, `gated = false`, on both the Real and Virtual repos |
| TransPhy3D | `ACCESS UNVERIFIED` | **permitted**, size-capped | `apache-2.0`, `gated = false` |
| ClearPose | `ACCESS UNVERIFIED` | licence `verified`, **still manual** | README states MIT, but the data is on a Dropbox share link — no stable URL, no checksum, not reproducible |
| DREDS | already `verified` (CC BY-NC 4.0) | unchanged, **still manual** | non-commercial terms; official links are not automatable |
| Mirror3D, Booster, MD-3k, MAGD, MVMD, GIFT, PDI, DepthFocus | `ACCESS UNVERIFIED` | unchanged | no confirmed public release, or gated behind separate signed agreements |

**Sizes below are measured byte counts** from that file listing, not estimates.
Where a size is still `unknown`, the official page was not reached — guessing one
would be fabricating a dataset fact.

Two corrections worth calling out, because both would have caused silent damage:

- **3D Visual Illusion's 455 GB "VirtualData" training split has no ground-truth
  depth.** Its `depth` files are DepthAnythingV2 *predictions* rescaled by the
  provided `scale_factors*.csv`. Training on it would import another model's
  errors as labels. Only the 8.10 GB RealData split carries true disparity — 455
  rectified stereo pairs from a ZED rig cross-calibrated to an Intel RealSense
  L515 (verified by opening the archive, not from the README). The registry wires
  up RealData only, deliberately.
- **LayeredDepth's 15.15 GB test split has no local labels.** Scoring goes
  through the authors' server at `layereddepth.cs.princeton.edu`, three
  submissions per user per seven days. The 4.13 GB validation split (300
  examples, `image.png` + `tuples.json`) is the only one that supports a local
  metric — its card declares exactly these two splits, 300 + 1,200 = the paper's
  1,500 images.
- **The repository also carries an undeclared `data/train-00000-of-00001.parquet`.**
  It holds 10 examples and no `tuples.json`, and the dataset card's `configs`
  does not reference it. It is not a training split. Fetched and inspected on
  2026-08-29, then deleted; no variant exposes it. Worth recording because the
  filename invites exactly the wrong assumption.

## Status vocabulary

| Term | Meaning |
|---|---|
| `verified` | The stated facts were confirmed against the primary source (dates above). |
| `ACCESS UNVERIFIED` | The paper is real and cited correctly, but the download mechanism, licence or terms were **not** confirmed. Not a claim that it is unavailable. |
| `manual` | A human must accept terms, obtain credentials, or use a host that exposes no stable checksummable URL. The tooling will never attempt an automated download. |
| `NOT DOWNLOADED` | Not present on this machine. The normal state. |

---

## 1. The matrix

| # | Dataset | Venue | Paper | Data source | Licence | Licence status | Auto-download | Auth | Storage | Role in Intervene3D | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **LayeredDepth** | ICCV 2025 | arXiv:2503.11633 `verified` | HF `princeton-vl/LayeredDepth` `verified` | **CC0-1.0** | `verified` | **yes** | no | 19.27 GB (val 4.13 = 300 ex. + GT / test 15.15 = 1,200 ex., images only) | external test: interface vs behind-glass depth preserved? | ★★★★★ |
| 2 | **TransPhy3D** | arXiv 2512.23705 | `verified` | HF `Daniellesry/TransPhy3D` `verified` | **Apache-2.0** | `verified` | **yes, capped** | no | ~1.5 TB total; only `sample` 0.72 GB and `test` 4.12 GB are wired up | transition model (Gate 7) — **the only registered dataset with per-frame camera extrinsics**, i.e. real observer motion | ★★★★★ |
| 3 | **3D Visual Illusion** | NeurIPS 2025 | arXiv:2505.13061 `verified` | HF `AdamYao/…_Real` + `AdamYao/…` `verified` | **Apache-2.0** | `verified` | **yes, real split only** | no | Real 8.10 GB (true GT); Virtual 455.52 GB (**depth = DepthAnythingV2 predictions, not GT**) | closest existing analogue of our display/mirror variants | ★★★★★ |
| 4 | **Mirror3D** | CVPR 2021 | arXiv:2106.06629 `verified` | `github.com/3dlg-hcvc/mirror3d` `verified` | annotations only; frames governed by Matterport3D / ScanNet / NYUv2 | `ACCESS UNVERIFIED` | no | **yes — three separate agreements** | unknown | external test of physical mirror contact depth | ★★★★★ |
| 5 | **GIFT benchmark** | arXiv 2608.02068 | `verified` (paper) | no release page found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | the most directly comparable external evaluation, **if released** | ★★★★★ if available |
| 6 | **LayeredDepth-Syn** | ICCV 2025 | arXiv:2503.11633 `verified` | HF `princeton-vl/LayeredDepth-Syn` `verified` | **BSD-3-Clause** (≠ the benchmark's CC0) | `verified` | **yes** | no | 24.75 GB (train 23.95 / val 0.80) | optional pretraining with *true* multi-layer GT | ★★★★☆ |
| 7 | **ClearPose** | ECCV 2022 | arXiv:2203.03890 `verified` | `github.com/opipari/ClearPose` `verified` | **MIT** (README) | `verified` | no — Dropbox share, no stable URL or checksum | no | large (350k+ frames) | contact depth on real transparent objects | ★★★★☆ |
| 8 | **DREDS + STD** | ECCV 2022 | — | `github.com/PKU-EPIC/DREDS` `verified` | **CC BY-NC 4.0** | `verified` | no (non-commercial terms) | no | unknown | domain randomisation, unseen materials | ★★★★☆ |
| 9 | **Booster** | TPAMI 2023 | arXiv:2301.08245 `verified` | official page not reached | unknown | `ACCESS UNVERIFIED` | no | likely registration | unknown | independent external contact-depth test | ★★★★☆ |
| 10 | **MD-3k** | arXiv 2606.29600 | `verified` | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | 3,161 images (from GDD) | does the hypothesis generator keep multiple explanations alive? | ★★★★☆ |
| 11 | **DepthFocus synthetic** | CVPR 2026 | arXiv:2511.16993 `verified` | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | ~500k stereo pairs | optional pretraining | ★★★☆☆ |
| 12 | **MAGD** | MTAP 2024 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | 36 videos / 9,960 frames | mechanism stability through real camera motion | ★★★☆☆ |
| 13 | **MVMD** | WACV 2025 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | closest analogue of viewpoint-intervention evidence for mirrors | ★★★☆☆ |
| 14 | **PDI-Dataset** | arXiv 2605.15185 | not re-verified | none found | unknown | `ACCESS UNVERIFIED` | no | unknown | unknown | geometric coherence of the world model | ★★☆☆☆ |
| — | **Intervene3D-Synth** (ours) | — | this repo | `data/processed/` | MIT (this repo) | `verified` | **generated locally** | no | ~40 MB for 288 variants | **the only dataset behind any reported result** | — |

**Storage is `unknown` wherever the official page was not reached.** Guessing a
figure would be fabricating dataset facts, which the integrity rules forbid.
Where a size *is* given it is a measured byte count from the host's own file
listing, taken on 2026-08-29.

`verified` licence and `Auto-download: no` is a normal, deliberate combination:
ClearPose (MIT) and DREDS (CC BY-NC 4.0) are freely usable, but neither is hosted
anywhere that yields a stable, checksummable URL, so an automated fetch could not
be made reproducible. Permission is about *mechanism* as much as licence.

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

For **external sequence data** the equivalent unit is the rendered sequence: split
by sequence (one shard = one sequence in TransPhy3D), never by frame or frame
pair. Frames within a sequence are near-duplicates, so a frame-level split reports
a validation number the model has effectively already seen. This is enforced in
`models/torch_transition.py`: `build_pair_dataset` emits one group label **per
surviving row**, and `train_torch_residual` holds out whole groups. Reconstructing
those labels positionally is what a caller must not do -- pairs that yield no rows
are skipped, and a positional mapping then mislabels every row after the first
skip. `tests/unit/test_torch_transition.py` pins both properties.

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
python scripts/validate_datasets.py --list                    # registry: status, licence, local presence
bash   scripts/download_datasets.sh --dataset X --dry-run     # list remote files and exact byte count
bash   scripts/download_datasets.sh --dataset X --variant V   # fetch (needs --yes above 1 GB)
python scripts/validate_datasets.py --dataset X               # verify checksums against manifest.json
python scripts/validate_datasets.py --all                     # everything, synthetic and external
```

`download_datasets.sh` fetches only when **both** the licence and the download
permission are recorded as verified in `configs/datasets/external.yaml` **and**
the entry carries a `fetch:` block naming the remote repository. Four of the
fourteen entries meet that bar; the other ten print acquisition instructions and
refuse. Refusal is the intended behaviour, not a missing feature.

The order of operations is fixed and enforced in code
(`src/intervene3d/data/external/fetchers.py`):

1. **check policy** — licence `verified` *and* permission `true`, re-checked at
   fetch time rather than trusted from the plan;
2. **list before transferring** — the exact file count and byte total are read
   from the host and printed. Nothing of unknown size is ever pulled;
3. **hold anything over 1 GB** until `--yes` is passed explicitly, and refuse
   outright if free disk is under the transfer size + 10 %;
4. **transfer**, resuming a partial `.part` file rather than restarting it;
5. **verify** each file against *the publisher's own* SHA-256 — Hugging Face
   exposes it as `lfs.oid`, so integrity is checked against the source, not
   against something we computed from what we happened to receive;
6. **record** `manifest.json`: per-file SHA-256, the resolved commit SHA (not
   just the `main` label), the source URL, the allow-patterns and the licence.

Point 6 is what makes an external result reproducible: the manifest pins an
immutable revision, so "we evaluated on LayeredDepth" becomes "we evaluated on
`princeton-vl/LayeredDepth@a2aad776…`".

**Variants.** A single dataset can publish splits three orders of magnitude apart
in size, so each entry declares named variants and `--variant` selects one. This
is what keeps 3D Visual Illusion's 8.10 GB Real split separable from its 455 GB
Virtual one, and it is why `transphy3d` exposes only `sample` (0.75 GB) and
`test` (4.12 GB) rather than its ~1.5 TB training set.

A variant exists only for data the dataset card actually declares *and* that is
useful here. LayeredDepth's stray `data/train-*.parquet` has neither property, so
it has no variant — see §0.

**Credentials.** `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` are forwarded if the user
has already set them. Nothing here creates, requests, prompts for or stores a
credential, and no gated dataset is in the permitted set.

### 5.1 What is present on this machine

Acquired 2026-08-29; none of it has entered any reported result.

| Dataset | Variant | Path | Size | Licence |
|---|---|---|---|---|
| LayeredDepth | `validation` | `data/raw/layereddepth/validation/` | 4.13 GB · 300 ex. | CC0-1.0 |
| LayeredDepth-Syn | `validation` | `data/raw/layereddepth_syn/validation/` | 0.80 GB · 500 ex. | BSD-3-Clause |
| LayeredDepth-Syn | `train` | `data/raw/layereddepth_syn/train/` | 23.95 GB · 14,800 ex. | BSD-3-Clause |
| 3D Visual Illusion | `real` | `data/raw/visual_illusion_3d/real/` | 8.10 GB · 455 stereo pairs | Apache-2.0 |
| 3D Visual Illusion | `scale_factors` | `data/raw/visual_illusion_3d/virtual_meta/` | 0.03 GB | Apache-2.0 |
| TransPhy3D | `sample` | `data/raw/transphy3d/sample/` | 0.72 GB · 5 sequences | Apache-2.0 |
| TransPhy3D | `test` | `data/raw/transphy3d/test/` | 4.12 GB · 28 sequences | Apache-2.0 |

**`transphy3d/sample` is a strict subset of `transphy3d/test`** (shards 10-14 of
28). They must never be concatenated -- doing so would duplicate five sequences
and put the same frames on both sides of a train/validation split.

Re-verify with `python scripts/validate_datasets.py --all`.

Formats differ: LayeredDepth and LayeredDepth-Syn ship as parquet, so reading
them needs `pip install "pyarrow>=15"` (the `data` extra); TransPhy3D ships as
WebDataset `.tar` shards readable with plain `tarfile`; 3D Visual Illusion Real
is a single `tar.gz`. Acquisition itself needs nothing beyond the standard
library.

Every acquired variant was opened and inspected on 2026-08-29, not merely
downloaded:

| Dataset | Per-example contents | Why it matters here |
|---|---|---|
| LayeredDepth `validation` | `image.png` + `tuples.json` — `layer_all` and `layer_first` ordinal **pairs, triplets and quads**, each with an `is_real` flag | ordinal multi-layer relations, so "interface vs behind-glass" can be scored without metric depth |
| LayeredDepth-Syn `validation` | `image.png` + `depth_1..depth_8.png` | eight ray-ordered depth layers — precisely the structure `H_D` and `H_T` disagree about, with *true* GT |
| TransPhy3D `sample` | 120 frames/shard: `image.png`, `depth.png` + `depth.json` (a `max_depth` scale), `normal.png`, `metadata.json` with **per-frame 4×4 extrinsics and 3×3 normalised intrinsics** | the only registered dataset carrying real observer motion; a frame pair `(t, t+k)` is a genuine `(F_t, a)` observation for the transition model |
| 3D Visual Illusion `real` | 6,611 files. `test/` holds **455 rectified stereo pairs**: `left/*.png`, `right/*.png`, `disp/*.pfm` (float disparity), `mask/*.jpg`. Plus raw captures under `Dataset/`, `Dataset_new_structure/`, `Dataset_rect/`, SAM masks under `SAM/`, and `calib/{L515_calib,ZED_calib,L515_ZEDleft}.yaml` | true disparity from a **ZED stereo rig cross-calibrated to an Intel RealSense L515** (the yaml carries intrinsics, distortion and the L515→ZED-left extrinsic E). Categories present: `PaperOnTable`, `PosterAndObject`, `video`, `video_monitor` |

TransPhy3D's camera path is the renderer's, not chosen by any agent, so it can
supply the **transition model** but not the **selector** — the intervention-choice
question still has no external dataset.

---

## 6. Before using any external dataset (Gate 5)

1. Re-verify the official source, licence and terms — they change.
2. Record the licence verbatim in `configs/datasets/external.yaml`.
3. Confirm storage and any authentication before downloading.
4. Preserve original files; write derived data to `data/interim/` or `data/processed/`.
5. Write a `manifest.json` with per-file SHA-256 checksums and the source URL and version.
6. Never bypass an access restriction.
7. Keep several real benchmarks **out of training** so a zero-shot claim is possible.
