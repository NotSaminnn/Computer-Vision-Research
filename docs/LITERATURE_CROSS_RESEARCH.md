# Literature Cross-Research

**Audit date:** 2026-08-28
**Method:** independent web search plus direct fetches of official pages
(arXiv abstract pages, CVF Open Access, GitHub, Hugging Face). The source
document's citations were **not** taken on trust.
**Scope of this audit:** the citations that carry weight for the novelty claim,
plus the top-priority datasets. It is not exhaustive over all ~50 references in
`research(1).md`.

> **Integrity note.** Every row below is marked `VERIFIED` only if a page for
> that exact work was actually retrieved during this audit. Anything else is
> marked `NOT VERIFIED HERE`, which means "not checked", not "does not exist".

---

## 1. Verification of the novelty-critical citations

| Claim in `research(1).md` | Verification | Status |
|---|---|---|
| **GIFT: Geometry-Invariant Fine-Tuning for Non-Lambertian Monocular Depth Estimation**, arXiv:2608.02068 | Abstract page fetched. Title and authors (Fan, Chen, Wu, Li, Zeng, Cui, Xu, Huang, Yang) confirmed. Abstract confirms the described method: a lightweight post-training approach using images under varying lighting with fixed camera and geometry, reducing hallucinated depth on mirrors and glass. | **VERIFIED** |
| **LayeredDepth / Seeing and Seeing Through the Glass**, ICCV 2025 | arXiv:2503.11633; CVF Open Access page; official repo `github.com/princeton-vl/LayeredDepth`. Authors Wen, Zuo, Subramanian, Chen, Deng. 1,500 real images, ~14.2M relative-depth tuples, 15,300 synthetic images from a procedural Infinigen-Indoor-based generator. Benchmark data stated as CC0. | **VERIFIED** |
| **SeeGroup: Multi-Layer Depth Estimation of Transparent Surfaces via Self-Determined Grouping**, CVPR 2026 | arXiv:2605.28735; official repo `github.com/princeton-vl/SeeGroup`, labelled CVPR 2026 **Oral**. Authors Wen and Deng (Princeton). Confirms the point-process / unordered-events-along-a-ray formulation. | **VERIFIED** |
| **GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport**, CVPR 2026 | arXiv:2603.26181; repo `github.com/youngju-na/GLINT`; CVPR 2026 virtual page. Confirmed **Oral and Award Candidate**. Authors Na, Yun, Ryu, Kim, Yoon, Yeon (KAIST / NAVER LABS). Confirms explicit interface / transmitted / reflected radiance decomposition. | **VERIFIED** |
| **DepthFocus: Controllable Depth Estimation for See-Through Scenes**, CVPR 2026 | arXiv:2511.16993; CVF Open Access PDF. Confirms steerable ViT conditioned on a focus distance and a ~500k synthetic multi-layer stereo training set. | **VERIFIED** |
| **3D Visual Illusion Depth Estimation**, NeurIPS 2025 | arXiv:2505.13061; NeurIPS poster page; repo `github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation`. Confirms ~3,000 scenes / ~200k images (178,912 train + 617 test frames) across five illusion categories: inpainting, picture, replay (screens), holography, mirror. | **VERIFIED** |
| **One Scene, Two Depths / MD-3k**, arXiv:2606.29600 | PDF located. Confirms MD-3k as a sparse two-layer ordinal benchmark of **3,161 RGB images sourced from the GDD dataset**, and the "model-intrinsic depth-layer preference" framing. | **VERIFIED** |
| **Diffusion Knows Transparency / TransPhy3D**, arXiv:2512.23705 | Abstract page, project page `daniellli.github.io/projects/DKT/`, and a Hugging Face dataset page `Daniellesry/TransPhy3D`. Confirms ~11,000 Blender/Cycles sequences with RGB, depth and normals, and zero-shot results on ClearPose / DREDS / TransPhy3D-Test. | **VERIFIED** |
| **VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model**, arXiv:2603.12655 | Abstract and HTML located. Authors Sun, Wang, Zhang, Liu, Jia, Song, Huang, Luo; March 2026. Confirms the central architectural precedent: **forecasting frozen geometry-foundation-model features instead of future RGB**, evaluated on KITTI, Cityscapes and TartanAir. | **VERIFIED** |
| **Mirror3D: Depth Refinement for Mirror Surfaces**, CVPR 2021 | arXiv:2106.06629; CVF page; repo `github.com/3dlg-hcvc/mirror3d`; project page. Confirms 7,011 mirror instance masks / 3D planes over frames from Matterport3D, ScanNet and NYUv2. | **VERIFIED** |
| **ClearPose**, ECCV 2022, arXiv:2203.03890 | Abstract page and repo `github.com/opipari/ClearPose`. Confirms 350k+ real RGB-D frames, ~5M instance annotations, 63 transparent objects, RealSense L515. | **VERIFIED** |
| **DREDS + STD**, ECCV 2022 | Repo `github.com/PKU-EPIC/DREDS`. Confirms the split sizes and, importantly, a **CC BY-NC 4.0** licence. | **VERIFIED** |
| **Booster**, TPAMI 2023, arXiv:2301.08245 | Abstract page. Confirms 606 labelled samples from 85 scenes, unbalanced stereo pairs, material masks, ~15k unlabelled samples. Download/registration terms not reached. | **VERIFIED (paper) / access not checked** |
| **CITRIS: Causal Identifiability from Temporal Intervened Sequences** | PMLR page located (Lippe et al., ICML 2022). Confirms causal identifiability under temporal interventions as established prior art. | **VERIFIED** |
| **Next-Best-View Prediction for Active Stereo Cameras and Highly Reflective Objects**, arXiv:2202.13263 | Located. Confirms information-gain viewpoint selection for reflective objects as prior art. | **VERIFIED** |
| Video mirror detection: CVPR 2024 inconsistent motion; MVMD (WACV 2025); MiD-VMD (AAAI 2026); MAGD (MTAP 2024) | Not individually re-fetched in this audit. The source document's URLs are plausible and consistent, but they were not confirmed here. | **NOT VERIFIED HERE** |
| 3WM (ICLR 2026), P3Sim (CVPR 2026), WorldReel, WorldStereo, NeoVerse, PhysMind, GeoWAM, PDI-Bench, 4DWorldBench | Not re-fetched. | **NOT VERIFIED HERE** |
| UniDepthV2, MoGe-2, PRoPE, and the CVPR 2025 video-depth cluster | Not re-fetched. | **NOT VERIFIED HERE** |

**Outcome: no fabricated citation was found among those checked.** Every
novelty-critical paper the source document leans on exists and says what the
document claims it says. One first-pass search for "GIFT" surfaced a different
paper (arXiv:2408.06083, *Towards Robust Monocular Depth Estimation in
Non-Lambertian Surfaces*) instead; fetching the cited arXiv ID directly confirmed
GIFT is real. That near-miss is worth recording — the two papers are close
neighbours and a reviewer may well raise the 2024 one as additional prior art.

---

## 2. New work found that the source document does not cite

### 2.1 The 3D Mirage: Probing and Taming 3D Hallucinations — arXiv:2512.15423

**Submitted December 2025, revised July 2026.** Authors Hoang Nguyen,
Xiaohao Xu, Xiaonan Huang. Retrieved during this audit; an earlier version
appears under the title *Photorealistic Phantom Roads in Real Scenes:
Disentangling 3D Hallucinations from Physical Geometry*.

The abstract states the work addresses monocular depth models that
"hallucinate illusory 3D structures from planar/low-curvature but perceptually
ambiguous inputs", introduces a benchmark combining context variation with
precise annotation, proposes metrics (DCS, CCS) and a parameter-efficient
training strategy (Grounded Self-Distillation).

**Why it matters:** this is the closest work found to Intervene3D's *motivation*
and it is **not in the source document's related-work list**. It should be added.
Assessment of the overlap:

- It shares the apparent-vs-physical framing and the benchmark-with-controlled-
  variation methodology.
- Based on the abstract, it does **not** appear to address camera motion,
  action sets, identifiability, or abstention; its contribution is a
  probing benchmark plus a training strategy that *resolves* hallucination.
- It therefore reinforces the audit's standing conclusion: the apparent-vs-
  physical distinction is **not novel**, and the novelty must sit entirely in
  *action-set-dependent identifiability and explicit non-identifiability*.

Recorded as a risk in `docs/NOVELTY_RISK_REGISTER.md`.

### 2.2 Searches that returned nothing overlapping

| Query | Result |
|---|---|
| `"interventional identifiability" OR "action-set identifiability"` + camera motion + competing image-formation hypotheses + abstain | Returned interventional identifiability in SDEs and Gaussian LTI systems, CITRIS, and active-vision hypothesis testing. **No work combining action-set-dependent identifiability of competing optical explanations with an abstention output.** |
| active vision / next-best-view + resolve optical ambiguity + mirror/display + identifiability, 2026 | Returned generic NBV and active object recognition (including ambiguity-rank viewpoint selection). Nothing about optical *explanation* identifiability. |
| geometry foundation models 2026 | VGGT-World confirmed as the strongest architectural precedent for geometry-feature forecasting. |

**This does not prove novelty.** Absence of evidence in a targeted search is weak
evidence of absence, and the plan is explicit: *do not conclude "novel" merely
because no obvious paper was found.*

---

## 3. Standing conclusions

1. The source document's own novelty position survives this audit: multilayer
   depth, mirror/glass physical depth, reflection/transmission decomposition,
   motion as a mirror cue, active NBV, counterfactual world models and causal
   identifiability are all **occupied**, and must not be claimed.
2. The defensible gap remains **action-set-dependent identifiability of competing
   optical explanations, explicit prediction of non-identifiability, and a matched
   counterfactual benchmark that contains unresolvable cases**.
3. One new threat (`The 3D Mirage`) was found and does not close that gap, but
   does further crowd the apparent-vs-physical framing.
4. **MCRB has no naming collision** among the works checked, so the source
   document's name is retained.

---

## 4. Cross-research to repeat before the next milestones

Per the plan's section 39, literature checking is not a one-off. Re-run before:

| Milestone | Queries |
|---|---|
| Gate 5 — external datasets | licence and release status for every `ACCESS UNVERIFIED` row in `docs/DATASET_MATRIX.md` |
| Gate 6 — geometry foundation encoder | `latest geometry foundation models 2027`; verify MoGe-2 / VGGT repo, licence, checkpoint, GPU memory |
| Gate 7 — learned world model | `latest geometry world models 2027`; re-check VGGT-World, 3WM, P3Sim |
| Any novelty claim | `active vision hypothesis verification optical ambiguity`; `optical ambiguity benchmark physical apparent geometry intervention`; `non-identifiability abstention 3D geometry` |
| Related-work writing | individually verify every row currently marked **NOT VERIFIED HERE** |

---

## 5. Sources retrieved during this audit

- [GIFT (arXiv:2608.02068)](https://arxiv.org/abs/2608.02068)
- [LayeredDepth (arXiv:2503.11633)](https://arxiv.org/abs/2503.11633) · [repo](https://github.com/princeton-vl/LayeredDepth) · [ICCV 2025 page](https://openaccess.thecvf.com/content/ICCV2025/html/Wen_Seeing_and_Seeing_Through_the_Glass_Real_and_Synthetic_Data_ICCV_2025_paper.html)
- [SeeGroup (arXiv:2605.28735)](https://arxiv.org/abs/2605.28735) · [repo](https://github.com/princeton-vl/SeeGroup)
- [GLINT (arXiv:2603.26181)](https://arxiv.org/abs/2603.26181) · [repo](https://github.com/youngju-na/GLINT) · [CVPR 2026 oral](https://cvpr.thecvf.com/virtual/2026/oral/40261)
- [DepthFocus (arXiv:2511.16993)](https://arxiv.org/abs/2511.16993)
- [3D Visual Illusion Depth Estimation (arXiv:2505.13061)](https://arxiv.org/pdf/2505.13061) · [repo](https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation) · [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115511)
- [One Scene, Two Depths / MD-3k (arXiv:2606.29600)](https://arxiv.org/pdf/2606.29600)
- [Diffusion Knows Transparency / TransPhy3D (arXiv:2512.23705)](https://arxiv.org/abs/2512.23705) · [dataset](https://huggingface.co/datasets/Daniellesry/TransPhy3D)
- [VGGT-World (arXiv:2603.12655)](https://arxiv.org/abs/2603.12655)
- [Mirror3D (arXiv:2106.06629)](https://arxiv.org/abs/2106.06629) · [repo](https://github.com/3dlg-hcvc/mirror3d)
- [ClearPose (arXiv:2203.03890)](https://arxiv.org/abs/2203.03890) · [repo](https://github.com/opipari/ClearPose)
- [DREDS repo](https://github.com/PKU-EPIC/DREDS)
- [Booster (arXiv:2301.08245)](https://arxiv.org/abs/2301.08245)
- [CITRIS (PMLR v162)](https://proceedings.mlr.press/v162/lippe22a/lippe22a.pdf)
- [NBV for reflective objects (arXiv:2202.13263)](https://arxiv.org/abs/2202.13263)
- [The 3D Mirage (arXiv:2512.15423)](https://arxiv.org/abs/2512.15423) — **newly found, not in the source document**
- [Towards Robust Monocular Depth in Non-Lambertian Surfaces (arXiv:2408.06083)](https://arxiv.org/abs/2408.06083) — **newly found neighbour of GIFT**


---

## 6. Second pass — 2026-08-29

Searches run while preparing the selector and abstention claims. Each entry
states what was actually confirmed, and what it costs us.

### 6.1 Model-discrimination experiment design — **a field we had missed**

| Finding | Source | Consequence |
|---|---|---|
| Optimal design *for discriminating between rival models* is a mature sub-field with its own criteria, distinct from information gain | Hunter & Reiner (1965) — first recorded criterion; Fedorov & Malyutov (1972); Atkinson & Cox (1974); **Atkinson & Fedorov (1975) — T-optimality** | **The maximin criterion is not ours.** It is classical statistics. Claiming it would have been caught. |
| The Bayesian literature states explicitly that when the goal is deciding *which* model holds, criteria developed for model discrimination are the appropriate ones rather than generic expected information gain | Modern Bayesian Experimental Design (arXiv:2302.14545); optimal Bayesian design for model discrimination via classification (arXiv:1809.05301) | Our finding is a **transfer**, not a discovery. It is still publishable because vision NBV overwhelmingly uses EIG and does not cite this literature — but the framing must be transfer-plus-measurement. |
| Finding T-optimal designs is hard: the criterion is non-differentiable with nested optimisation, and closed forms exist only for simple cases | Robust T-optimal discriminating designs (Ann. Statist. 41(4)); computing T-optimal designs via nested semi-infinite programming (arXiv:2208.13439) | Supports our use of a discrete, enumerable action set: the hard part of T-optimality is the continuous design space, which we do not have. |

**Verdict:** downgrade the maximin claim from "new criterion" to "criterion
imported from experimental design, with the first measurement of why the
prevailing vision objective fails." Cite Atkinson & Fedorov as support.

### 6.2 Selective prediction and abstention

| Finding | Source | Consequence |
|---|---|---|
| Selective prediction with a reject option is ~60 years old; Chow's rule is the optimal decision under a cost-sensitive loss given posterior class probabilities | Chow (1970); selective classification for deep networks (arXiv:1705.08500) | **Abstention itself is not novel.** Only the *quantity abstained on* can be. |
| Conformal prediction supplies finite-sample coverage guarantees for selective prediction | standard | The strongest opponent, and the one we must beat. Now measured — see E13. |
| Depth estimation and 3-D reconstruction are **not** prominent in the selective-prediction literature | absence noted across searches | Genuine gap, but argued from absence — state it as "we found no work", never as "no work exists". |

### 6.3 Depth foundation models on non-Lambertian surfaces

| Finding | Source | Consequence |
|---|---|---|
| Depth Anything V2 is documented as vulnerable to transparent objects and reflections | arXiv:2406.09414; arXiv:2408.06083 | The failure we measure is acknowledged by the field, which strengthens the motivation and weakens any claim to have discovered it. |
| Fine-tuning approaches for non-Lambertian depth exist and are active | GIFT (arXiv:2608.02068); Costanzino et al., ICCV 2023 (arXiv:2307.15052); `prompt-depth-anything-*-transparent-hf` | **Do not train a model to fix this.** That space is occupied and our thesis is that some cases are unresolvable in principle. Test whether *their* training helps instead. |

### 6.4 Licence facts corrected on inspection

Model cards are not a reliable licence source. Checked against source repositories:

| Model | HF card | Actual | Usable |
|---|---|---|---|
| Metric3D | none declared | **BSD 2-Clause** | yes |
| UniDepth v2 | none declared | **CC BY-NC 4.0** | research only |
| Depth-Anything-V3 (`DA3-*`) | — | Apache-2.0, except `DA3NESTED-GIANT-*` which is CC BY-NC 4.0 | yes, with care |
| Depth-Anything-V2 Base/Large | cc-by-nc-4.0 | as stated | research only |
| VGGT-1B | cc-by-nc-4.0 | as stated | research only |
| 3D Visual Illusion authors' own model | **none declared** | not determined | **not used** — licence must be confirmed with the authors first |

An earlier note in this session excluded UniDepth for having "no licence". That
was wrong: the card omits the field, the repository carries CC BY-NC 4.0.

### 6.5 Re-verification obligation

This pass covers work visible up to 2026-08-29 and used web search plus source
repositories. It is not exhaustive, and the area moves quickly. Re-run these
queries before any submission.
