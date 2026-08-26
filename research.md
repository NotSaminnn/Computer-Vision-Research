Yes. Based on all of the iterations, I would now freeze the project around a much narrower and more defensible idea.

The main correction is this: **do not build a paper about “better depth for mirrors/glass using a world model.”** Too many individual ingredients are already occupied. The paper should introduce a **new vision problem**: determining whether the physical cause of an ambiguous geometric observation is identifiable from available visual interventions, and if so, which intervention resolves it most efficiently. This is consistent with the strongest conclusion from our previous novelty audit. 

# Final research concept

## **Intervene3D: Interventional Identifiability of Physical Geometry under Optical Ambiguity**

### One-sentence pitch

> When several physical explanations produce the same apparent 3D structure, Intervene3D predicts whether they can be distinguished visually at all, which camera motion would distinguish them, and what physical geometry is actually justified by the observations.

The intellectual center is therefore:

$$
\boxed{
\text{Is the physical interpretation of this visual geometry identifiable under the allowed observer interventions?}
}
$$

not:

$$
\text{What is the depth?}
$$

and not:

$$
\text{Where should the camera move next?}
$$

---

# 1. The exact problem

Suppose a camera sees something that looks like a corridor.

Several hypotheses may explain the observation:

$$
H_D=\text{direct physical corridor}
$$

$$
H_R=\text{corridor reflected by a mirror}
$$

$$
H_T=\text{corridor transmitted through glass}
$$

$$
H_E=\text{corridor emitted/displayed by a screen}
$$

$$
H_M=\text{mixed optical process}.
$$

At the initial view \(C_0\),

$$
p(I_0|H_D)
\approx
p(I_0|H_R)
\approx
p(I_0|H_T)
\approx
p(I_0|H_E).
$$

The image alone may therefore not contain enough information.

Now intervene on observer pose:

$$
a = \Delta C \in SE(3).
$$

Each physical explanation implies a different distribution over future observations:

$$
p(O'|H_k,a).
$$

The central quantity becomes

$$
\Delta_{ij}(a)
=
D\left(
p(O'|H_i,a),
p(O'|H_j,a)
\right).
$$

If

$$
\Delta_{ij}(a)
$$

is large, action \(a\) separates the hypotheses.

Then define **action-set identifiability**

$$
\boxed{
\mathcal I_{\mathcal A}(H_i,H_j)
=
\max_{a\in\mathcal A}
\Delta_{ij}(a)
}
$$

for an allowed action set \(\mathcal A\).

The critical point is that identifiability is **relative to an action set and a perceptual resolution**, not an absolute property.

If

$$
\mathcal I_{\mathcal A}(H_i,H_j)<\epsilon,
$$

the correct conclusion is

$$
\boxed{\text{not identifiable under }\mathcal A}
$$

rather than forcing the model to choose one hypothesis.

That is the project.

---

# 2. Why this is scientifically interesting

Current depth systems usually behave as if

$$
I\rightarrow D.
$$

Multi-layer methods now acknowledge

$$
I\rightarrow\{D_1,\ldots,D_K\}.
$$

LayeredDepth introduced multi-layer depth estimation for transparent scenes, and SeeGroup goes further by modeling the layers as unordered events along each camera ray. ([Open Access CVF][1])

Recent work explicitly shows that foundation depth models can expose multiple valid geometric interpretations of the same scene. ([arXiv][2])

But these formulations still largely ask:

> What geometry is represented?

Your project asks:

> **Can vision establish which image-formation explanation is physically responsible for that geometry?**

That is a different question.

---

# 3. Scope the term “physical” carefully

This is important for the paper.

Don't define “physical” as “real-looking.”

Define a **contact geometry**

$$
D_{\text{contact}}
$$

as the first physical surface an agent moving along the corresponding ray would encounter.

Examples:

| Observation  | Apparent geometry     | Contact geometry  |
| ------------ | --------------------- | ----------------- |
| Wall         | wall                  | wall              |
| Window       | room behind glass     | glass interface   |
| Mirror       | reflected room/person | mirror surface    |
| TV           | displayed corridor    | screen plane      |
| Photograph   | pictured scene        | paper/wall        |
| Open doorway | room behind doorway   | actual background |

GIFT is especially important prior work here: it explicitly shows that monocular depth models can predict reflected content in mirrors or transmitted content behind glass rather than the actual physical surface, and adapts them toward the true surface. ([arXiv][3])

Therefore **physical-versus-apparent depth itself is not your novelty**.

Your novelty begins with:

$$
\boxed{
\text{When is that distinction visually identifiable?}
}
$$

---

# 4. The three contributions I would claim

Keep the claims to **three**.

### Contribution 1 — New problem: Interventional Geometric Identifiability

Introduce the task of determining whether competing causal image-formation explanations of apparent geometry can be distinguished by an allowed set of observer interventions.

This includes predicting:

$$
P(H_k|I_{1:t})
$$

and

$$
P(\text{resolvable}|\mathcal A,I_{1:t}).
$$

This should be the paper's **main novelty claim**.

### Contribution 2 — Method: Hypothesis-conditioned geometry world model

Given a candidate optical hypothesis \(H_k\) and camera intervention \(a\), predict what the geometric representation should become:

$$
\boxed{
p_\theta(
F_{t+1}
|
F_t,H_k,a
)
}
$$

where \(F_t\) is a geometry-foundation-model representation rather than raw RGB.

The model is therefore not introduced as a new world model per se.

Its purpose is:

$$
\boxed{
\text{predict the outcome of a visual experiment under a specific physical explanation}
}
$$

and compare competing explanations.

### Contribution 3 — Benchmark: matched ambiguity and resolvability

Build a benchmark where:

* different physical mechanisms intentionally produce closely matched initial observations;
* controlled camera interventions are available;
* physical contact geometry is known;
* the causal image-formation mechanism is known;
* some examples become distinguishable after movement;
* some remain indistinguishable within the permitted action set.

This benchmark may end up being as important as the architecture.

This three-part formulation is much stronger than treating the method as multilayer depth + world model + active perception. Our previous audit reached the same conclusion. 

---

# 5. Proposed architecture

I would deliberately keep the architecture relatively simple.

## Stage A — Geometry foundation encoder

Input:

$$
I_t
$$

Output:

$$
F_t=E(I_t).
$$

Use a **frozen or lightly adapted geometry foundation model**.

A VGGT/MoGe-style representation is preferable to an RGB/video-diffusion latent because geometry is the object of interest.

This choice is also consistent with recent work such as VGGT-World, which explicitly forecasts frozen geometry-foundation-model features instead of spending world-model capacity on raw photometric prediction. ([arXiv][4])

So:

$$
RGB
\rightarrow
\boxed{\text{geometry latent}}
$$

not

$$
RGB
\rightarrow
\text{giant video generator}.
$$

---

# 6. Stage B — Candidate causal hypotheses

For version 1, keep the hypothesis family explicit:

$$
\mathcal H=
\{
H_D,
H_R,
H_T,
H_E,
H_M
\}.
$$

Where:

* \(H_D\): direct view;
* \(H_R\): reflection;
* \(H_T\): transmission/refraction;
* \(H_E\): emissive/display-induced geometry;
* \(H_M\): mixed mechanism.

Do **not** include “unidentifiable” as a physical mechanism.

Unidentifiable is a conclusion about

$$
(\mathcal H,\mathcal A,O)
$$

not a light-transport class.

That conceptual separation matters.

---

# 7. Stage C — Intervention-conditioned prediction

For every candidate action

$$
a_j\in\mathcal A
$$

and hypothesis

$$
H_k,
$$

predict:

$$
\hat F_{t+1}^{k,j}
=
W_\theta(F_t,H_k,a_j).
$$

Therefore if there are five hypotheses and eight candidate motions, you obtain a matrix of predicted consequences.

$$
\begin{array}{c|cccc}
 & a_1&a_2&\cdots&a_M\\
\hline
H_D&\hat F_D^1&\hat F_D^2&\cdots\\
H_R&\hat F_R^1&\hat F_R^2&\cdots\\
H_T&\hat F_T^1&\hat F_T^2&\cdots\\
H_E&\hat F_E^1&\hat F_E^2&\cdots
\end{array}
$$

This gives every hypothesis an **intervention response profile**.

I would avoid strongly claiming the phrase “intervention signature” itself as novel because active vision has used controlled camera motions for hypothesis verification for decades. A 1997 CVIU paper, for example, explicitly uses known camera motion and active-vision strategies to refine competing object hypotheses. ([ScienceDirect][5])

What is potentially new is that the hypotheses here correspond specifically to **competing optical explanations of the same apparent geometry**.

---

# 8. Stage D — Separability estimator

For every pair

$$
(H_i,H_j)
$$

evaluate:

$$
\Delta_{ij}(a)
=
D(
\hat F_{t+1}^{i,a},
\hat F_{t+1}^{j,a}
).
$$

Use several distances:

$$
D =
\lambda_gD_{\text{geometry}}
+
\lambda_fD_{\text{feature}}
+
\lambda_mD_{\text{motion}}
+
\lambda_oD_{\text{occlusion}}.
$$

The action yielding maximum separation is

$$
a^*
=
\arg\max_a
\mathbb E_{i,j}
[\Delta_{ij}(a)].
$$

A more principled version uses expected posterior entropy reduction:

$$
a^*
=
\arg\max_a
I(H;O'|a).
$$

Again, **information-gain next-best-view is not novel**. Reflective-object active stereo already predicts candidate viewpoints based on expected information gain. ([arXiv][6])

Your contribution is what the information gain is *about*:

$$
\boxed{\text{causal optical explanation}}
$$

rather than depth completion or object pose.

---

# 9. Stage E — Observe and update belief

Execute:

$$
a^*.
$$

Obtain:

$$
I_{t+1}.
$$

Encode:

$$
F_{t+1}=E(I_{t+1}).
$$

Evaluate each prediction:

$$
e_k
=
D(
F_{t+1},
\hat F_{t+1}^{k,a^*}
).
$$

Update:

$$
p_{t+1}(H_k)
\propto
e^{-\beta e_k}
p_t(H_k).
$$

Repeat if necessary.

Now the model behaves like an experimenter:

$$
\text{hypothesis}
\rightarrow
\text{experiment}
\rightarrow
\text{predicted consequence}
\rightarrow
\text{observation}
\rightarrow
\text{belief update}.
$$

This is a much stronger story than simply “temporal depth.”

---

# 10. The critical abstention mechanism

This deserves prominence.

Suppose after all allowed interventions:

$$
P(H_D)=0.49
$$

and

$$
P(H_E)=0.47.
$$

Most classification systems would still output:

$$
H_D.
$$

Your system should return:

$$
\boxed{
\text{physical explanation unresolved under available visual evidence}
}
$$

when:

$$
\mathcal I_{\mathcal A}<\epsilon.
$$

This is different from generic uncertainty.

You should explicitly distinguish:

$$
U_{\text{prediction}}
$$

from

$$
U_{\text{identifiability}}.
$$

A model can be certain that an apparent corridor has depth structure while being fundamentally unable to decide whether that structure is a real corridor or a view-conditioned display.

That distinction is one of the strongest conceptual pieces of the paper.

---

# 11. Add one theoretical result

This could materially improve the CVPR paper.

You do not need a huge theoretical section.

Start with the easiest case:

## Static planar display vs real 3D scene

Suppose the observer translates laterally by baseline \(B\).

For a real point at depth \(Z\),

$$
\Delta u
\approx
\frac{fB}{Z}.
$$

If two scene points have depths

$$
Z_1\neq Z_2,
$$

their differential parallax is approximately

$$
|\Delta u_1-\Delta u_2|
=
fB
\left|
\frac1{Z_1}-\frac1{Z_2}
\right|.
$$

For a planar static display, the content remains constrained to the same physical plane after compensating for the display-plane homography.

Therefore if your perceptual system requires displacement difference at least \(\delta\),

$$
fB
\left|
\frac1{Z_1}-\frac1{Z_2}
\right|
>\delta.
$$

Hence:

$$
\boxed{
B_{\min}
>
\frac{
\delta
}{
f
\left|
\frac1{Z_1}-\frac1{Z_2}
\right|
}
}
$$

gives a simple theoretical resolving-baseline relationship.

This is valuable because your **Minimum Causal Resolving Baseline** is then grounded in projective geometry rather than being an arbitrary dataset metric.

---

# 12. Keep analytic optics where possible

I would make the world model hybrid.

For simple hypotheses, use known transformations.

### Direct geometry

$$
\Phi_D:
X'\!=RX+t.
$$

### Planar mirror

Use a virtual reflected camera:

$$
C_{\text{virt}}
=
Reflect(C,\Pi).
$$

### Static planar display

Use the screen-plane homography.

### Complex transparent/reflective/mixed scenes

Learn:

$$
\Phi_\theta
$$

as a residual around the analytical model.

So:

$$
\boxed{
\text{known projective optics}
+
\text{learned residual world model}
}
$$

rather than asking a transformer to rediscover basic physics.

That should improve generalization and reviewer confidence.

---

# 13. The benchmark is the centerpiece

Working name:

## **Intervene3D-Bench**

Do **not** build a dataset that merely contains:

* mirrors,
* glass,
* televisions,
* posters.

That would be another recognition benchmark.

Instead build **matched physical counterfactuals**.

Our previous iteration identified matched observations and explicit non-identifiable cases as critical. 

---

# 14. Matched scene design

For every underlying semantic scene \(S\), construct variants.

### Direct

A genuine 3D scene.

### Display

A high-resolution image/video of \(S\) shown on a display.

### Mirror

A reflected version of \(S\).

### Transmission

\(S\) viewed behind a transparent plane/object.

### Mixed

Reflection + transmission.

At reference viewpoint:

$$
C_0,
$$

optimize/capture them so that:

$$
D_{\text{perceptual}}
(
I_i(C_0),
I_j(C_0)
)
$$

is small.

Then capture systematic interventions:

$$
C_1,\ldots,C_N.
$$

Now you know how quickly each ambiguity separates.

---

# 15. Build an ambiguity ladder

The dataset should deliberately contain increasing difficulty.

### Display

$$
\text{poster}
\rightarrow
\text{static monitor}
\rightarrow
\text{perspective-correct display}
\rightarrow
\text{head-tracked display}.
$$

### Reflection

$$
\text{flat mirror}
\rightarrow
\text{curved mirror}
\rightarrow
\text{partial reflection}
\rightarrow
\text{reflection + transmission}.
$$

### Transparency

$$
\text{flat glass}
\rightarrow
\text{multi-pane glass}
\rightarrow
\text{curved transparent object}
\rightarrow
\text{mixed refraction/reflection}.
$$

Difficulty should correspond to increasing:

$$
MCRB.
$$

Eventually some cases should become:

$$
\boxed{
\text{unresolvable within }\mathcal A.
}
$$

That last category is extremely valuable.

---

# 16. Real-world ground truth

This is probably the hardest practical issue.

Depth sensors are unreliable exactly where your task matters.

GIFT explicitly notes this difficulty for non-Lambertian surfaces. ([arXiv][3])

I would therefore use **controlled geometry capture**.

For mirror/glass planes:

* precisely calibrate the physical plane;
* optionally capture the surface with removable diffuse coating / matte proxy;
* capture geometry while the optical ambiguity is disabled;
* return the object to reflective/transparent state without changing pose.

For displays:

* physical screen plane is directly calibrated.

For real 3D scenes:

* conventional LiDAR/structured-light/multi-view reconstruction.

This provides:

$$
D_{\text{contact}}
$$

without requiring the depth sensor to see through the optical effect.

---

# 17. Synthetic component

Synthetic data is unavoidable because it gives perfect light-path labels.

Use Blender/Cycles or Mitsuba to output:

$$
RGB
$$

$$
D_{\text{contact}}
$$

$$
D_{\text{visible/apparent}}
$$

$$
H
$$

$$
\text{surface normals}
$$

$$
\text{material class}
$$

$$
C_t
$$

$$
\text{optical interaction mask}.
$$

Existing work gives useful starting points: LayeredDepth already has a procedural transparent-scene generator, while TransPhy3D contains 11k transparent/reflective video sequences with depth and normal supervision. ([Open Access CVF][1])

Your data contribution must therefore add the **matched causal ambiguity + intervention + resolvability structure**, not merely more transparent renderings.

---

# 18. Proposed dataset scale

For a realistic first CVPR paper, I would target roughly:

**Synthetic:** 10k–30k base scenes, multiple causal variants per scene, 8–20 camera interventions per scene.

**Real:** approximately 300–800 high-quality controlled sequences.

Quality matters more than sheer scale because the benchmark's contribution is controlled causal pairing.

A particularly valuable real subset would include a **camera-tracked display** that updates its rendered perspective based on observer motion.

That gives you experimentally difficult or nearly non-identifiable display cases rather than only easy posters and monitors.

---

# 19. Main metrics

Don't overload the paper with ten new metrics.

Use five.

### A. Causal Explanation Accuracy

$$
CEA
=
P(\hat H=H^*).
$$

Report by mechanism.

---

### B. Physical Contact Depth Error

Standard:

$$
AbsRel_{\text{contact}},
RMSE_{\text{contact}}.
$$

This determines whether the model recovered physically usable geometry.

---

### C. Identifiability AUROC

Binary target:

$$
y_{\text{id}}\in\{0,1\}.
$$

Does the model correctly predict whether the ambiguity is resolvable under \(\mathcal A\)?

This should be a headline metric.

---

### D. Minimum Causal Resolving Baseline

Ground truth:

$$
B_{\min}^{GT}.
$$

Prediction:

$$
\hat B_{\min}.
$$

Evaluate:

$$
MAE_{\text{MCRB}}
=
|\hat B_{\min}-B_{\min}^{GT}|.
$$

---

### E. False Physical Certainty Rate

For non-identifiable samples:

$$
FPCR
=
P(
\max_kp(H_k)>\tau
|
y_{\text{id}}=0
).
$$

This is one of the most compelling metrics.

A good system should **not confidently hallucinate physical certainty**.

---

# 20. Optional metric: intervention regret

For chosen action \(\hat a\):

$$
Regret
=
\Delta(a^*)-\Delta(\hat a).
$$

This measures whether the model selected a near-optimal visual experiment.

Useful, but secondary.

---

# 21. Baselines you need

A strong paper needs baselines from several families.

| Family                             | Relevant work                                  |
| ---------------------------------- | ---------------------------------------------- |
| Conventional monocular geometry    | strong depth foundation models                 |
| Physical-surface correction        | **GIFT**                                       |
| Multilayer depth                   | **LayeredDepth / SeeGroup**                    |
| Controllable multilayer perception | **DepthFocus**                                 |
| Transparent video depth            | **DKT / TransPhy3D**                           |
| Visual-illusion robustness         | **3D Visual Illusion Depth Estimation**        |
| Mirror motion                      | CVPR 2024 inconsistent-motion mirror detection |
| Multi-view mirror detection        | **MVMD**                                       |
| 3D-motion mirror detection         | **MiD-VMD, AAAI 2026**                         |
| Geometry world models              | **3WM / P3Sim / VGGT-World-style predictor**   |
| Active reflective perception       | reflective-object next-best-view               |

DepthFocus already provides steerable layer selection in see-through scenes, DKT gives temporally coherent depth for transparent/reflective video, and NeurIPS 2025 already shows that monocular, stereo and multi-view depth systems can all be fooled by visual illusions. ([Junhong Min][7])

That means your benchmark must expose a question they do not answer.

---

# 22. Most important baseline: a static classifier

This is crucial.

Train:

$$
I_t
\rightarrow H.
$$

If a large foundation vision model can identify all cases from appearance alone, your whole interventional story becomes weaker.

Therefore construct matched examples where single-view appearance classification genuinely fails.

Your first experimental result should ideally be:

$$
Accuracy_{\text{single-frame}}\approx\text{poor}
$$

while

$$
Accuracy_{\text{interventional}}\gg Accuracy_{\text{single-frame}}.
$$

That validates the task itself.

---

# 23. Ablations

The most important ones are:

| Ablation                                    | Purpose                                              |
| ------------------------------------------- | ---------------------------------------------------- |
| Single image only                           | Is intervention necessary?                           |
| Random camera movement                      | Is planned intervention useful?                      |
| Maximum-baseline movement                   | Is information-aware intervention useful?            |
| Generic uncertainty NBV                     | Does causal hypothesis conditioning matter?          |
| No hypothesis conditioning                  | Is the world model simply forecasting geometry?      |
| RGB prediction instead of geometry features | Does geometry-space modeling help?                   |
| Learned-only world model                    | Do analytic optical priors help?                     |
| Analytic-only                               | Does learned residual modeling matter?               |
| No unresolvable state                       | Does forced classification increase false certainty? |
| No matched-counterfactual training          | Does the dataset design matter?                      |
| Noisy camera action                         | Robustness to real execution/pose error              |

These are much more meaningful than architectural micro-ablations such as “4 vs 6 transformer blocks.”

---

# 24. Generalization experiments

For CVPR impact, don't stop at IID performance.

Test:

### unseen materials

Train:

* common mirror/glass.

Test:

* tinted glass,
* curved mirrors,
* polished metal.

### unseen scene categories

Indoor → laboratory/office/outdoor storefront.

### unseen optical compositions

Train:

$$
D,R,T,E
$$

individually.

Test:

$$
R+T.
$$

### action noise

Inject:

$$
\Delta C+\epsilon.
$$

### unseen cameras

Different:

* focal lengths;
* sensors;
* resolutions.

Those results determine whether this is a research method or just a controlled benchmark solution.

---

# 25. What is definitely **not novel**

This section should almost mirror your future related-work section.

### Multilayer depth — not novel

LayeredDepth introduced the task/dataset, SeeGroup represents multiple depths as unordered point-process events, and One Scene, Two Depths directly studies alternative valid geometric interpretations. ([Open Access CVF][1])

### Selecting different depth layers — not novel

DepthFocus already provides steerable depth estimation in transparent and reflective scenes. ([Junhong Min][7])

### Recovering true physical mirror/glass surface — not novel

GIFT directly targets reflected/transmitted hallucinations and recovers the underlying physical surface. ([arXiv][3])

### Explicit reflection/transmission decomposition — not novel

GLINT reconstructs the transparent interface while explicitly separating transmitted and reflected radiance. It was a CVPR 2026 Oral/Award Candidate, so reviewers in this area are likely to know it. ([Youngju Na][8])

### Motion as evidence for mirrors — not novel

CVPR 2024 exploits inconsistent motion; WACV 2025 MVMD exploits viewpoint-dependent reflection changes; AAAI 2026 uses RGB, depth and 3D motion for video mirror detection. ([CityU Computer Science][9])

### World models for geometric transformations — not novel

3WM already unifies RGB, flow, camera and depth through probabilistic world modeling. P3Sim predicts future scene states under partial 3D transformations with geometric conditioning and persistent memory. ([ICLR Proceedings][10])

### Counterfactual world modeling — not novel

CWM already uses counterfactual prediction for visual representations, and more recent work explicitly formalizes intervention-conditioned world models. ([arXiv][11])

### Active hypothesis verification — definitely not novel

Active vision has long used controlled camera movement to refine competing hypotheses; even the phrase “hypothesis verification” appears in 1990s active-vision work. ([ScienceDirect][5])

### Causal identifiability from interventions — not novel generally

CITRIS explicitly studies identifiability of causal factors from temporally intervened visual sequences. ([Proceedings of Machine Learning Research][12])

This is why the wording of the paper matters enormously.

---

# 26. What I believe is still plausibly novel

Based on the targeted literature search through **August 27, 2026**, I did **not** find a paper whose central task jointly asks:

$$
\boxed{
\begin{aligned}
&\text{Given competing optical explanations of apparent geometry,}\\
&\text{are they distinguishable under a specified set of camera actions?}\\
&\text{If yes, which action separates them?}\\
&\text{If no, can the system explicitly recognize non-identifiability?}\\
&\text{What physical contact geometry remains justified?}
\end{aligned}
}
$$

That is the defensible gap.

Notice how narrow it is.

You are **not** introducing:

* causality;
* identifiability;
* active vision;
* light transport;
* world models;
* depth ambiguity.

You are introducing a specific **vision task formed at their intersection**.

---

# 27. The strongest novelty claim

I would eventually write something close to:

> **We formulate physical-geometry inference under optical ambiguity as an action-set-dependent identifiability problem. Rather than forcing a unique 3D interpretation, the proposed framework maintains competing image-formation hypotheses and predicts whether controlled observer interventions can distinguish them.**

Second sentence:

> **This enables a system to distinguish uncertainty caused by insufficient prediction quality from ambiguity that is fundamentally unresolved by the available visual experiments.**

That second sentence is particularly powerful.

---

# 28. The benchmark novelty may be stronger than the model novelty

This is worth emphasizing.

Architecturally:

$$
GFM
+
conditional\ transformer
+
belief\ update
+
active\ view
$$

is incremental.

I would rate architecture novelty maybe:

$$
5/10.
$$

But:

$$
\text{new task}
+
\text{formal identifiability criterion}
+
\text{matched counterfactual benchmark}
+
\text{unresolvable cases}
$$

could be much more distinctive.

That is okay.

CVPR papers do not need a brand-new neural operation if the paper establishes an important new problem and demonstrates it rigorously.

Our prior review already reached this conclusion: novelty should be claimed at the **problem-definition level**, not for generic components. 

---

# 29. Biggest novelty risk

The biggest reviewer attack will be:

> “This is just active perception / next-best-view applied to mirror and glass classification.”

You must prevent that interpretation.

The response is not:

> “But ours uses a world model.”

That's weak.

The response is:

> **Existing NBV assumes there is a state to estimate and asks where to observe it better. We ask first whether competing image-formation explanations are distinguishable under the available sensing actions at all.**

Mathematically:

traditional NBV:

$$
\arg\max_a
I(X;O'|a).
$$

Your task:

$$
\boxed{
\text{first determine whether }
\sup_{a\in\mathcal A}
D[
p(O'|H_i,a),
p(O'|H_j,a)
]
>\epsilon.
}
$$

If not:

$$
\text{abstain}.
$$

That's the difference you need to defend.

---

# 30. Second major novelty risk

Reviewer:

> “GIFT already distinguishes physical geometry from reflections/transmission.”

Correct.

Your answer:

GIFT essentially solves:

$$
I\rightarrow D_{\text{physical}}
$$

through geometry-invariant fine-tuning. ([arXiv][3])

You solve:

$$
(I,\mathcal A)
\rightarrow
\left[
p(H),
\mathcal I_{\mathcal A},
a^*,
D_{\text{contact}}
\right].
$$

You're asking whether the physical interpretation can actually be established from the available visual evidence rather than assuming one should always be output.

---

# 31. Third major risk

Reviewer:

> “GLINT already models interface/reflection/transmission.”

Also correct.

GLINT's goal is physically grounded transparent-scene reconstruction by explicitly decomposing interface, transmitted and reflected radiance. ([arXiv][13])

Your problem does not assume that one optical decomposition is already correct.

Instead:

$$
\mathcal G_1,\ldots,\mathcal G_K
$$

remain competing explanations.

The intervention is used to determine **which explanation is supported and whether the observations are sufficient to decide**.

That's different.

---

# 32. Fourth risk: misuse of “causal”

This could hurt the paper badly.

Do not use “causal” merely because frames happen sequentially.

Your intervention must be explicit.

For instance:

$$
do(C_{t+1}=C')
$$

means the observer/camera is moved to a controlled pose.

The hypotheses correspond to explicit image-formation mechanisms.

That makes the causal interpretation meaningful.

If you cannot formally specify the intervention, I would actually use the safer term:

## **Interventional Identifiability**

instead of:

## Causal Geometry.

I think **Interventional Identifiability** is the better title language.

---

# 33. Final title recommendation

My strongest title now is:

# **Intervene3D: Interventional Identifiability of Physical Geometry under Optical Ambiguity**

Alternative:

**Intervene3D: When Is Apparent 3D Geometry Physically Identifiable?**

The second is more memorable.

---

# 34. Final research questions

### RQ1

> Can competing physical explanations of visually ambiguous geometry be distinguished by their predicted responses to controlled observer motion?

### RQ2

> Can a vision system predict when no available observer intervention is sufficient to identify the physical explanation?

### RQ3

> Can an intervention-conditioned geometry world model select camera actions that resolve optical ambiguity with less motion than passive or generic next-best-view strategies?

### RQ4

> Does identifiability-aware inference reduce confidently incorrect physical geometry predictions on mirrors, transparent surfaces, displays and visual illusions?

---

# 35. Core hypotheses

$$
\mathbf{H1:}
$$

Hypothesis-conditioned intervention prediction will classify physical image-formation mechanisms more accurately than static or passive-video models on appearance-matched ambiguities.

$$
\mathbf{H2:}
$$

Modeling **resolvability explicitly** will significantly reduce false physical certainty compared with ordinary uncertainty calibration.

$$
\mathbf{H3:}
$$

Choosing interventions using causal-hypothesis separability will require less camera movement than random, maximum-baseline or generic entropy-based NBV.

$$
\mathbf{H4:}
$$

Hybrid analytical + learned optical transition models will generalize better to unseen materials and camera trajectories than fully learned intervention predictors.

---

# 36. Development plan

I would build this incrementally.

### Phase 1 — Prove the problem exists

Create only:

* direct 3D scene;
* flat display;
* flat mirror.

Controlled synthetic trajectories.

Show that:

$$
\text{single-view classifier}
<
\text{multi-view passive}
<
\text{intervention-aware}.
$$

Derive the display-vs-real resolving-baseline result.

If this fails, stop before building a huge system.

### Phase 2 — Build the benchmark core

Add:

* glass;
* display;
* mirror;
* direct geometry;
* non-identifiable synthetic cases.

Establish MCRB and FPCR metrics.

### Phase 3 — Geometry world model

Frozen GFM + hypothesis-conditioned transition model.

Show that predicted geometry features distinguish causal mechanisms.

### Phase 4 — Active intervention

Add:

$$
a^*.
$$

Compare against generic NBV.

### Phase 5 — Real-world validation

Capture high-quality controlled sequences.

Test cross-domain generalization.

### Phase 6 — Mixed optics

Only after everything else works:

$$
reflection+transmission
$$

and curved/complex surfaces.

---

# 37. What I would cut for CVPR 2027

Do **not** include:

* diffusion depth;
* generic camera calibration;
* VLM reasoning;
* SLAM;
* navigation policy learning;
* full 3D reconstruction;
* semantic object recognition;
* generative RGB video;
* arbitrary object actions.

The previous versions were beginning to accumulate too many modules; we already identified that as a weakness. 

Assume controlled camera motion is known.

That is perfectly acceptable for the first paper.

You can later relax it.

---

# 38. CVPR 2027 feasibility

There is one practical issue: it is already **August 27, 2026**.

The official source I found confirms CVPR 2027 will be held in Seattle in June 2027, but I did not find official 2027 paper-submission dates yet. CVPR 2025 and 2026 paper deadlines were both in November of the preceding year, so if the schedule remains similar, the available research window is short. ([CVPR 2026][14])

Therefore, for a **CVPR 2027 submission**, I would not attempt the full grand vision immediately.

The minimum viable strong paper should be:

$$
\boxed{
\text{new identifiability formulation}
+
\text{small theory}
+
\text{strong matched benchmark}
+
\text{simple intervention-conditioned model}
}
$$

rather than a huge foundation world model.

A highly polished benchmark plus a conceptually clear method would be more convincing than six underdeveloped modules.

---

# 39. Final novelty verdict

My current assessment after all of the searches is:

| Element                                                              |                                  Novelty |
| -------------------------------------------------------------------- | ---------------------------------------: |
| Physical vs apparent geometry                                        |                                      Low |
| Mirror/glass physical depth                                          |                                      Low |
| Multilayer depth                                                     |                                      Low |
| Reflection/transmission decomposition                                |                                      Low |
| Camera motion as disambiguation cue                                  |                                      Low |
| Active next-best-view                                                |                                      Low |
| Counterfactual world models                                          |                                      Low |
| Causal identifiability generally                                     |                                      Low |
| Optical-hypothesis-conditioned future prediction                     |                              Medium–High |
| Action-set-dependent **physical geometry identifiability**           |                                 **High** |
| Predicting **non-identifiability** rather than forced classification |                                 **High** |
| MCRB for competing optical geometric explanations                    |                              Medium–High |
| Matched counterfactual optical benchmark                             |                                 **High** |
| Explicit non-identifiable benchmark cases                            |                       **Very promising** |
| Full task + method + benchmark together                              | **Strong CVPR-level research direction** |

The caveat matters: this is a targeted literature audit through **August 27, 2026**, not proof that no unpublished, obscure, or upcoming CVPR 2027 work overlaps.

---

## The version I would now commit to

Don't pitch:

> **“We use a world model to fix depth estimation in mirrors and glass.”**

Pitch:

> **Computer vision normally assumes that enough visual evidence will eventually reveal the correct 3D interpretation. We challenge that assumption. Intervene3D studies whether the physical cause of apparent geometry is identifiable from a bounded set of observer actions. The model maintains competing optical explanations, predicts how each should respond to controlled camera interventions, selects experiments that separate them, and explicitly abstains when the available visual evidence cannot establish physicality.**

That is now a **coherent research problem**, rather than a bundle of trendy components. Of everything we explored, this is the version I would take forward toward a serious CVPR submission.

Because related work in this area is moving unusually fast, I can also monitor new depth/world-model/active-vision papers for anything that threatens this novelty position.

[1]: https://openaccess.thecvf.com/content/ICCV2025/html/Wen_Seeing_and_Seeing_Through_the_Glass_Real_and_Synthetic_Data_ICCV_2025_paper.html?utm_source=chatgpt.com "ICCV 2025 Open Access Repository"
[2]: https://arxiv.org/abs/2606.29600?utm_source=chatgpt.com "One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models"
[3]: https://arxiv.org/abs/2608.02068?utm_source=chatgpt.com "GIFT: Geometry-Invariant Fine-Tuning for Non-Lambertian Monocular Depth Estimation"
[4]: https://arxiv.org/abs/2603.12655?utm_source=chatgpt.com "VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model"
[5]: https://www.sciencedirect.com/science/article/pii/S1077314297905543?utm_source=chatgpt.com "Hypothesis Verification Using Parametric Models and Active Vision Strategies - ScienceDirect"
[6]: https://arxiv.org/abs/2202.13263?utm_source=chatgpt.com "Next-Best-View Prediction for Active Stereo Cameras and Highly Reflective Objects"
[7]: https://junhong-3dv.github.io/depthfocus-project/?utm_source=chatgpt.com "DepthFocus - CVPR 2026"
[8]: https://youngju-na.github.io/GLINT/?utm_source=chatgpt.com "GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport"
[9]: https://www.cs.cityu.edu.hk/~rynson/papers/cvpr24d.pdf?utm_source=chatgpt.com "This CVPR paper is the Open Access version, provided by the Computer Vision Foundation."
[10]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3e6c9a8f3863f0343bf5cbabf1f85ca6-Abstract-Conference.html?utm_source=chatgpt.com "Unified 3D Scene Understanding Through Physical World Modeling"
[11]: https://arxiv.org/abs/2306.01828?utm_source=chatgpt.com "Unifying (Machine) Vision via Counterfactual World Modeling"
[12]: https://proceedings.mlr.press/v162/lippe22a.html?utm_source=chatgpt.com "CITRIS: Causal Identifiability from Temporal Intervened Sequences"
[13]: https://arxiv.org/abs/2603.26181?utm_source=chatgpt.com "GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport"
[14]: https://cvpr.thecvf.com/Conferences/2026/News/Closing?utm_source=chatgpt.com "CVPR 2026 Shatters Records, Revealing Latest Breakthroughs in Computer Vision and AI"

Yes. Below is the consolidated literature and dataset map I would actually use for **Intervene3D**. I have separated the papers that define or threaten our novelty from papers that are useful mainly as technical foundations.

The important conclusion from our review remains: **the paper should not claim novelty for multilayer depth, mirror/glass correction, world models, counterfactual prediction, or active next-best-view. The novelty should sit in action-set-dependent identifiability of competing optical explanations and explicit prediction of non-identifiability.** 

# A. Core literature — these must be in the related-work review

| #  | Paper                                                                                                            | Venue / year | Why it matters for Intervene3D                                                                                                                                                                       |
| -- | ---------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation (LayeredDepth)** | ICCV 2025    | Establishes multi-layer depth as a task and introduces LayeredDepth: 1,500 real images + 15,300 synthetic images. We therefore cannot claim multiple depths per ray as novel. ([Open Access CVF][1]) |
| 2  | **SeeGroup: Multi-Layer Depth Estimation of Transparent Surfaces via Self-Determined Grouping**                  | CVPR 2026    | Strong modern multi-layer-depth baseline; avoids forcing transparent geometry into a single depth. Direct competitor to any ray-hypothesis decoder. ([Open Access CVF][2])                           |
| 3  | **One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models**                            | 2026         | Explicitly demonstrates that different valid depth hypotheses can coexist. Introduces **MultiDepth-3k / MD-3k**. ([arXiv][3])                                                                        |
| 4  | **GIFT: Geometry-Invariant Fine-Tuning for Non-Lambertian Monocular Depth Estimation**                           | 2026         | Very important novelty threat. It explicitly tries to recover the true physical mirror/glass surface rather than reflected/transmitted apparent geometry. ([arXiv][4])                               |
| 5  | **3D Visual Illusion Depth Estimation**                                                                          | NeurIPS 2025 | Shows monocular, binocular and multi-view depth can be fooled by pictures, screens, mirrors and other 3D illusions. Dataset has almost 3,000 scenes / 200k images. ([NeurIPS Proceedings][5])        |
| 6  | **Mirror3D: Depth Refinement for Mirror Surfaces**                                                               | CVPR 2021    | Establishes correction of physical mirror depth; includes 7,011 mirror-plane annotations in 5,894 RGB-D frames. ([arXiv][6])                                                                         |
| 7  | **Depth-Aware Mirror Segmentation**                                                                              | CVPR 2021    | Explicitly discusses **apparent reflected depth vs true mirror depth**. Important precedent against claiming “physical ≠ apparent” as new. ([Open Access CVF][7])                                    |
| 8  | **GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport**                                     | CVPR 2026    | Decomposes transparent scenes into interface, transmission and reflection. Therefore optical component decomposition alone is not novel. ([arXiv][8])                                                |
| 9  | **Mirror-NeRF: Learning Neural Radiance Fields for Mirrors with Whitted-Style Ray Tracing**                      | 2023         | Models mirror reflection using explicit physical ray transport/reflection probabilities. Important optical-path precedent.                                                                           |
| 10 | **Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization**                    | 2024         | Treats reflections using virtual reflected cameras. Relevant to analytical mirror intervention signatures. ([arXiv][9])                                                                              |
| 11 | **Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation** | 2025/26      | Strong transparent/reflective video-depth baseline; introduces **TransPhy3D**, 11k video sequences. ([arXiv][10])                                                                                    |
| 12 | **DepthFocus: Controllable Depth Estimation for See-Through Scenes**                                             | CVPR 2026    | Allows selective estimation of different see-through depth layers; trained primarily on ~500k synthetic multilayer stereo pairs. ([arXiv][11])                                                       |

These twelve define the **physical/apparent/multilayer ambiguity** side of the literature.

---

# B. Motion, mirror-video and active-disambiguation literature

These papers are particularly important because they prevent us from claiming that “moving the camera to identify mirrors” is new.

| #  | Paper                                                                                                                | Venue / year                         | Relevance                                                                                                                                                                   |
| -- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13 | **Effective Video Mirror Detection with Inconsistent Motion Cues**                                                   | CVPR 2024                            | Explicitly uses inconsistent reflected motion to detect mirrors. ([CityU Computer Science][12])                                                                             |
| 14 | **MVMD: A Multi-View Approach for Enhanced Mirror Detection**                                                        | WACV 2025                            | Exploits viewpoint-dependent shifts of reflected objects across multiple views; also introduces a multi-view mirror dataset. ([Open Access CVF][13])                        |
| 15 | **Video Mirror Detection with the Motion-in-Depth Cue**                                                              | AAAI 2026                            | Combines RGB, depth and 3D motion to identify mirror regions. Very close to motion-based physicality reasoning. ([AAAI Publications][14])                                   |
| 16 | **Key Points Trajectory and Multi-Level Depth Distinction Based Refinement for Video Mirror and Glass Segmentation** | Multimedia Tools & Applications 2024 | Uses trajectory inconsistencies and depth cues for mirror/glass video segmentation; introduces **MAGD**, 36 videos / 9,960 frames. ([Springer][15])                         |
| 17 | **Next-Best-View Prediction for Active Stereo Cameras and Highly Reflective Objects**                                | 2022                                 | Explicitly builds depth/reflection hypotheses, computes information gain for candidate camera poses and chooses the next best view. Crucial novelty boundary. ([arXiv][16]) |
| 18 | **Hypothesis Verification Using Parametric Models and Active Vision Strategies**                                     | classic active vision                | Important conceptual predecessor: camera actions as experiments for verifying competing hypotheses.                                                                         |

Therefore we should **not** claim:

$$
a^*=\arg\max I(H;O'|a)
$$

as a new active-vision principle.

Our novelty is that \(H\) represents **competing causal optical explanations of the same apparent geometry**, and we explicitly model whether they are distinguishable at all.

---

# C. World-model literature

These are the papers a CVPR reviewer would most likely invoke if we overclaim the world-model contribution.

| #  | Paper                                                                                                       | Venue/year | Relevance                                                                                                                                                       |
| -- | ----------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 19 | **Unifying (Machine) Vision via Counterfactual World Modeling**                                             | 2023       | Introduces counterfactual visual world modelling and extracts depth, flow, segmentation, occlusion, etc. Counterfactual vision itself is not new. ([arXiv][17]) |
| 20 | **Understanding Physical Dynamics with Counterfactual World Modeling**                                      | ECCV 2024  | Extends CWM to physical-dynamics understanding. ([ECVA][18])                                                                                                    |
| 21 | **Unified 3D Scene Understanding Through Physical World Modeling (3WM)**                                    | ICLR 2026  | Probabilistic 3D world model spanning RGB, optical flow, camera pose, depth, manipulation and NVS. ([ICLR Proceedings][19])                                     |
| 22 | **Perceptual 3D Simulation With Physical World Modeling (P3Sim)**                                           | CVPR 2026  | Extremely important competitor: world model + partial observations + 3D transformations + geometric conditioning + persistent memory. ([Open Access CVF][20])   |
| 23 | **Physical Object Understanding with a Physically Controllable World Model**                                | CVPR 2026  | Uses probabilistic world models and physically controlled interventions to understand objects and their interactions. ([Open Access CVF][21])                   |
| 24 | **VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model**                               | 2026       | Particularly useful architecture precedent: predicts **geometry-foundation-model features instead of future RGB**. ([arXiv][22])                                |
| 25 | **WorldReel: 4D Video Generation with Consistent Geometry and Motion Modeling**                             | CVPR 2026  | Joint RGB + pointmaps + camera trajectories + dense motion. Relevant to persistent geometry-world states. ([Open Access CVF][23])                               |
| 26 | **WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories** | CVPR 2026  | Camera-conditioned generation with persistent geometric memory. ([Open Access CVF][24])                                                                         |
| 27 | **NeoVerse: Enhancing 4D World Model with In-the-Wild Monocular Videos**                                    | CVPR 2026  | Pose-free 4D reconstruction + novel-trajectory generation from monocular videos. ([Open Access CVF][25])                                                        |
| 28 | **PhysMind: From Video to Executable Worlds for Training-Free Physical Reasoning**                          | 2026       | Builds executable scene models and evaluates actual/counterfactual physical outcomes. ([arXiv][26])                                                             |
| 29 | **GeoWAM: Visual Geometry World Action Models for Autonomous Driving**                                      | 2026       | Argues for predicting future **geometry** rather than future RGB. Supports our choice of geometry-state world modeling. ([arXiv][27])                           |

For Intervene3D, I would explicitly say:

$$
\boxed{\text{the world model is an enabling mechanism, not our main novelty}}
$$

Its unique role is to estimate

$$
p(F_{t+1}|F_t,H_k,a)
$$

for **different optical hypotheses \(H_k\)**.

---

# D. Causality and identifiability foundations

| #  | Paper                                                                                      | Why we need it                                                                                                                                                                                         |
| -- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 30 | **CITRIS: Causal Identifiability from Temporal Intervened Sequences**                      | Demonstrates that causal identifiability under temporal interventions is an established theoretical idea. We must therefore specialize the novelty to optical/geometric identifiability. ([arXiv][28]) |
| 31 | **Observability/Identifiability of Rigid Motion under Perspective Projection**             | Classical geometric precedent for identifiability/observability under camera motion.                                                                                                                   |
| 32 | **Unveiling the Ambiguity in Neural Inverse Rendering: A Parameter Compensation Analysis** | Demonstrates intrinsic ambiguity in inverse rendering where different physical parameters produce similar observations. ([Open Access CVF][29])                                                        |
| 33 | **Dynamic Inverse Rendering for Enhanced Material-Lighting Decomposition**                 | Shows motion can help resolve otherwise ill-posed material/lighting ambiguity. Very relevant conceptual precedent. ([arXiv][30])                                                                       |

These provide the theoretical lineage:

$$
\text{inverse-problem ambiguity}
\rightarrow
\text{intervention}
\rightarrow
\text{identifiability}.
$$

Our specific contribution becomes:

$$
\boxed{\text{identifiability of physical image-formation explanations}}
$$

rather than causal identifiability generally.

---

# E. Video depth literature we investigated

These are important baselines but are not the novelty-defining papers anymore.

| #  | Paper                                                                                    | Venue/year | Role                                                                                                                               |
| -- | ---------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 34 | **Video Depth Anything: Consistent Depth Estimation for Super-Long Videos**              | CVPR 2025  | Strong discriminative temporally consistent video-depth baseline; supports arbitrary long video inference. ([Open Access CVF][31]) |
| 35 | **DepthCrafter: Generating Consistent Long Depth Sequences for Open-World Videos**       | CVPR 2025  | Diffusion-based temporal depth baseline. ([Open Access CVF][32])                                                                   |
| 36 | **Learning Temporally Consistent Video Depth from Video Diffusion Priors (ChronoDepth)** | CVPR 2025  | Another strong video-diffusion temporal-depth baseline. ([Open Access CVF][33])                                                    |
| 37 | **Video Depth without Video Models / RollingDepth**                                      | CVPR 2025  | Shows strong long-video consistency without a dedicated video foundation model. ([Open Access CVF][34])                            |
| 38 | **Align3R: Aligned Monocular Depth Estimation for Dynamic Videos**                       | CVPR 2025  | Produces temporally aligned depth and camera pose in dynamic videos. ([Open Access CVF][35])                                       |
| 39 | **Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors**                   | CVPR 2025  | Temporal depth/normal consistency using image priors, without paired video-depth training. ([Open Access CVF][36])                 |

These establish that:

$$
\text{temporal consistency}
$$

itself is definitely not sufficient novelty.

---

# F. Monocular geometry / camera-aware foundation models

| #  | Paper                                                                       | Role                                                                                                       |                                                                                                                                                                         |
| -- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 40 | **UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler**    | Strong camera-aware metric geometry baseline; also predicts uncertainty. ([arXiv][37])                     |                                                                                                                                                                         |
| 41 | **MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details** | Strong candidate foundation encoder for Intervene3D; predicts open-domain metric point maps. ([arXiv][38]) |                                                                                                                                                                         |
| 42 | **Cameras as Relative Positional Encoding (PRoPE)**                         | NeurIPS 2025                                                                                               | Camera-conditioned transformer representation incorporating full relative camera geometry. Useful architectural reference for action conditioning. ([ML Anthology][39]) |
| 43 | **Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers**      | NeurIPS 2025                                                                                               | We initially considered it for generative refinement; no longer central to the final proposal. ([NeurIPS][40])                                                          |

For the actual implementation I would start with **MoGe-2 or a VGGT-like representation**, not build a new depth backbone.

---

# G. Other NeurIPS 2025 papers we initially investigated

These informed the earlier brainstorming but are **not core related work for the final Intervene3D paper**.

| #  | Paper                                                                                         | Why it came up                                                                                                                  |
| -- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 44 | **SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models**     | Metric/spatial reasoning with depth-conditioned VLMs. ([NeurIPS Proceedings][41])                                               |
| 45 | **EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes**     | Robust geometry under unreliable RGB observations. ([NeurIPS Proceedings][42])                                                  |
| 46 | **RaySt3R: Predicting Novel Depth Maps for Zero-Shot Object Completion**                      | Ray-conditioned novel-view depth and uncertainty. ([NeurIPS Proceedings][43])                                                   |
| 47 | **PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding**               | Detailed/long-video understanding; originally considered during broad NeurIPS search. ([NeurIPS Proceedings][44])               |
| 48 | **Perceive Anything: Recognize, Explain, Caption, and Segment Anything in Images and Videos** | Streaming region-level video understanding; useful if semantic region proposals are later required. ([NeurIPS Proceedings][45]) |

I would **not spend much related-work space on 44–48** in the CVPR submission unless their components are actually used.

---

# H. Evaluation/world-model benchmark papers

| #  | Paper                                                                                    | Use                                                                                                      |
| -- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 49 | **PDI-Bench: Quantitative Video World Model Evaluation for Geometric-Consistency**       | Useful precedent for geometry-specific evaluation of world models; introduces PDI-Dataset. ([arXiv][46]) |
| 50 | **4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models** | Shows increasing emphasis on explicit physical/4D consistency evaluation. ([Open Access CVF][47])        |

So the consolidated literature set is around **50 papers**, of which roughly **25–30 deserve serious treatment in the actual paper**.

---

# Datasets we should actually use

Now the more important part.

There is **no single existing dataset sufficient for Intervene3D**.

We need a combination of existing benchmarks plus a new dataset.

## 1. LayeredDepth — MUST USE

**Source:** *Seeing and Seeing Through the Glass*, ICCV 2025.

Contains:

* 1,500 real-world images;
* 15,300 synthetic images;
* multi-layer depth relationships;
* roughly 14.2M relative-depth tuples in the real benchmark;
* household, restaurant, laboratory, urban and other scenes. ([Open Access CVF][1])

### Use

Evaluate whether our model preserves:

$$
D_{\text{interface}}
$$

and

$$
D_{\text{behind-glass}}
$$

rather than collapsing them.

### Priority

**★★★★★**

---

# 2. TransPhy3D — MUST USE

Introduced by **Diffusion Knows Transparency**.

Contains approximately:

$$
11,000\text{ sequences}
$$

and about

$$
1.32M\text{ frames}.
$$

Rendered using Blender/Cycles with:

* RGB video;
* metric depth;
* surface normals;
* camera calibration;
* transparent materials;
* reflective materials;
* glass/plastic/metal;
* static and procedural objects. ([arXiv][10])

### Use

Probably our best existing source for:

$$
\text{video}+
\text{optical ambiguity}+
\text{perfect geometry}.
$$

Useful for pretraining the hypothesis-conditioned transition/world model.

### Priority

**★★★★★**

---

# 3. 3D Visual Illusion Dataset — MUST USE

From NeurIPS 2025.

Approximately:

* 3,000 scenes;
* 200,000 images.

It includes several categories of image-induced geometry such as pictures/screens/mirrors and other 3D illusions. ([NeurIPS Proceedings][5])

### Use

This is essential for the:

$$
\text{apparent geometry}\neq\text{physical geometry}
$$

evaluation.

Particularly valuable for:

* displayed scenes;
* planar images;
* fake spatial structure;
* mirror-like illusions.

### Priority

**★★★★★**

---

# 4. Mirror3D — MUST USE

Contains:

* **5,894 RGB-D frames**;
* **7,011 mirror planes**;
* 457 scenes;
* mirror masks;
* mirror 3D-plane annotations.

Built on:

* Matterport3D;
* ScanNet;
* NYUv2. ([Open Access CVF][48])

### Use

Provides reliable:

$$
D_{\text{mirror/contact}}
$$

for evaluating physical mirror geometry.

### Priority

**★★★★★**

---

# 5. MAGD — highly useful

From the video mirror/glass segmentation paper.

Contains:

* 36 videos;
* 9,960 frames;
* pixel-level mirror/glass masks;
* indoor video sequences;
* 30 fps;
* sequences ranging roughly 130–600 frames. ([Springer][15])

### Use

Real temporal mirror/glass observations.

Useful for testing whether:

$$
H_t
$$

stays stable through actual camera motion.

### Priority

**★★★★☆**

---

# 6. MVMD dataset — highly useful

Introduced by WACV 2025's **MVMD** as a database specifically designed for **multi-view mirror detection**. ([Open Access CVF][49])

### Use

It is particularly relevant because our system also depends on:

$$
\text{viewpoint intervention}
\rightarrow
\text{changed reflection evidence}.
$$

### Priority

**★★★★☆**

---

# 7. MultiDepth-3k / MD-3k — MUST EVALUATE

From **One Scene, Two Depths**.

A sparse two-layer ordinal benchmark explicitly designed to evaluate ambiguity and layer preference. ([arXiv][3])

### Use

Determine whether our initial hypothesis generator preserves multiple plausible geometric explanations instead of prematurely collapsing to one.

### Priority

**★★★★☆**

It isn't sufficient for training dense metric geometry because its supervision is primarily sparse/ordinal.

---

# 8. DepthFocus synthetic multilayer dataset — useful if obtainable

DepthFocus constructs roughly:

$$
500,000
$$

synthetic stereo RGB pairs with:

* per-layer depth;
* per-layer disparity;
* transmissive-object masks;
* semantic masks;
* camera intrinsics;
* camera extrinsics. ([Open Access CVF][50])

### Use

Excellent pretraining source for optical layer representations.

### Priority

**★★★★☆**

---

# 9. ClearPose — strong real-world transparency benchmark

Contains:

* **350k+ real RGB-depth frames**;
* ~5M instance annotations;
* 63 transparent household objects;
* depth;
* normals;
* object poses;
* difficult conditions such as liquids, occlusions and translucent covers. ([arXiv][51])

### Use

Evaluate physical contact depth on real transparent objects.

### Priority

**★★★★☆**

---

# 10. DREDS + STD — strong auxiliary data

DREDS contains photorealistic synthetic scenes with transparent, specular and diffuse materials.

The official release reports:

* DREDS-CatKnown: 100,200 train + 19,380 test;
* DREDS-CatNovel: 11,520;
* real STD-CatKnown: 27,000;
* real STD-CatNovel: 11,000. ([GitHub][52])

### Use

Excellent for domain randomization and unseen-material generalization.

### Priority

**★★★★☆**

---

# 11. Booster — strong non-Lambertian benchmark

Designed specifically for:

* transparent surfaces;
* specular surfaces;
* high-resolution depth.

The later benchmark description contains 606 labeled samples from 85 scenes, plus 15k unlabeled samples; the original CVPR release contains a smaller subset. ([arXiv][53])

### Use

Independent high-quality test benchmark for:

$$
D_{\text{contact}}.
$$

### Priority

**★★★★☆**

---

# 12. RGB-D Mirror Segmentation dataset

From **Depth-Aware Mirror Segmentation**.

Contains:

$$
3,049
$$

RGB-D mirror exemplars. ([Open Access CVF][7])

### Use

Auxiliary mirror-region and physical/apparent-depth evaluation.

### Priority

**★★★☆☆**

---

# 13. GIFT controlled benchmark

GIFT introduces a controlled benchmark specifically for:

* mirror depth;
* transparent-surface depth;
* appearance changes while geometry remains fixed;
* preservation of normal-region depth performance. ([arXiv][4])

### Use

This is conceptually very close to what we need.

It would be a very valuable independent evaluation if released/accessible.

### Priority

**★★★★★ if available**

---

# 14. PDI-Dataset

From PDI-Bench.

Designed to stress:

* scale-depth consistency;
* 3D motion consistency;
* structural rigidity;
* world-model geometric coherence. ([arXiv][46])

### Use

Not necessary for the optical-ambiguity task itself, but useful to test whether our geometry world model actually predicts geometrically coherent futures.

### Priority

**★★☆☆☆**

---

# 15. Generic geometry/video datasets

For pretraining or sanity checks:

* KITTI;
* Cityscapes;
* TartanAir;
* ScanNet;
* NYUv2;
* Matterport3D;
* Bonn RGB-D;
* DyDToF.

For example VGGT-World evaluates geometry forecasting on KITTI, Cityscapes and TartanAir, while RollingDepth uses ScanNet/Bonn/DyDToF among its video evaluations. ([arXiv][22])

These datasets are useful for:

$$
\text{ordinary physical scenes}
$$

so that our model doesn't become specialized only to optical anomalies.

### Priority

**★★★☆☆**

---

# But we still need a NEW dataset

This is the most important conclusion.

None of the datasets above gives us:

$$
\boxed{
\begin{aligned}
&\text{matched competing physical explanations}\\
+&\text{controlled observer interventions}\\
+&\text{ground-truth physical contact geometry}\\
+&\text{causal image-formation labels}\\
+&\text{resolvable/non-resolvable labels}\\
+&\text{minimum resolving camera motion}.
\end{aligned}}
$$

That is why **Intervene3D-Bench** needs to be a contribution of the paper. Our prior review similarly concluded that matched counterfactual scenes, rather than simply collecting more mirrors and windows, are essential. 

# Intervene3D-Bench — what we must collect

Every underlying scene should have multiple **causal variants**.

| Scene variant    | Meaning                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| **Direct**       | Actual physical 3D scene                                                         |
| **Reflection**   | Same apparent scene through mirror reflection                                    |
| **Transmission** | Same scene viewed through glass                                                  |
| **Display**      | Same scene displayed on monitor/projector                                        |
| **Planar image** | Printed/photo version                                                            |
| **Mixed**        | reflection + transmission                                                        |
| **Hard display** | viewpoint-responsive / head-tracked display                                      |
| **Unresolvable** | competing hypotheses intentionally indistinguishable within allowed action range |

The important property is:

$$
I_A(C_0)\approx I_B(C_0)
$$

at the initial pose.

Otherwise a single-image classifier can solve the task and the whole paper loses its motivation.

---

# Required camera trajectories

For every scene, capture systematic interventions such as:

$$
x=
\{-30,-20,-10,0,+10,+20,+30\}\text{ cm}
$$

and similarly:

* lateral translation;
* vertical translation;
* forward/backward translation;
* yaw;
* pitch;
* limited roll.

The **same calibrated trajectories** should be applied across matched causal variants.

Then we can determine:

$$
B_{\min}
$$

or the Minimum Causal Resolving Baseline.

---

# Ground-truth fields for every sequence

I would store at minimum:

| Annotation                              | Why                                             |
| --------------------------------------- | ----------------------------------------------- |
| RGB                                     | observation                                     |
| camera intrinsics \(K_t\)               | geometric intervention definition               |
| camera pose \(T_t\)                     | exact action                                    |
| \(D_{\text{contact}}\)                  | first physically interactable surface           |
| \(D_{\text{apparent}}\) where available | apparent geometric interpretation               |
| optical mechanism \(H\)                 | direct/reflection/transmission/display/mixed    |
| mechanism mask                          | where ambiguity occurs                          |
| surface normals                         | physical geometry                               |
| physical object IDs                     | temporal identity                               |
| material type                           | generalization analysis                         |
| resolvability label                     | whether \(\mathcal A\) distinguishes hypotheses |
| \(B_{\min}\)                            | minimum resolving baseline                      |
| hypothesis pair ID                      | matched counterfactual relation                 |

For synthetic data also record the **full ray interaction chain**:

$$
Camera
\rightarrow
Interface
\rightarrow
Reflection/Transmission
\rightarrow
Source.
$$

That will be extremely valuable.

---

# Recommended scale

For a CVPR-first version, I would target:

### Synthetic

$$
10,000-30,000
$$

base scenes.

Each with roughly:

$$
3-5
$$

causal variants and:

$$
8-20
$$

camera views/interventions.

That easily gives hundreds of thousands to millions of frames.

### Real

Approximately:

$$
300-800
$$

carefully controlled real sequences.

I would prefer **400 excellent matched sequences** over 5,000 loosely collected videos.

The novelty is in the **experimental control**, not raw scale.

---

# Recommended training/evaluation usage

A practical structure would be:

| Dataset                           |                         Train | Validation | External test |
| --------------------------------- | ----------------------------: | ---------: | ------------: |
| TransPhy3D                        |                             ✓ |          ✓ |             ✓ |
| LayeredDepth synthetic            |                             ✓ |          ✓ |             — |
| LayeredDepth real                 |                             — |          — |             ✓ |
| DepthFocus synthetic if available |                             ✓ |          ✓ |             — |
| DREDS                             |                             ✓ |          ✓ |             ✓ |
| Intervene3D synthetic             |                         **✓** |      **✓** |         **✓** |
| Intervene3D real                  |                       limited |          ✓ | **main test** |
| Mirror3D                          |            optional fine-tune |          — |             ✓ |
| MAGD                              |                             — |          — |             ✓ |
| MVMD                              |                             — |          — |             ✓ |
| 3D Visual Illusion                | limited/no training initially |          — |         **✓** |
| ClearPose                         |                             — |          — |             ✓ |
| Booster                           |                             — |          — |             ✓ |
| MD-3k                             |                             — |          — |             ✓ |
| GIFT benchmark                    |                             — |          — |             ✓ |

The strongest experimental design is to **avoid training on several of the real benchmarks**.

Then you can demonstrate:

$$
\boxed{\text{zero-shot causal/physical generalization}}
$$

rather than dataset memorization.

---

# The five datasets I would secure first

If we start implementation now, my priority order is:

1. **TransPhy3D** — video + transparent/reflective + perfect depth/normals.
2. **LayeredDepth** — explicit multilayer geometry.
3. **3D Visual Illusion dataset** — fake/apparent 3D geometry.
4. **Mirror3D** — physical mirror-plane ground truth.
5. **ClearPose / DREDS** — real and synthetic difficult transparent objects.

Then add:

**MAGD + MVMD + Booster + MD-3k + GIFT benchmark**

for external testing.

And simultaneously start building:

$$
\boxed{\textbf{Intervene3D-Bench}}
$$

because **that new benchmark is what converts this from “combine existing methods” into a new research problem**. Existing data separately covers multilayer depth, mirror geometry, transparency, illusions, motion and world-model consistency, but none currently supplies the matched, controlled **“are these physical explanations distinguishable under action set \(\mathcal A\)?”** supervision that our final formulation needs. 

If you want, I can next turn this into a **proper Related Work matrix of ~30 core papers with columns: task, input, output, dataset, limitation, overlap with Intervene3D, and exact novelty gap**—which would be the most useful format for starting the actual CVPR 2027 paper.

[1]: https://openaccess.thecvf.com/content/ICCV2025/html/Wen_Seeing_and_Seeing_Through_the_Glass_Real_and_Synthetic_Data_ICCV_2025_paper.html?utm_source=chatgpt.com "ICCV 2025 Open Access Repository"
[2]: https://openaccess.thecvf.com/content/CVPR2026/html/Wen_SeeGroup_Multi-Layer_Depth_Estimation_of_Transparent_Surfaces_via_Self-Determined_Grouping_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[3]: https://arxiv.org/abs/2606.29600?utm_source=chatgpt.com "One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models"
[4]: https://arxiv.org/abs/2608.02068?utm_source=chatgpt.com "GIFT: Geometry-Invariant Fine-Tuning for Non-Lambertian Monocular Depth Estimation"
[5]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/70c1e653e0594c78897c13b423b7c919-Abstract-Conference.html?utm_source=chatgpt.com "3D Visual Illusion Depth Estimation"
[6]: https://arxiv.org/abs/2106.06629?utm_source=chatgpt.com "Mirror3D: Depth Refinement for Mirror Surfaces"
[7]: https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Depth-Aware_Mirror_Segmentation_CVPR_2021_paper.pdf?utm_source=chatgpt.com "Depth-Aware Mirror Segmentation"
[8]: https://arxiv.org/abs/2603.26181?utm_source=chatgpt.com "GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport"
[9]: https://arxiv.org/abs/2410.01614?utm_source=chatgpt.com "Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization"
[10]: https://arxiv.org/abs/2512.23705?utm_source=chatgpt.com "Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation"
[11]: https://arxiv.org/abs/2511.16993?utm_source=chatgpt.com "DepthFocus: Controllable Depth Estimation for See-Through Scenes"
[12]: https://www.cs.cityu.edu.hk/~rynson/papers/cvpr24d.pdf?utm_source=chatgpt.com "This CVPR paper is the Open Access version, provided by the Computer Vision Foundation."
[13]: https://openaccess.thecvf.com/content/WACV2025/html/Shen_MVMD_A_Multi-View_Approach_for_Enhanced_Mirror_Detection_WACV_2025_paper.html?utm_source=chatgpt.com "WACV 2025 Open Access Repository"
[14]: https://ojs.aaai.org/index.php/AAAI/article/view/38023?utm_source=chatgpt.com "Video Mirror Detection with the Motion-in-Depth Cue | Proceedings of the AAAI Conference on Artificial Intelligence"
[15]: https://link.springer.com/article/10.1007/s11042-024-19627-5?utm_source=chatgpt.com "Key points trajectory and multi-level depth distinction based refinement for video mirror and glass segmentation | Multimedia Tools and Applications | Springer Nature Link"
[16]: https://arxiv.org/abs/2202.13263?utm_source=chatgpt.com "Next-Best-View Prediction for Active Stereo Cameras and Highly Reflective Objects"
[17]: https://arxiv.org/abs/2306.01828?utm_source=chatgpt.com "Unifying (Machine) Vision via Counterfactual World Modeling"
[18]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03523.pdf?utm_source=chatgpt.com "Understanding Physical Dynamics with Counterfactual World Modeling"
[19]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3e6c9a8f3863f0343bf5cbabf1f85ca6-Abstract-Conference.html?utm_source=chatgpt.com "Unified 3D Scene Understanding Through Physical World Modeling"
[20]: https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Perceptual_3D_Simulation_With_Physical_World_Modeling_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[21]: https://openaccess.thecvf.com/content/CVPR2026/html/Venkatesh_Physical_Object_Understanding_with_a_Physically_Controllable_World_Model_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[22]: https://arxiv.org/abs/2603.12655?utm_source=chatgpt.com "VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model"
[23]: https://openaccess.thecvf.com/content/CVPR2026/html/Fang_WorldReel_4D_Video_Generation_with_Consistent_Geometry_and_Motion_Modeling_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[24]: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_WorldStereo_Bridging_Camera-Guided_Video_Generation_and_Scene_Reconstruction_via_3D_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[25]: https://openaccess.thecvf.com/content/CVPR2026/html/Yang_NeoVerse_Enhancing_4D_World_Model_with_in-the-wild_Monocular_Videos_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[26]: https://arxiv.org/abs/2608.04575?utm_source=chatgpt.com "PhysMind: From Video to Executable Worlds for Training-Free Physical Reasoning"
[27]: https://arxiv.org/abs/2608.23486?utm_source=chatgpt.com "GeoWAM: Visual Geometry World Action Models for Autonomous Driving"
[28]: https://arxiv.org/abs/2202.03169?utm_source=chatgpt.com "CITRIS: Causal Identifiability from Temporal Intervened Sequences"
[29]: https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kouros_Unveiling_the_Ambiguity_in_Neural_Inverse_Rendering_A_Parameter_Compensation_CVPRW_2024_paper.html?utm_source=chatgpt.com "CVPR 2024 Open Access Repository"
[30]: https://arxiv.org/abs/2607.09329?utm_source=chatgpt.com "Dynamic Inverse Rendering for Enhanced Material-Lighting Decomposition"
[31]: https://openaccess.thecvf.com/CVPR2025?day=2025-06-15&utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[32]: https://openaccess.thecvf.com/content/CVPR2025/html/Hu_DepthCrafter_Generating_Consistent_Long_Depth_Sequences_for_Open-world_Videos_CVPR_2025_paper.html?utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[33]: https://openaccess.thecvf.com/content/CVPR2025/html/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.html?utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[34]: https://openaccess.thecvf.com/content/CVPR2025/html/Ke_Video_Depth_without_Video_Models_CVPR_2025_paper.html?utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[35]: https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Align3R_Aligned_Monocular_Depth_Estimation_for_Dynamic_Videos_CVPR_2025_paper.html?utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[36]: https://openaccess.thecvf.com/content/CVPR2025/html/Kuang_Buffer_Anytime_Zero-Shot_Video_Depth_and_Normal_from_Image_Priors_CVPR_2025_paper.html?utm_source=chatgpt.com "CVPR 2025 Open Access Repository"
[37]: https://arxiv.org/abs/2502.20110?utm_source=chatgpt.com "UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler"
[38]: https://arxiv.org/abs/2507.02546?utm_source=chatgpt.com "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details"
[39]: https://mlanthology.org/neurips/2025/li2025neurips-cameras/?utm_source=chatgpt.com "Cameras as Relative Positional Encoding | ML Anthology"
[40]: https://neurips.cc/virtual/2025/loc/san-diego/poster/115793?utm_source=chatgpt.com "NeurIPS Poster Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers"
[41]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/30bc3a3a44c9d2e3f32e6dd1cd18f552-Abstract-Conference.html?utm_source=chatgpt.com "SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models"
[42]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/6a0480190bbe6b622c7f1d3aa9be9c0f-Abstract-Conference.html?utm_source=chatgpt.com "EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes"
[43]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/821d20219c2f14850af1b5220f0ed13f-Abstract-Conference.html?utm_source=chatgpt.com "RaySt3R: Predicting Novel Depth Maps for Zero-Shot Object Completion"
[44]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/21f1c5bbf2519321c1bee9bfa9edcd46-Abstract-Conference.html?utm_source=chatgpt.com "PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding"
[45]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8a7495d84596b831dbdd938453bd3697-Abstract-Conference.html?utm_source=chatgpt.com "Perceive Anything: Recognize, Explain, Caption, and Segment Anything in Images and Videos"
[46]: https://arxiv.org/abs/2605.15185?utm_source=chatgpt.com "Quantitative Video World Model Evaluation for Geometric-Consistency"
[47]: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[48]: https://openaccess.thecvf.com/content/CVPR2021/papers/Tan_Mirror3D_Depth_Refinement_for_Mirror_Surfaces_CVPR_2021_paper.pdf?utm_source=chatgpt.com "Mirror3D: Depth Refinement for Mirror Surfaces"
[49]: https://openaccess.thecvf.com/content/WACV2025/papers/Shen_MVMD_A_Multi-View_Approach_for_Enhanced_Mirror_Detection_WACV_2025_paper.pdf?utm_source=chatgpt.com "MVMD: A Multi-View Approach for Enhanced Mirror Detection"
[50]: https://openaccess.thecvf.com/content/CVPR2026/supplemental/Min_DepthFocus_Controllable_Depth_CVPR_2026_supplemental.pdf?utm_source=chatgpt.com "Supplementary Materials"
[51]: https://arxiv.org/abs/2203.03890?utm_source=chatgpt.com "ClearPose: Large-scale Transparent Object Dataset and Benchmark"
[52]: https://github.com/PKU-EPIC/DREDS?utm_source=chatgpt.com "GitHub - PKU-EPIC/DREDS: [ECCV 2022] Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects · GitHub"
[53]: https://arxiv.org/abs/2301.08245?utm_source=chatgpt.com "Booster: a Benchmark for Depth from Images of Specular and Transparent Surfaces"

