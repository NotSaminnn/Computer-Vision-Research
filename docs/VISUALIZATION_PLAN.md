# Visualization Plan

**Rule:** no `matplotlib` configuration exists anywhere outside
`intervene3d/visualization/ieee_style.py`, and no figure is ever drawn from
numbers held only in memory.

---

## 1. The IEEE style system

`ieee_style.py` is the single source of typography, sizing and colour.

- **Base style:** SciencePlots `["science", "ieee", "no-latex"]` when available.
  The `no-latex` variant is essential — the plain `science` style sets
  `text.usetex=True`, which fails on any machine without a LaTeX installation.
  A self-contained fallback keeps figure generation working either way, and
  `style_provenance()` records which path was taken into every `metrics.json`.
- **Column widths:** `IEEE_SINGLE_COLUMN_IN = 3.5"`, `IEEE_DOUBLE_COLUMN_IN = 7.16"`,
  declared through `FigureSpec(column=..., height_ratio=...)`.
- **Typography:** serif, 8 pt body / 7 pt ticks / 5–6 pt legends; `pdf.fonttype=42`
  so text stays selectable and editable in the PDF.
- **Output:** vector PDF **and** 400 dpi PNG for every figure, `bbox_inches="tight"`.
- **Palette:** Paul Tol's **muted** qualitative scheme
  (`#3D4A52 #4477AA #CC6677 #117733 #AA4499 #88CCEE #DDCC77`), which replaced the
  saturated Okabe–Ito set: dense, high-chroma fills read as slide graphics rather
  than journal figures, and Tol muted keeps the colour-vision-deficiency safety
  while sitting far quieter on the page. Ordered so adjacent series stay separable.
- **Grayscale-safe:** every line series is distinguished by **colour, marker and
  linestyle** simultaneously. Enforced by `test_series_styles_are_grayscale_safe`.
- **Bar charts no longer hatch.** Dense hatching prints as noise at journal
  scale, so grouped bars now separate by **lightness** instead:
  `bar_style(index, colour=None, emphasis=False)` returns a light fill against a
  full-strength edge of the same hue, built on `tint(colour, amount)`, which
  mixes a colour toward white. Lightness survives greyscale reproduction exactly
  as hatching did, without the texture. `emphasis=True` tints less (0.30 rather
  than 0.62), which is how the method under test is set apart from its baselines.
- **Mathematical notation:** mathtext with the `dejavuserif` fontset, so
  `$\mathcal{I}_{\mathcal{A}}(H_i,H_j)$` renders without LaTeX. Tested.
- **Stable semantics:** `MECHANISM_COLORS` fixes one colour per optical mechanism
  across every figure; `METHOD_SHORT_NAMES` gives compact legend labels while the
  data files keep the long explicit keys.

## 2. Module layout

```
visualization/
├── ieee_style.py         the only rcParams in the repository
├── export.py             save_figure (PDF + PNG), figure_index
├── geometry_plots.py     contact vs apparent, landmark views, error maps, renders
├── ambiguity_plots.py    initial-view similarity, separability matrix, posteriors,
│                         resolvability distribution, uncertainty decomposition
├── intervention_plots.py separability vs baseline, action utility, trajectory,
│                         regret, motion cost, predicted vs observed
├── metric_plots.py       CEA, ROC, FPCR, contact depth, MCRB validation and error,
│                         summary table
├── ablation_plots.py     ablation grid, generalisation curves
└── pipeline_figure.py    the conceptual overview, matched-variant strip
```

One `matplotlib` reference lives outside this package:
`data/synthetic/dataset_writer._save_preview` calls `matplotlib.use("Agg")` and
`image.imsave` to write a preview PNG without adding a Pillow dependency. It
sets no `rcParams`, draws no axes and produces no figure, so the rule above
still holds — but it is the one exception and is named here rather than left to
be discovered.

## 3. Data flow

```
experiment  →  metrics/figure_data.json  →  generate_figures()  →  figures/*.{pdf,png}
```

`build_figure_data` produces a plain JSON dictionary; `generate_figures` reads
*only* that dictionary. The separation is what makes
`scripts/generate_all_figures.py` able to rebuild every figure from a finished run
without recomputing anything — and it makes hand-transcribing a number into a plot
structurally impossible.

## 4. The figure set (all 22 produced per run)

| # | Figure | Family | Communicates |
|---|---|---|---|
| 01 | pipeline overview | conceptual | the required chain: same apparent geometry → hypotheses → intervention → predicted consequences → observation → belief update → **resolved OR abstained** |
| 02 | initial-view similarity | ambiguity | `Δ_ij` at `C_0` (zero) vs after `a*` — the premise, measured |
| 03 | matched variants | conceptual | the same base scene across mechanisms, before and after one action |
| 04 | separability vs baseline | intervention | `Δ_ij(a(B))` per pair, with the MCRB marked |
| 05 | separability matrix | ambiguity | `I_A(H_i,H_j)`; entries below `ε` in red |
| 06 | action utility | intervention | utility of every candidate; selected vs oracle-optimal |
| 07 | camera trajectory | intervention | the bounded action set and the executed intervention |
| 08 | hypothesis probabilities | ambiguity | posterior before and after; abstention flagged |
| 09 | predicted vs observed | intervention | each hypothesis' predicted response against what was observed |
| 10 | landmark views | geometry | reference vs post-intervention, by channel |
| 11 | explanation accuracy | metrics | CEA per method, overall and per mechanism, against chance |
| 12 | identifiability ROC | metrics | resolvability prediction, all methods |
| 13 | FPCR | metrics | false physical certainty on non-identifiable cases |
| 14 | resolvability distribution | ambiguity | `I_A` split by ground-truth label — shows the benchmark is not trivially solvable |
| 15 | uncertainty decomposition | ambiguity | `U_prediction` vs `U_identifiability` — the paper's key distinction |
| 16 | contact depth error | metrics | AbsRel and RMSE per method |
| 17 | contact vs apparent | geometry | same appearance, different physical surface |
| 18 | MCRB theory validation | metrics | `1/B_min` vs `f\|1/Z₁−1/Z₂\|` with the fit and `R²` |
| 19 | MCRB error | metrics | predicted vs ground-truth resolving baseline |
| 20 | intervention regret | intervention | distance from the oracle-optimal experiment |
| 21 | motion cost | intervention | camera motion spent per scene |
| 22 | metric summary | metrics | the headline table as a figure |

Ablation and generalisation plot families (`plot_ablation_grid`,
`plot_generalisation`) are implemented and unit-tested but still **not wired into
any run** — nothing in `experiments/` calls them. The action-noise generalisation
study has since been executed (`docs/EXPERIMENT_PLAN.md` E8), so the data for
`plot_generalisation` now exists; the wiring does not.

The four external scripts (`train_transition.py`, `evaluate_external.py`,
`evaluate_identifiability.py`, `evaluate_conformal.py`) **produce no figures at
all** — they create the `figures/` directory in their run folder and leave it
empty, reporting through `metrics/metrics.json`, a per-image CSV and
`summary.md`. Their results reach the paper as tables. Figures for the external
work are not implemented.

## 5. Defects found and fixed, 2026-08-29

Four figures were found wrong on visual inspection, and all four were wrong in
the same way: the generator ran without error and the picture still did not
communicate — which is precisely what a passing test suite cannot catch. Each
fix now carries a regression test in `tests/unit/test_visualization.py`. This is
a record of four specific defects, not a claim that the remaining eighteen
figures have been audited to the same standard.

**fig02 — initial-view similarity.** Two defects at once. The legend was drawn
inside the axes, where it covered the leftmost bar *and* the `ε` reference line —
so the figure hid both the premise and the threshold. It now sits above the axes
(`bbox_to_anchor=(0.5, 1.06)`, `frameon=False`) with the title padded clear.
Separately, the reference-view bars are **exactly zero** — that is the premise of
the project — and zero has no position on a log axis, so clamping them to the
floor drew nothing at all and the figure showed only the post-action bars. They
are now drawn as a **visible stub at the floor, annotated `0`**, so "identical at
`C_0`" is something the reader sees rather than infers from an absence.

**fig05 — separability matrix.** Cell annotations were hardcoded white. `viridis`
runs from near-black to bright yellow, so a white label is invisible on every
high-value cell — and the high-value cells are exactly the resolvable pairs the
figure exists to show. Ink is now chosen per cell from its own luminance
(`_prefers_dark_ink`, a WCAG relative-luminance comparison), and because
viridis' mid-range caps at roughly 4.3:1 — short of WCAG AA whichever ink is
picked — every label additionally carries a thin halo in the opposite tone.
The diagonal was left blank, which reads as missing data rather than as an
undefined quantity; it is now tinted and explicitly marked `—`, with the title
saying `— = self`. The red "unresolvable" convention is explained in the title
**only when a red cell is actually present**, since a legend for something absent
from the plot is noise.

**fig22 — metric summary table.** The canvas was sized `0.30 + 0.06 × n_methods`,
which for nine methods is a six-inch-tall figure holding a two-inch table, and
`loc="center"` then stranded it: about 95 % of the saved image was empty. Height
now follows the **row count** and the table is placed with
`bbox=[0, 0, 1, 1]` so it fills the axes.

**fig03 — matched-variant strip.** Colour there encodes the **channel**
(content / frame / marker), not the mechanism — the mechanism is the column — and
there was no legend saying so, which invited exactly the wrong reading. A figure
legend now labels the channel encoding.

The tests are deliberately behavioural rather than golden-image: they assert
that the chosen ink is the better of the two on every point along viridis *and*
that the old always-white policy genuinely fails; that a nine-row table needs
under four inches and carries ink in both the top and bottom eighths of the
canvas; and that the strip and every other plot family still render.

## 6. Standards applied to every figure

- axis labels carry units (`px`, `m`, `px m⁻¹`) and mathematical symbols matching
  the specification;
- `ε` and `τ` are drawn as annotated reference lines wherever relevant;
- error bars are 95 % CIs when aggregated across seeds, and their absence is
  explicit rather than implied;
- a zero-height bar is annotated `0` rather than left as a mystery gap;
- legends move outside the axes when they would otherwise cover data;
- symlog axes are clipped at zero when the quantity is non-negative;
- no chartjunk: no 3-D bars, no gradients, no decorative colour.
