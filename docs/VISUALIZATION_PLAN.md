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
- **Grayscale-safe:** every series is distinguished by **colour, marker and
  linestyle** simultaneously, and bar charts add hatching. Enforced by
  `test_series_styles_are_grayscale_safe`.
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
`plot_generalisation`) are implemented and unit-tested but not yet wired into a
run, because the corresponding experiments have not been executed.

## 5. Standards applied to every figure

- axis labels carry units (`px`, `m`, `px m⁻¹`) and mathematical symbols matching
  the specification;
- `ε` and `τ` are drawn as annotated reference lines wherever relevant;
- error bars are 95 % CIs when aggregated across seeds, and their absence is
  explicit rather than implied;
- a zero-height bar is annotated `0` rather than left as a mystery gap;
- legends move outside the axes when they would otherwise cover data;
- symlog axes are clipped at zero when the quantity is non-negative;
- no chartjunk: no 3-D bars, no gradients, no decorative colour.
